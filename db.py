"""
ระบบเก็บข้อมูลถาวรบน Neon Postgres (แทน ChromaDB + ไฟล์ JSON ในเครื่อง)
ใช้เฉพาะกับ app.py (เว็บ Streamlit) เท่านั้น — ไม่แตะ rag.py/line_bot.py
เพื่อให้ LINE bot เดิมยังทำงานกับ ChromaDB ได้ตามปกติ

ต้องมี extension "vector" (pgvector) เปิดใช้ได้บน Neon (โค้ดนี้เปิดให้อัตโนมัติตอน init_db)
"""
import os
import time
import shutil
import hashlib
import zipfile
import tempfile
import json
import uuid
import base64
import hmac
import secrets
import re
import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector import Vector
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types

# ใช้ฟังก์ชันอ่านไฟล์/ตัด chunk ตัวเดียวกับ rag.py (ไม่ผูกกับ ChromaDB เลย เอามาใช้ซ้ำได้ปลอดภัย)
from rag import read_file, chunk_text, build_chunks, iter_chunks, SUPPORTED_EXTS, ZIP_EXT
from embedding_service import get_embedding_provider, LOCAL_DIMENSION, LOCAL_PROFILE
from retrieval_utils import (
    extract_retrieval_terms,
    lexical_relevance,
    normalize_retrieval_query,
    retrieval_topic_key,
)

EMBED_DIM = 768  # pgvector index (hnsw/ivfflat) รองรับสูงสุด 2000 มิติ เลยตัด gemini-embedding-001 ลงมาที่ 768
EMBED_BATCH_SIZE = 100  # Gemini embedContent รับได้สูงสุด 100 รายการต่อ batch

_database_url = None
_genai_client = None
_pool = None
_schema_ready = False
_POOL_MAXCONN = 5


def init_db(database_url, api_key):
    """เรียกทุกครั้งที่ Streamlit rerun สคริปต์ แต่ตั้งค่าจริง (schema + connection pool) แค่ครั้งแรก
    ครั้งเดียวต่อการรันเซิร์ฟเวอร์ (ตัวแปร module-level พวกนี้อยู่ยาวข้าม rerun/session ในเครื่องเดียวกัน)"""
    global _database_url, _genai_client, _pool, _schema_ready
    _database_url = database_url
    _genai_client = genai.Client(api_key=api_key)

    if not _schema_ready:
        _setup_schema()
        _schema_ready = True

    if _pool is None:
        # ใช้ pool แทนการเปิด connection ใหม่ทุกครั้ง — ลด latency มหาศาล เพราะแต่ละครั้งที่
        # เปิด connection ใหม่ต้องทำ TCP+SSL handshake ไปหา Neon ใหม่ทุกรอบ (ช้ามากถ้าระยะทางไกล)
        _pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=_POOL_MAXCONN, dsn=database_url)


class _PooledConn:
    """context manager: ยืม connection จาก pool มาใช้ แล้วคืนกลับ pool ตอนจบ

    Neon free tier auto-suspend หลังไม่มีคนใช้ ~5 นาที ทำให้ connection ที่ค้างอยู่ใน pool
    ตายไปเงียบๆ ฝั่ง server โดยที่ pool ฝั่งเราไม่รู้ตัว ถ้าไม่เช็ค/ทิ้งให้ถูกต้อง connection
    ที่ตายจะรั่วออกจาก pool ถาวร (getconn สำเร็จแต่ใช้ไม่ได้ → error → __enter__ ไม่จบ →
    __exit__ ไม่ถูกเรียก → ไม่มีใครคืน connection กลับ pool) พอรั่วครบ maxconn ตัว pool จะ
    "exhausted" ค้างตลอดไปจนกว่าจะ restart แอป โค้ดข้างล่างนี้เลย retry + ทิ้ง connection
    ที่ตายแบบชัดเจน (close=True) แทนที่จะปล่อยรั่ว"""

    def __enter__(self):
        last_err = None
        for _ in range(_POOL_MAXCONN):
            conn = _pool.getconn()
            try:
                register_vector(conn)  # แถมเป็นการเช็คว่า connection นี้ยังใช้ได้จริงในตัว
                self._conn = conn
                return self._conn
            except Exception as e:
                last_err = e
                try:
                    _pool.putconn(conn, close=True)  # ทิ้ง connection ที่ตายแล้วให้ pool สร้างใหม่แทน
                except Exception:
                    pass
        raise last_err

    def __exit__(self, exc_type, exc, tb):
        close_it = False
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:
                close_it = True  # rollback เองยังพัง แปลว่า connection นี้ตายแล้ว ทิ้งไปเลย
        try:
            _pool.putconn(self._conn, close=close_it)
        except Exception:
            pass
        return False


def _get_conn():
    """ยืม connection จาก pool (คืนกลับอัตโนมัติเมื่อออกจาก with block)"""
    return _PooledConn()


def _get_vector_conn():
    """เหมือน _get_conn() เอาไว้ให้โค้ดอ่านง่ายว่าจุดนี้ผูกค่า embedding เป็นพารามิเตอร์โดยตรง"""
    return _PooledConn()


def _setup_schema():
    """สร้างตาราง/extension ที่ยังไม่มี รันซ้ำได้ปลอดภัย (CREATE ... IF NOT EXISTS ทั้งหมด)
    ใช้ connection เปล่าๆ ตรงๆ (ยังไม่มี pool/ยังไม่ register_vector) เพราะตอนนี้ extension "vector"
    อาจยังไม่มีอยู่ในฐานข้อมูลเลยด้วยซ้ำ"""
    conn = psycopg2.connect(_database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    slug TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    topic_slug TEXT NOT NULL REFERENCES topics(slug) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (topic_slug, filename)
                );
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    topic_slug TEXT NOT NULL REFERENCES topics(slug) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({EMBED_DIM}) NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS documents_topic_idx ON documents(topic_slug);")
            cur.execute("CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(topic_slug, source);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING hnsw (embedding vector_cosine_ops);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),
                    active BOOLEAN NOT NULL DEFAULT true,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    filename TEXT PRIMARY KEY,
                    title TEXT,
                    pinned BOOLEAN NOT NULL DEFAULT false,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    chat_filename TEXT NOT NULL REFERENCES chats(filename) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT,
                    seq INT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS chat_messages_chat_idx ON chat_messages(chat_filename, seq);")
            # Older versions rewrote the whole chat and did not enforce sequence
            # uniqueness. Keep the oldest copy if a previous concurrent write
            # produced duplicate sequence numbers, then enforce the invariant.
            cur.execute("""
                DELETE FROM chat_messages newer
                USING chat_messages older
                WHERE newer.chat_filename = older.chat_filename
                  AND newer.seq = older.seq
                  AND newer.id > older.id
            """)
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS chat_messages_chat_seq_unique "
                "ON chat_messages(chat_filename, seq)"
            )

            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    topic TEXT,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("ALTER TABLE chats ADD COLUMN IF NOT EXISTS owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE")
            cur.execute("ALTER TABLE chats ADD COLUMN IF NOT EXISTS active_topic_slug TEXT REFERENCES topics(slug) ON DELETE SET NULL")
            cur.execute("ALTER TABLE chats ADD COLUMN IF NOT EXISTS active_topic_name TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS chats_owner_idx ON chats(owner_id, pinned, filename)")
            cur.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS knowledge_version BIGINT NOT NULL DEFAULT 1")
            cur.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS embedding_profile TEXT NOT NULL DEFAULT %s", (LOCAL_PROFILE,))
            cur.execute(
                "ALTER TABLE qa_cache ADD COLUMN IF NOT EXISTS sources JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents_v2 (
                    id BIGSERIAL PRIMARY KEY,
                    topic_slug TEXT NOT NULL REFERENCES topics(slug) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    location_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding_profile TEXT NOT NULL,
                    embedding VECTOR({LOCAL_DIMENSION}) NOT NULL,
                    UNIQUE(topic_slug, source, chunk_index, embedding_profile)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS documents_v2_source_idx ON documents_v2(topic_slug, source)")
            cur.execute("CREATE INDEX IF NOT EXISTS documents_v2_hash_profile_idx ON documents_v2(content_hash, embedding_profile)")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_v2_embedding_idx
                ON documents_v2 USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id UUID PRIMARY KEY,
                    topic_slug TEXT NOT NULL REFERENCES topics(slug) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed')),
                    total_chunks INT NOT NULL DEFAULT 0,
                    completed_chunks INT NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS ingestion_job_chunks (
                    job_id UUID NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    location_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding VECTOR({LOCAL_DIMENSION}),
                    PRIMARY KEY (job_id, chunk_index)
                )
            """)
        conn.commit()
    finally:
        conn.close()


# ===== embedding =====
def embed(texts):
    """Create embeddings locally in the hosted container (no Gemini quota)."""
    return get_embedding_provider().embed_documents(texts)


# ===== หัวข้อความรู้ (topics) =====
def create_topic(display_name):
    display_name = display_name.strip()
    if not display_name:
        raise ValueError("ต้องตั้งชื่อหัวข้อ")

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM topics WHERE lower(name) = lower(%s)", (display_name,))
            row = cur.fetchone()
            if row:
                return row[0]

            slug = "topic_" + hashlib.md5(display_name.encode("utf-8")).hexdigest()[:10]
            cur.execute("INSERT INTO topics (slug, name) VALUES (%s, %s)", (slug, display_name))
        conn.commit()
    return slug


def list_topics():
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT slug, name FROM topics ORDER BY name")
            rows = cur.fetchall()
    return [{"slug": r[0], "name": r[1]} for r in rows]


def rename_topic(topic_slug, new_name):
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("ต้องตั้งชื่อหัวข้อ")
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE topics SET name = %s WHERE slug = %s", (new_name, topic_slug))
        conn.commit()


def delete_topic(topic_slug):
    """ลบหัวข้อทั้งหมด: sources/documents ลบตามด้วย ON DELETE CASCADE"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM topics WHERE slug = %s", (topic_slug,))
        conn.commit()


# ===== เอกสาร/ไฟล์ในแต่ละหัวข้อ =====
def _upsert_source_text(cur, topic_slug, filename, text):
    cur.execute(
        "INSERT INTO sources (topic_slug, filename, full_text) VALUES (%s, %s, %s) "
        "ON CONFLICT (topic_slug, filename) DO UPDATE SET full_text = EXCLUDED.full_text",
        (topic_slug, filename, text),
    )


def _insert_chunks(cur, topic_slug, filename, chunks, embeddings):
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            "INSERT INTO documents (topic_slug, source, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            (topic_slug, filename, i, chunk, Vector(emb)),
        )


def add_document(topic_slug, path, filename, progress_callback=None):
    """Extract and locally embed a document, publishing it atomically.

    Old chunks remain searchable until every new vector is ready.  Job state is
    persisted so the UI can report failures without losing the prior source.
    """
    if os.path.splitext(filename)[1].lower() == ZIP_EXT:
        return _add_zip(topic_slug, path)

    job_id = str(uuid.uuid4())
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingestion_jobs (id, topic_slug, filename, status, total_chunks) "
                "VALUES (%s, %s, %s, 'pending', %s)",
                (job_id, topic_slug, filename, 0),
            )
        conn.commit()
    try:
        staged = 0
        batch = []
        for index, chunk in enumerate(iter_chunks(path)):
            batch.append((
                job_id, index, chunk["text"],
                hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
                psycopg2.extras.Json({"location": chunk["location"]}),
            ))
            if len(batch) >= 250:
                _stage_ingestion_chunks(job_id, batch)
                staged += len(batch)
                batch = []
        if batch:
            _stage_ingestion_chunks(job_id, batch)
            staged += len(batch)
        if staged == 0:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM ingestion_jobs WHERE id = %s", (job_id,))
                conn.commit()
            return 0
        return resume_ingestion_job(job_id, progress_callback=progress_callback)
    except Exception as exc:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingestion_jobs SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                    (str(exc)[:1000], job_id),
                )
            conn.commit()
        raise


def _stage_ingestion_chunks(job_id, rows):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "INSERT INTO ingestion_job_chunks "
                "(job_id, chunk_index, content, content_hash, location_metadata) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (job_id, chunk_index) DO NOTHING",
                rows,
                page_size=250,
            )
            cur.execute(
                "UPDATE ingestion_jobs SET total_chunks = "
                "(SELECT count(*) FROM ingestion_job_chunks WHERE job_id = %s), updated_at = now() WHERE id = %s",
                (job_id, job_id),
            )
            # Reuse a vector already produced for identical content. This makes
            # retries, duplicate files, and small document edits much faster.
            cur.execute(
                "UPDATE ingestion_job_chunks c SET embedding = ("
                "SELECT d.embedding FROM documents_v2 d "
                "WHERE d.content_hash = c.content_hash AND d.embedding_profile = %s LIMIT 1"
                ") WHERE c.job_id = %s AND c.embedding IS NULL AND EXISTS ("
                "SELECT 1 FROM documents_v2 d WHERE d.content_hash = c.content_hash AND d.embedding_profile = %s"
                ")",
                (LOCAL_PROFILE, job_id, LOCAL_PROFILE),
            )
            cur.execute(
                "UPDATE ingestion_jobs SET completed_chunks = "
                "(SELECT count(*) FROM ingestion_job_chunks WHERE job_id = %s AND embedding IS NOT NULL) "
                "WHERE id = %s",
                (job_id, job_id),
            )
        conn.commit()


def resume_ingestion_job(job_id, progress_callback=None):
    """Continue a staged ingestion job after interruption or process restart."""
    try:
        provider = None
        processed = 0
        total_chunks = 0
        while True:
            with _get_vector_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE ingestion_jobs SET status = 'running', error = NULL, updated_at = now() WHERE id = %s", (job_id,))
                    cur.execute(
                        "SELECT chunk_index, content FROM ingestion_job_chunks "
                        "WHERE job_id = %s AND embedding IS NULL ORDER BY chunk_index LIMIT 16",
                        (job_id,),
                    )
                    batch = cur.fetchall()
                    cur.execute("SELECT total_chunks, completed_chunks FROM ingestion_jobs WHERE id = %s", (job_id,))
                    progress_row = cur.fetchone() or (0, 0)
                    total_chunks = progress_row[0]
                conn.commit()
            if progress_callback:
                progress_callback(progress_row[1], progress_row[0], "embedding")
            if not batch:
                break
            if provider is None:
                provider = get_embedding_provider()
            vectors = provider.embed_documents([row[1] for row in batch])
            with _get_vector_conn() as conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(
                        cur,
                        "UPDATE ingestion_job_chunks SET embedding = %s WHERE job_id = %s AND chunk_index = %s",
                        [(Vector(vector), job_id, chunk_index) for (chunk_index, _), vector in zip(batch, vectors)],
                        page_size=16,
                    )
                    cur.execute(
                        "UPDATE ingestion_jobs SET completed_chunks = "
                        "(SELECT count(*) FROM ingestion_job_chunks WHERE job_id = %s AND embedding IS NOT NULL), "
                        "updated_at = now() WHERE id = %s",
                        (job_id, job_id),
                    )
                conn.commit()
            processed += len(batch)

        with _get_vector_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT topic_slug, filename FROM ingestion_jobs WHERE id = %s", (job_id,))
                job = cur.fetchone()
                if not job:
                    raise ValueError("ไม่พบ ingestion job")
                topic_slug, filename = job
                cur.execute(
                    "SELECT string_agg(content, E'\\n\\n' ORDER BY chunk_index) FROM ingestion_job_chunks WHERE job_id = %s",
                    (job_id,),
                )
                full_text = cur.fetchone()[0] or ""
                _upsert_source_text(cur, topic_slug, filename, full_text)
                cur.execute(
                    "DELETE FROM documents_v2 WHERE topic_slug = %s AND source = %s",
                    (topic_slug, filename),
                )
                cur.execute(
                    "INSERT INTO documents_v2 "
                    "(topic_slug, source, chunk_index, content, content_hash, location_metadata, embedding_profile, embedding) "
                    "SELECT %s, %s, chunk_index, content, content_hash, location_metadata, %s, embedding "
                    "FROM ingestion_job_chunks WHERE job_id = %s ORDER BY chunk_index",
                    (topic_slug, filename, provider.profile if provider else LOCAL_PROFILE, job_id),
                )
                cur.execute(
                    "UPDATE topics SET knowledge_version = knowledge_version + 1, embedding_profile = %s WHERE slug = %s",
                    (provider.profile if provider else LOCAL_PROFILE, topic_slug),
                )
                cur.execute("DELETE FROM qa_cache WHERE topic = %s", (topic_slug,))
                cur.execute(
                    "UPDATE ingestion_jobs SET status = 'completed', completed_chunks = total_chunks, updated_at = now() WHERE id = %s",
                    (job_id,),
                )
            conn.commit()
        if progress_callback:
            progress_callback(1, 1, "completed")
        return total_chunks
    except Exception as exc:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingestion_jobs SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                    (str(exc)[:1000], job_id),
                )
            conn.commit()
        raise


def get_source_text(topic_slug, filename):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT full_text FROM sources WHERE topic_slug = %s AND filename = %s",
                (topic_slug, filename),
            )
            row = cur.fetchone()
    return row[0] if row else ""


def delete_source_file(topic_slug, filename):
    """ลบไฟล์เดียวออกจากหัวข้อ: ลบทั้ง chunk และเนื้อหาต้นฉบับที่เก็บไว้"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE topic_slug = %s AND source = %s", (topic_slug, filename))
            cur.execute("DELETE FROM documents_v2 WHERE topic_slug = %s AND source = %s", (topic_slug, filename))
            cur.execute("DELETE FROM sources WHERE topic_slug = %s AND filename = %s", (topic_slug, filename))
            cur.execute("UPDATE topics SET knowledge_version = knowledge_version + 1 WHERE slug = %s", (topic_slug,))
            cur.execute("DELETE FROM qa_cache WHERE topic = %s", (topic_slug,))
        conn.commit()


def replace_source_file(topic_slug, path, filename):
    """Atomically replace a source; add_document removes old vectors only after embedding succeeds."""
    return add_document(topic_slug, path, filename)


def replace_source_text(topic_slug, filename, new_text):
    """Replace editable text atomically using the active local embedding profile."""
    new_text = new_text.strip()
    if not new_text:
        delete_source_file(topic_slug, filename)
        return 0

    chunks = chunk_text(new_text, size=450, overlap=60)
    provider = get_embedding_provider()
    embeddings = provider.embed_documents(chunks)
    with _get_vector_conn() as conn:
        with conn.cursor() as cur:
            _upsert_source_text(cur, topic_slug, filename, new_text)
            cur.execute("DELETE FROM documents_v2 WHERE topic_slug = %s AND source = %s", (topic_slug, filename))
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    "INSERT INTO documents_v2 "
                    "(topic_slug, source, chunk_index, content, content_hash, location_metadata, embedding_profile, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        topic_slug, filename, index, chunk,
                        hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                        psycopg2.extras.Json({"location": "เนื้อหา"}),
                        provider.profile, Vector(vector),
                    ),
                )
            cur.execute("UPDATE topics SET knowledge_version = knowledge_version + 1, embedding_profile = %s WHERE slug = %s", (provider.profile, topic_slug))
            cur.execute("DELETE FROM qa_cache WHERE topic = %s", (topic_slug,))
        conn.commit()
    return len(chunks)


def _source_exists(topic_slug, filename):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM sources WHERE topic_slug = %s AND filename = %s LIMIT 1",
                (topic_slug, filename),
            )
            return cur.fetchone() is not None


def _add_zip(topic_slug, zip_path):
    """แตกไฟล์ในซิปลง temp dir ชั่วคราว แล้วเพิ่มทุกไฟล์ที่รองรับข้างใน (กันเพิ่มซ้ำเหมือนไฟล์ทั่วไป)"""
    total = 0
    tmp_dir = tempfile.mkdtemp(prefix="cbot_zip_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                # ใช้แค่ basename กันไฟล์ในซิปเขียนออกนอก temp dir (zip-slip)
                inner_name = os.path.basename(member)
                inner_ext = os.path.splitext(inner_name)[1].lower()
                if not inner_name or inner_ext not in SUPPORTED_EXTS:
                    continue

                extract_path = os.path.join(tmp_dir, inner_name)
                with zf.open(member) as src, open(extract_path, "wb") as dst:
                    dst.write(src.read())

                if not _source_exists(topic_slug, inner_name):
                    total += add_document(topic_slug, extract_path, inner_name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return total


def search(topic_slug, query, top_k=4):
    """Compatibility formatter for callers that still expect one context string."""
    results = search_with_sources(topic_slug, query, top_k=top_k)
    return "\n\n---\n\n".join(
        f"[{item['citation_id']}] {item['source']} — {item['location']}\n{item['content']}"
        for item in results
    )


def _normalize_retrieval_query(query):
    return normalize_retrieval_query(query)


def _extract_retrieval_terms(query):
    return extract_retrieval_terms(query)


def _lexical_relevance(content, terms):
    return lexical_relevance(content, terms)


def _retrieval_topic_key(content):
    return retrieval_topic_key(content)


def search_with_sources(topic_slug, query, top_k=5, min_score=0.28, timing_out=None):
    """Return relevant chunks using semantic retrieval plus exact-term boosting."""
    provider = get_embedding_provider()
    normalized_query = _normalize_retrieval_query(query)
    # Preserve the user's natural wording for semantic similarity. Expanded terms
    # are used by the lexical branch below instead of diluting the query embedding.
    semantic_query = re.sub(r"\s+", " ", str(query or "").strip())
    embed_started = time.perf_counter()
    query_emb = provider.embed_query(semantic_query)
    embed_ms = (time.perf_counter() - embed_started) * 1000
    exact_terms = _extract_retrieval_terms(normalized_query)
    candidate_limit = max(top_k * 8, 40)
    database_started = time.perf_counter()
    with _get_vector_conn() as conn:
        with conn.cursor() as cur:
            # Semantic search can miss an exact Thai heading. Fetch direct term
            # matches as a second candidate set. Both candidate sets are fetched
            # in one database round trip and reranked together below.
            lexical_terms = sorted(
                (term for term in exact_terms if term != "c" and len(term) >= 3),
                key=len,
                reverse=True,
            )[:14]
            if lexical_terms:
                conditions = " OR ".join("content ILIKE %s" for _ in lexical_terms)
                patterns = [f"%{term}%" for term in lexical_terms]
                cur.execute(
                    "WITH semantic_candidates AS ("
                    "SELECT content, source, location_metadata, 1 - (embedding <=> %s) AS score "
                    "FROM documents_v2 WHERE topic_slug = %s AND embedding_profile = %s "
                    "ORDER BY embedding <=> %s LIMIT %s"
                    "), lexical_candidates AS ("
                    "SELECT content, source, location_metadata, 1 - (embedding <=> %s) AS score "
                    "FROM documents_v2 WHERE topic_slug = %s AND embedding_profile = %s "
                    f"AND ({conditions}) ORDER BY embedding <=> %s LIMIT %s"
                    ") SELECT content, source, location_metadata, score FROM semantic_candidates "
                    "UNION ALL SELECT content, source, location_metadata, score FROM lexical_candidates",
                    (
                        Vector(query_emb), topic_slug, provider.profile, Vector(query_emb), candidate_limit,
                        Vector(query_emb), topic_slug, provider.profile,
                        *patterns, Vector(query_emb), candidate_limit,
                    ),
                )
            else:
                cur.execute(
                    "SELECT content, source, location_metadata, 1 - (embedding <=> %s) AS score "
                    "FROM documents_v2 WHERE topic_slug = %s AND embedding_profile = %s "
                    "ORDER BY embedding <=> %s LIMIT %s",
                    (Vector(query_emb), topic_slug, provider.profile, Vector(query_emb), candidate_limit),
                )
            rows = cur.fetchall()
    database_ms = (time.perf_counter() - database_started) * 1000
    if timing_out is not None:
        timing_out.update({
            "embedding_ms": round(embed_ms, 1),
            "database_ms": round(database_ms, 1),
        })
    unique_rows = []
    seen_rows = set()
    for row in rows:
        metadata_key = json.dumps(row[2], ensure_ascii=False, sort_keys=True) if isinstance(row[2], dict) else str(row[2])
        row_key = (row[0], row[1], metadata_key)
        if row_key not in seen_rows:
            seen_rows.add(row_key)
            unique_rows.append(row)
    ranked = []
    for row in unique_rows:
        semantic_score = float(row[3])
        lexical_boost, lexical_coverage = _lexical_relevance(row[0], exact_terms)
        score = semantic_score + lexical_boost
        metadata = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
        ranked.append({
            "content": row[0],
            "source": row[1],
            "location": metadata.get("location", "เนื้อหา"),
            "score": score,
            "semantic_score": semantic_score,
            "lexical_coverage": lexical_coverage,
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    if not ranked or ranked[0]["score"] < min_score:
        return []
    adaptive_floor = max(0.26, ranked[0]["score"] - 0.11)
    results = []
    seen_content = set()
    topic_counts = {}
    for item in ranked:
        if item["score"] < adaptive_floor:
            continue
        content_key = re.sub(r"\s+", " ", item["content"].lower()).strip()
        if content_key in seen_content:
            continue
        topic_key = _retrieval_topic_key(item["content"])
        if topic_counts.get(topic_key, 0) >= 2:
            continue
        seen_content.add(content_key)
        topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
        results.append(item)
        if len(results) >= top_k:
            break
    for index, item in enumerate(results, start=1):
        item["citation_id"] = f"D{index}"
    return results


def list_ingestion_jobs(topic_slug, limit=20):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, filename, status, total_chunks, completed_chunks, error, updated_at "
                "FROM ingestion_jobs WHERE topic_slug = %s ORDER BY created_at DESC LIMIT %s",
                (topic_slug, limit),
            )
            rows = cur.fetchall()
    return [
        {"id": r[0], "filename": r[1], "status": r[2], "total": r[3], "completed": r[4], "error": r[5], "updated_at": r[6]}
        for r in rows
    ]


def list_sources(topic_slug):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM sources WHERE topic_slug = %s ORDER BY filename",
                (topic_slug,),
            )
            rows = cur.fetchall()
    return [r[0] for r in rows]


def topic_needs_reindex(topic_slug):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM sources WHERE topic_slug = %s", (topic_slug,))
            source_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM documents_v2 WHERE topic_slug = %s", (topic_slug,))
            vector_count = cur.fetchone()[0]
    return source_count > 0 and vector_count == 0


def reindex_topic(topic_slug):
    """Safely migrate legacy source text to the local embedding profile."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, full_text FROM sources WHERE topic_slug = %s ORDER BY filename", (topic_slug,))
            sources = cur.fetchall()
    provider = get_embedding_provider()
    prepared = []
    for filename, full_text in sources:
        chunks = chunk_text(full_text, size=450, overlap=60)
        vectors = provider.embed_documents(chunks)
        prepared.append((filename, chunks, vectors))
    with _get_vector_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents_v2 WHERE topic_slug = %s", (topic_slug,))
            total = 0
            for filename, chunks, vectors in prepared:
                for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    cur.execute(
                        "INSERT INTO documents_v2 "
                        "(topic_slug, source, chunk_index, content, content_hash, location_metadata, embedding_profile, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            topic_slug, filename, index, chunk,
                            hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                            psycopg2.extras.Json({"location": "ข้อมูลเดิม (ไม่มี metadata ตำแหน่ง)"}),
                            provider.profile, Vector(vector),
                        ),
                    )
                    total += 1
            cur.execute(
                "UPDATE topics SET embedding_profile = %s, knowledge_version = knowledge_version + 1 WHERE slug = %s",
                (provider.profile, topic_slug),
            )
            cur.execute("DELETE FROM qa_cache WHERE topic = %s", (topic_slug,))
        conn.commit()
    return total


# ===== ผู้ใช้ =====
def _hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$" + base64.b64encode(salt).decode("ascii") + "$" + base64.b64encode(digest).decode("ascii")


def _verify_password(password, stored_hash):
    try:
        algorithm, salt_b64, digest_b64 = stored_hash.split("$", 2)
        if algorithm != "scrypt":
            return False
        actual = _hash_password(
            password, base64.b64decode(salt_b64)
        ).split("$", 2)[2]
        return hmac.compare_digest(actual, digest_b64)
    except (AttributeError, TypeError, ValueError):
        return False


def ensure_admin_user(password):
    """Create the initial admin and attach legacy chats without an owner."""
    if not password:
        return None
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = 'admin'")
            row = cur.fetchone()
            if row:
                user_id = row[0]
            else:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES ('admin', %s, 'admin') RETURNING id",
                    (_hash_password(password),),
                )
                user_id = cur.fetchone()[0]
            cur.execute("UPDATE chats SET owner_id = %s WHERE owner_id IS NULL", (user_id,))
        conn.commit()
    return user_id


def authenticate_user(username, password):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, role FROM users WHERE lower(username) = lower(%s) AND active = true",
                (username.strip(),),
            )
            row = cur.fetchone()
    if not row:
        return None
    if not _verify_password(password, row[2]):
        return None
    return {"id": row[0], "username": row[1], "role": row[3]}


def create_user(username, password, role="user"):
    username = username.strip()
    if len(username) < 3 or len(password) < 8 or role not in ("admin", "user"):
        raise ValueError("ชื่อผู้ใช้ต้องมีอย่างน้อย 3 ตัว และรหัสผ่านอย่างน้อย 8 ตัว")
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id",
                (username, _hash_password(password), role),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
    return user_id


def list_users():
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, role, active, created_at FROM users ORDER BY username")
            rows = cur.fetchall()
    return [{"id": r[0], "username": r[1], "role": r[2], "active": r[3], "created_at": r[4]} for r in rows]


def set_user_active(user_id, active):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET active = %s WHERE id = %s AND username <> 'admin'", (bool(active), user_id))
        conn.commit()


def reset_user_password(user_id, password):
    if len(password) < 8:
        raise ValueError("รหัสผ่านต้องมีอย่างน้อย 8 ตัว")
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (_hash_password(password), user_id))
        conn.commit()


def change_user_password(user_id, current_password, new_password):
    """Change an authenticated user's own password after checking the old one."""
    if len(new_password or "") < 8:
        raise ValueError("รหัสผ่านใหม่ต้องมีอย่างน้อย 8 ตัว")
    if current_password == new_password:
        raise ValueError("รหัสผ่านใหม่ต้องไม่ซ้ำกับรหัสผ่านเดิม")
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users "
                "WHERE id = %s AND active = true FOR UPDATE",
                (user_id,),
            )
            row = cur.fetchone()
            if not row or not _verify_password(current_password or "", row[0]):
                raise ValueError("รหัสผ่านเดิมไม่ถูกต้อง")
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (_hash_password(new_password), user_id),
            )
        conn.commit()
    return True


# ===== ประวัติแชท =====
def create_chat(filename, owner_id=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (filename, owner_id) VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING",
                (filename, owner_id),
            )
        conn.commit()


def list_chats(owner_id=None):
    """คืนรายชื่อไฟล์แชท เรียงปักหมุดขึ้นก่อน แล้วใหม่สุดก่อนในแต่ละกลุ่ม"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM chats WHERE (%s IS NULL OR owner_id = %s) ORDER BY pinned DESC, filename DESC",
                (owner_id, owner_id),
            )
            rows = cur.fetchall()
    return [r[0] for r in rows]


def list_chat_summaries(owner_id=None):
    """Return all sidebar metadata in one query.

    This replaces separate list/title/pin/mode queries on every Streamlit
    rerun.  The topic display name follows the current topic name when it was
    renamed, while keeping the saved fallback for deleted/legacy topics.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.filename, c.title, c.pinned, c.active_topic_slug, "
                "COALESCE(t.name, c.active_topic_name) "
                "FROM chats c LEFT JOIN topics t ON t.slug = c.active_topic_slug "
                "WHERE (%s IS NULL OR c.owner_id = %s) "
                "ORDER BY c.pinned DESC, c.filename DESC",
                (owner_id, owner_id),
            )
            rows = cur.fetchall()
    return [
        {
            "filename": row[0],
            "title": row[1],
            "pinned": bool(row[2]),
            "topic_slug": row[3],
            "topic_name": row[4],
        }
        for row in rows
    ]


def load_chat(filename, owner_id=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, timestamp FROM chat_messages "
                "JOIN chats c ON c.filename = chat_messages.chat_filename "
                "WHERE chat_filename = %s AND (%s IS NULL OR c.owner_id = %s) ORDER BY seq",
                (filename, owner_id, owner_id),
            )
            rows = cur.fetchall()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]


def load_chat_page(filename, owner_id=None, limit=50, before_seq=None):
    """Load a newest-first page, returned in normal chronological order."""
    limit = max(1, min(int(limit), 200))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT m.role, m.content, m.timestamp, m.seq "
                "FROM chat_messages m JOIN chats c ON c.filename = m.chat_filename "
                "WHERE m.chat_filename = %s AND (%s IS NULL OR c.owner_id = %s) "
                "AND (%s IS NULL OR m.seq < %s) "
                "ORDER BY m.seq DESC LIMIT %s",
                (filename, owner_id, owner_id, before_seq, before_seq, limit + 1),
            )
            rows = cur.fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    oldest_seq = rows[-1][3] if rows else None
    messages = [
        {"role": row[0], "content": row[1], "timestamp": row[2]}
        for row in reversed(rows)
    ]
    return messages, has_more, oldest_seq


def save_chat(filename, messages, owner_id=None):
    """เขียนทับข้อความทั้งหมดของแชทนี้ (เรียบง่าย ปลอดภัยกว่าการอัปเดตทีละบรรทัด)"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (filename, owner_id) VALUES (%s, %s) "
                "ON CONFLICT (filename) DO NOTHING",
                (filename, owner_id),
            )
            cur.execute("SELECT owner_id FROM chats WHERE filename = %s", (filename,))
            chat_owner = cur.fetchone()
            if owner_id is not None and (not chat_owner or chat_owner[0] != owner_id):
                raise PermissionError("ไม่มีสิทธิ์แก้ไขแชทนี้")
            cur.execute("DELETE FROM chat_messages WHERE chat_filename = %s", (filename,))
            for i, msg in enumerate(messages):
                cur.execute(
                    "INSERT INTO chat_messages (chat_filename, role, content, timestamp, seq) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (filename, msg["role"], msg["content"], msg.get("timestamp"), i),
                )
        conn.commit()


def append_chat_messages(filename, messages, owner_id=None):
    """Append new messages atomically without rewriting the chat history.

    Locking the parent chat row serializes writers from multiple browser tabs.
    The unique database index on ``(chat_filename, seq)`` is a final guard
    against duplicate sequence numbers.
    """
    if not messages:
        return 0
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (filename, owner_id) VALUES (%s, %s) "
                "ON CONFLICT (filename) DO NOTHING",
                (filename, owner_id),
            )
            cur.execute(
                "SELECT owner_id FROM chats WHERE filename = %s FOR UPDATE",
                (filename,),
            )
            chat_owner = cur.fetchone()
            if owner_id is not None and (not chat_owner or chat_owner[0] != owner_id):
                raise PermissionError("ไม่มีสิทธิ์แก้ไขแชทนี้")
            cur.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM chat_messages WHERE chat_filename = %s",
                (filename,),
            )
            first_seq = cur.fetchone()[0]
            rows = [
                (
                    filename,
                    msg["role"],
                    msg["content"],
                    msg.get("timestamp"),
                    first_seq + offset,
                )
                for offset, msg in enumerate(messages)
            ]
            psycopg2.extras.execute_batch(
                cur,
                "INSERT INTO chat_messages "
                "(chat_filename, role, content, timestamp, seq) VALUES (%s, %s, %s, %s, %s)",
                rows,
                page_size=50,
            )
        conn.commit()
    return len(messages)


def get_chat_mode(filename, owner_id=None):
    """คืนค่า Knowledge ที่ผูกกับแชท หรือโหมดทั่วไปถ้าไม่มีหัวข้อ."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.active_topic_slug, COALESCE(t.name, c.active_topic_name) "
                "FROM chats c LEFT JOIN topics t ON t.slug = c.active_topic_slug "
                "WHERE c.filename = %s AND (%s IS NULL OR c.owner_id = %s)",
                (filename, owner_id, owner_id),
            )
            row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def set_chat_mode(filename, topic_slug, topic_name, owner_id=None):
    """บันทึกโหมดแยกต่อแชท เพื่อให้คงอยู่หลัง logout/reboot/เปลี่ยนอุปกรณ์."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET active_topic_slug = %s, active_topic_name = %s "
                "WHERE filename = %s AND (%s IS NULL OR owner_id = %s)",
                (topic_slug, topic_name, filename, owner_id, owner_id),
            )
        conn.commit()


def delete_chat(filename, owner_id=None):
    """ลบแชท: chat_messages ลบตามด้วย ON DELETE CASCADE"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chats WHERE filename = %s AND (%s IS NULL OR owner_id = %s)", (filename, owner_id, owner_id))
        conn.commit()


def toggle_pin(filename, owner_id=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE chats SET pinned = NOT pinned WHERE filename = %s AND (%s IS NULL OR owner_id = %s)", (filename, owner_id, owner_id))
        conn.commit()


def is_pinned(filename, owner_id=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pinned FROM chats WHERE filename = %s AND (%s IS NULL OR owner_id = %s)", (filename, owner_id, owner_id))
            row = cur.fetchone()
    return bool(row and row[0])


def get_pinned_set(owner_id=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM chats WHERE pinned = true AND (%s IS NULL OR owner_id = %s)", (owner_id, owner_id))
            rows = cur.fetchall()
    return {r[0] for r in rows}


def rename_chat(filename, new_title, owner_id=None):
    new_title = new_title.strip()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET title = %s WHERE filename = %s AND (%s IS NULL OR owner_id = %s)",
                (new_title if new_title else None, filename, owner_id, owner_id),
            )
        conn.commit()


def get_chat_titles(owner_id=None):
    """คืน dict {filename: title} เฉพาะแชทที่มีชื่อแล้ว (เอง หรืออัตโนมัติ)"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, title FROM chats WHERE title IS NOT NULL AND (%s IS NULL OR owner_id = %s)", (owner_id, owner_id))
            rows = cur.fetchall()
    return {r[0]: r[1] for r in rows}


# ===== cache คำตอบ =====
def get_cached_answer(cache_key, max_age_hours=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if max_age_hours is None:
                cur.execute("SELECT answer, sources FROM qa_cache WHERE cache_key = %s", (cache_key,))
            else:
                cur.execute(
                    "SELECT answer, sources FROM qa_cache WHERE cache_key = %s "
                    "AND created_at >= now() - (%s * interval '1 hour')",
                    (cache_key, float(max_age_hours)),
                )
            row = cur.fetchone()
    return {"answer": row[0], "sources": row[1] or []} if row else None


def store_cached_answer(cache_key, question, topic, answer, sources=None):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO qa_cache (cache_key, question, topic, answer, sources) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (cache_key) DO UPDATE SET question = EXCLUDED.question, "
                "topic = EXCLUDED.topic, answer = EXCLUDED.answer, sources = EXCLUDED.sources, created_at = now()",
                (
                    cache_key,
                    question,
                    topic,
                    answer,
                    psycopg2.extras.Json(sources or []),
                ),
            )
            # ตัด cache เก่าสุดทิ้งถ้าเกิน 500 รายการ
            cur.execute("SELECT count(*) FROM qa_cache")
            count = cur.fetchone()[0]
            if count > 500:
                cur.execute(
                    "DELETE FROM qa_cache WHERE cache_key IN ("
                    "SELECT cache_key FROM qa_cache ORDER BY created_at ASC LIMIT %s)",
                    (count - 500,),
                )
        conn.commit()


def clear_qa_cache():
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM qa_cache")
        conn.commit()


def count_qa_cache():
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM qa_cache")
            return cur.fetchone()[0]
