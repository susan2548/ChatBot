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
import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector import Vector
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types

# ใช้ฟังก์ชันอ่านไฟล์/ตัด chunk ตัวเดียวกับ rag.py (ไม่ผูกกับ ChromaDB เลย เอามาใช้ซ้ำได้ปลอดภัย)
from rag import read_file, chunk_text, SUPPORTED_EXTS, ZIP_EXT

EMBED_DIM = 768  # pgvector index (hnsw/ivfflat) รองรับสูงสุด 2000 มิติ เลยตัด gemini-embedding-001 ลงมาที่ 768
EMBED_BATCH_SIZE = 100  # Gemini embedContent รับได้สูงสุด 100 รายการต่อ batch

_database_url = None
_genai_client = None
_pool = None
_schema_ready = False


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
        _pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=database_url)


class _PooledConn:
    """context manager: ยืม connection จาก pool มาใช้ แล้วคืนกลับ pool ตอนจบ (ไม่ปิดทิ้งจริง)"""

    def __enter__(self):
        self._conn = _pool.getconn()
        register_vector(self._conn)  # ปลอดภัยเรียกซ้ำได้ทุกครั้ง เร็วมากเพราะ connection เปิดอยู่แล้ว
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        _pool.putconn(self._conn)
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

            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    topic TEXT,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
        conn.commit()
    finally:
        conn.close()


# ===== embedding =====
def embed(texts):
    """แปลงข้อความเป็น vector ด้วย Gemini embedding (ตัดเหลือ 768 มิติ + แบ่ง batch ถ้าเกิน 100 รายการ)"""
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = _genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        # แปลงเป็น float ธรรมดาให้ชัวร์ (ค่าดิบจาก SDK อาจไม่ใช่ float เพียวๆ) ก่อนห่อด้วย
        # pgvector.Vector(...) ตอนใช้งานจริง — เพราะ pgvector adapter รู้จักแค่ Vector/np.ndarray
        # เท่านั้น ถ้าส่ง list เปล่าๆ ตรงๆ psycopg2 จะ fallback เป็น numeric[] แทน vector แล้ว query
        # ที่ใช้ตัวดำเนินการ <=> จะพังด้วย "operator does not exist: vector <=> numeric[]"
        embeddings.extend([float(x) for x in e.values] for e in result.embeddings)
    return embeddings


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


def add_document(topic_slug, path, filename):
    """อ่านไฟล์ → chunk → embed → เก็บลง Postgres (.zip จะถูกแตกไฟล์แล้วเพิ่มทีละไฟล์ข้างในแทน)"""
    if os.path.splitext(filename)[1].lower() == ZIP_EXT:
        return _add_zip(topic_slug, path)

    text = read_file(path)
    if not text:
        return 0

    chunks = chunk_text(text)
    embeddings = embed(chunks)

    with _get_vector_conn() as conn:
        with conn.cursor() as cur:
            _upsert_source_text(cur, topic_slug, filename, text)
            _insert_chunks(cur, topic_slug, filename, chunks, embeddings)
        conn.commit()
    return len(chunks)


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
            cur.execute("DELETE FROM sources WHERE topic_slug = %s AND filename = %s", (topic_slug, filename))
        conn.commit()


def replace_source_file(topic_slug, path, filename):
    """แทนที่ไฟล์เดิมด้วยไฟล์ใหม่ (path ถูกอัปโหลดใหม่แล้ว) — ลบของเก่าก่อนแล้ว re-embed ใหม่ทั้งไฟล์"""
    delete_source_file(topic_slug, filename)
    return add_document(topic_slug, path, filename)


def replace_source_text(topic_slug, filename, new_text):
    """แก้ไขเนื้อหาไฟล์ text-based โดยตรงจากกล่องข้อความ (rechunk + re-embed ใหม่ทั้งหมด)"""
    delete_source_file(topic_slug, filename)
    new_text = new_text.strip()
    if not new_text:
        return 0

    chunks = chunk_text(new_text)
    embeddings = embed(chunks)
    with _get_vector_conn() as conn:
        with conn.cursor() as cur:
            _upsert_source_text(cur, topic_slug, filename, new_text)
            _insert_chunks(cur, topic_slug, filename, chunks, embeddings)
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
    tmp_dir = tempfile.mkdtemp(prefix="major_ai_zip_")
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
    """ค้นหา chunk ที่ใกล้เคียงกับ query มากที่สุด (cosine distance ผ่าน pgvector)"""
    query_emb = embed([query])[0]
    with _get_vector_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM documents WHERE topic_slug = %s "
                "ORDER BY embedding <=> %s LIMIT %s",
                (topic_slug, Vector(query_emb), top_k),
            )
            rows = cur.fetchall()
    if not rows:
        return ""
    return "\n\n---\n\n".join(r[0] for r in rows)


def list_sources(topic_slug):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM sources WHERE topic_slug = %s ORDER BY filename",
                (topic_slug,),
            )
            rows = cur.fetchall()
    return [r[0] for r in rows]


# ===== ประวัติแชท =====
def create_chat(filename):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (filename) VALUES (%s) ON CONFLICT (filename) DO NOTHING",
                (filename,),
            )
        conn.commit()


def list_chats():
    """คืนรายชื่อไฟล์แชท เรียงปักหมุดขึ้นก่อน แล้วใหม่สุดก่อนในแต่ละกลุ่ม"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM chats ORDER BY pinned DESC, filename DESC"
            )
            rows = cur.fetchall()
    return [r[0] for r in rows]


def load_chat(filename):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, timestamp FROM chat_messages "
                "WHERE chat_filename = %s ORDER BY seq",
                (filename,),
            )
            rows = cur.fetchall()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]


def save_chat(filename, messages):
    """เขียนทับข้อความทั้งหมดของแชทนี้ (เรียบง่าย ปลอดภัยกว่าการอัปเดตทีละบรรทัด)"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (filename) VALUES (%s) ON CONFLICT (filename) DO NOTHING",
                (filename,),
            )
            cur.execute("DELETE FROM chat_messages WHERE chat_filename = %s", (filename,))
            for i, msg in enumerate(messages):
                cur.execute(
                    "INSERT INTO chat_messages (chat_filename, role, content, timestamp, seq) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (filename, msg["role"], msg["content"], msg.get("timestamp"), i),
                )
        conn.commit()


def delete_chat(filename):
    """ลบแชท: chat_messages ลบตามด้วย ON DELETE CASCADE"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chats WHERE filename = %s", (filename,))
        conn.commit()


def toggle_pin(filename):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE chats SET pinned = NOT pinned WHERE filename = %s", (filename,))
        conn.commit()


def is_pinned(filename):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pinned FROM chats WHERE filename = %s", (filename,))
            row = cur.fetchone()
    return bool(row and row[0])


def get_pinned_set():
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM chats WHERE pinned = true")
            rows = cur.fetchall()
    return {r[0] for r in rows}


def rename_chat(filename, new_title):
    new_title = new_title.strip()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET title = %s WHERE filename = %s",
                (new_title if new_title else None, filename),
            )
        conn.commit()


def get_chat_titles():
    """คืน dict {filename: title} เฉพาะแชทที่มีชื่อแล้ว (เอง หรืออัตโนมัติ)"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, title FROM chats WHERE title IS NOT NULL")
            rows = cur.fetchall()
    return {r[0]: r[1] for r in rows}


# ===== cache คำตอบ =====
def get_cached_answer(cache_key):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT answer FROM qa_cache WHERE cache_key = %s", (cache_key,))
            row = cur.fetchone()
    return row[0] if row else None


def store_cached_answer(cache_key, question, topic, answer):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO qa_cache (cache_key, question, topic, answer) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (cache_key) DO UPDATE SET question = EXCLUDED.question, "
                "topic = EXCLUDED.topic, answer = EXCLUDED.answer, created_at = now()",
                (cache_key, question, topic, answer),
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
