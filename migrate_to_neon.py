"""
สคริปต์ migration ครั้งเดียว: ย้ายข้อมูลจาก ChromaDB + ไฟล์ JSON ในเครื่อง (ระบบเดิมของ app.py)
ขึ้น Neon Postgres (db.py) รันซ้ำได้อย่างปลอดภัย (ลบของเก่าที่ตรง key แล้วใส่ใหม่ทุกครั้ง)

วิธีใช้:
    python migrate_to_neon.py

ต้องมี .env ที่ตั้งค่า GEMINI_API_KEY และ DATABASE_URL ไว้แล้ว (เอา DATABASE_URL จาก Neon dashboard)

หมายเหตุ: เวกเตอร์เดิมใน ChromaDB เป็น 3072 มิติ (ค่า default ของ gemini-embedding-001)
แต่ Postgres/pgvector index รองรับสูงสุด 2000 มิติ ระบบใหม่เลยตัดเหลือ 768 มิติ
สคริปต์นี้ "ตัด + normalize ใหม่" เวกเตอร์เดิมโดยตรง (Matryoshka truncation ของ Gemini embedding
ทำแบบนี้ได้อย่างปลอดภัย) ไม่ต้องยิง Gemini API ซ้ำ ไม่กินโควตาเพิ่มเลย
"""
import os
import json
import math
from dotenv import load_dotenv
from pgvector import Vector

import rag
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")

API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not API_KEY or not DATABASE_URL:
    raise SystemExit("ต้องตั้งค่า GEMINI_API_KEY และ DATABASE_URL ใน .env ก่อนรัน migration")


def truncate_and_normalize(vec, target_dim=768):
    """ตัด vector ให้เหลือ target_dim มิติ แล้ว normalize ใหม่ (Matryoshka truncation)"""
    vec = list(vec)
    if len(vec) <= target_dim:
        return vec
    truncated = vec[:target_dim]
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm == 0:
        return truncated
    return [x / norm for x in truncated]


def _chunk_index_of(id_):
    try:
        return int(id_.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return 0


def migrate_knowledge():
    print("== ย้ายหัวข้อความรู้ (topics + เอกสาร) ==")
    meta = rag._load_topics_meta()
    if not meta:
        print("ไม่มีหัวข้อความรู้ในเครื่อง ข้าม")
        return

    for slug, info in meta.items():
        name = info["name"]
        print(f"- หัวข้อ: {name} ({slug})")

        with db._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO topics (slug, name) VALUES (%s, %s) "
                    "ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name",
                    (slug, name),
                )
            conn.commit()

        collection = rag.get_collection(slug)
        if collection.count() == 0:
            print("  (ไม่มีเอกสารในหัวข้อนี้)")
            continue

        data = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data["ids"]
        docs = data["documents"]
        metas = data["metadatas"]
        embeddings = data["embeddings"]

        # รวม chunk กลับเป็นข้อความเต็มต่อไฟล์ (ใช้กับฟีเจอร์ดู/แก้ไขไฟล์ในเว็บ)
        by_source_chunks = {}
        for id_, doc, meta_row in zip(ids, docs, metas):
            src = meta_row.get("source", "unknown")
            by_source_chunks.setdefault(src, []).append((_chunk_index_of(id_), doc))

        with db._get_conn() as conn:
            with conn.cursor() as cur:
                for src, chunk_list in by_source_chunks.items():
                    chunk_list.sort(key=lambda x: x[0])
                    full_text = "".join(doc for _, doc in chunk_list)
                    db._upsert_source_text(cur, slug, src, full_text)
            conn.commit()

        with db._get_vector_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE topic_slug = %s", (slug,))
                for id_, doc, meta_row, emb in zip(ids, docs, metas, embeddings):
                    src = meta_row.get("source", "unknown")
                    vec = truncate_and_normalize(emb)
                    cur.execute(
                        "INSERT INTO documents (topic_slug, source, chunk_index, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (slug, src, _chunk_index_of(id_), doc, Vector(vec)),
                    )
            conn.commit()
        print(f"  ย้าย {len(docs)} chunks จาก {len(by_source_chunks)} ไฟล์แล้ว")


def migrate_chat_history():
    print("== ย้ายประวัติแชท ==")
    if not os.path.isdir(HISTORY_DIR):
        print("ไม่มีโฟลเดอร์ chat_history ข้าม")
        return

    pinned_path = os.path.join(HISTORY_DIR, "_pinned.json")
    titles_path = os.path.join(HISTORY_DIR, "_titles.json")

    pinned = set()
    if os.path.exists(pinned_path):
        with open(pinned_path, "r", encoding="utf-8") as f:
            pinned = set(json.load(f))

    titles = {}
    if os.path.exists(titles_path):
        with open(titles_path, "r", encoding="utf-8") as f:
            titles = json.load(f)

    chat_files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json") and not f.startswith("_")]
    for fname in chat_files:
        with open(os.path.join(HISTORY_DIR, fname), "r", encoding="utf-8") as f:
            messages = json.load(f)
        db.save_chat(fname, messages)
        if fname in titles:
            db.rename_chat(fname, titles[fname])
        if fname in pinned:
            with db._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE chats SET pinned = true WHERE filename = %s", (fname,))
                conn.commit()
    print(f"ย้าย {len(chat_files)} แชทแล้ว")


def migrate_qa_cache():
    print("== ย้าย cache คำตอบ ==")
    cache_path = os.path.join(HISTORY_DIR, "_qa_cache.json")
    if not os.path.exists(cache_path):
        print("ไม่มี cache ข้าม")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    for key, entry in cache.items():
        db.store_cached_answer(key, entry["question"], entry.get("topic"), entry["answer"])
    print(f"ย้าย {len(cache)} รายการ cache แล้ว")


if __name__ == "__main__":
    rag.init_client(API_KEY)
    db.init_db(DATABASE_URL, API_KEY)  # สร้างตารางถ้ายังไม่มี

    migrate_knowledge()
    migrate_chat_history()
    migrate_qa_cache()

    print("\nเสร็จแล้ว! ข้อมูลทั้งหมดอยู่ใน Neon Postgres แล้ว")
