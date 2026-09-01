# Major.AI

Streamlit chatbot with Gemini generation/search, local multilingual ONNX
embeddings, and Neon Postgres/pgvector storage.

## Run

1. Install Python 3.11 dependencies: `pip install -r requirements.txt`.
2. Install the OS packages in `packages.txt` (required for Thai/image and
   scanned-PDF OCR).
3. Configure `GEMINI_API_KEY`, `DATABASE_URL`, and `ADMIN_PASSWORD` in `.env`
   locally or Streamlit Secrets in the cloud.
4. Run `streamlit run app.py`.

The first successful start creates an `admin` account whose password is
`ADMIN_PASSWORD`. Existing chats without an owner are assigned to this account.

## Knowledge ingestion

New documents use `local-minilm-l12-v1` (384 dimensions) in the hosted app, so
embedding does not consume Gemini quota. Existing 768-dimensional knowledge is
kept intact; use **ย้ายหัวข้อนี้ไป Local Embedding** in the Admin sidebar to
build the new index before switching retrieval.

Failed upload jobs keep their extracted chunks in Neon. Use **ทำงานนี้ต่อ** to
resume only chunks that do not have embeddings yet.

When a knowledge topic is active, answers use local documents only and cite the
file plus page/sheet/row/slide. If evidence is insufficient, the app asks before
using Google Search. General chat continues to use Gemini Google Search and
renders returned web sources.

## Tests

Run `python -m unittest discover -s tests` after installing dependencies.

## Performance behavior

- Streamlit initializes the database schema, admin bootstrap, Gemini client,
  rate limiter, and ingestion lock once per server process.
- A normal rerun fetches chat title, pin, and Knowledge mode metadata in one
  sidebar query. Admin user/Knowledge data is loaded only when its panel opens.
- Opening a chat loads the latest 50 messages; older pages are loaded on demand.
- New messages are appended atomically instead of deleting and rewriting the
  entire conversation.
- Successful answers stream to the UI and are cached. General answers expire
  from use after 24 hours; Knowledge answers are invalidated when sources change.
- Only one local embedding ingestion task can run at a time, preventing multiple
  admin uploads from exhausting the shared Streamlit CPU.

Admins can inspect the latest server-side preparation, retrieval, Gemini, and
save timings in **ประสิทธิภาพล่าสุด** in the sidebar. Community Cloud cold-start
time is platform-controlled and is separate from these warm-run timings.
The before/after implementation record and a live-test table are available in
[`PERFORMANCE_REPORT.md`](PERFORMANCE_REPORT.md).
