import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from google import genai
from google.genai import types
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import rag
from prompt import PROMPTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- Gemini ----
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
rag.init_client(api_key)

# ---- Line ----
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

app = FastAPI()

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# เก็บประวัติแยกตาม user Line (key = user_id)
user_histories = {}

# เก็บโหมดของแต่ละ user (key = user_id)
user_modes = {}

# คำสั่งพิมพ์ใน Line → ชื่อโหมดใน PROMPTS
MODE_COMMANDS = {
    "โหมดทั่วไป": "ทั่วไป (General)",
    "โหมดแรงงาน": "คุ้มครองแรงงาน (Excel)",
    "โหมดวิจัย": "ผู้ช่วยวิจัย (Research)",
}

# โหมดเริ่มต้น ถ้า user ยังไม่เคยเลือก
DEFAULT_MODE = "คุ้มครองแรงงาน (Excel)"


@app.get("/hello")
def hello():
    return {"hello": "world"}


@app.post("/message")
async def message(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print("Invalid signature. เช็ค channel access token / secret")
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_text = event.message.text.strip()

    # ---- คำสั่งดูวิธีใช้ ----
    if user_text.lower() in ("help", "เมนู", "โหมด"):
        menu = (
            "พิมพ์คำสั่งเพื่อสลับโหมด:\n"
            "• โหมดทั่วไป — คุยได้ทุกเรื่อง ค้นเน็ตได้\n"
            "• โหมดแรงงาน — ตอบเรื่องคุ้มครองแรงงานจากไฟล์\n"
            "• โหมดวิจัย — ผู้ช่วยงานวิจัย ค้นเน็ตได้"
        )
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=menu)
        )
        return

    # ---- คำสั่งสลับโหมด ----
    if user_text in MODE_COMMANDS:
        new_mode = MODE_COMMANDS[user_text]
        user_modes[user_id] = new_mode
        user_histories[user_id] = []  # ล้างประวัติตอนสลับโหมด
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"เปลี่ยนเป็น {new_mode} แล้ว ถามมาได้เลย"),
        )
        return

    # ---- คุยปกติ ----
    mode = user_modes.get(user_id, DEFAULT_MODE)
    history = user_histories.get(user_id, [])

    reply = chat_with_gemini(user_text, history, mode)

    history.append({"role": "user", "parts": [{"text": user_text}]})
    history.append({"role": "model", "parts": [{"text": reply}]})
    user_histories[user_id] = history[-20:]

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply),
    )


def chat_with_gemini(user_text, history, mode):
    cfg = PROMPTS[mode]

    tools = []
    if cfg["search"]:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=1024,
        system_instruction=cfg["prompt"],
        safety_settings=SAFETY_SETTINGS,
        tools=tools,
    )

    contents = []

    # RAG: ค้นความรู้ที่เกี่ยวข้อง
    if cfg["rag"]:
        context = rag.search(cfg["rag"], user_text, top_k=4)
        if context:
            contents.append({
                "role": "user",
                "parts": [{"text": f"ข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context}"}]
            })

    contents.extend(history)
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )
    return response.text