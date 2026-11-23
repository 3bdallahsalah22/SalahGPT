from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

# Project imports
from Agent.agent import get_agent

from Data.database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
)

from Rag.rag import add_document_to_rag
from Tools.tools import set_current_thread_id


# 
# Project paths
# 

# الملف في App/app.py  →  parent = App  →  parent.parent = جذر المشروع
BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

TEMPLATE_DIR = BASE_DIR / "Templates"
STATIC_DIR = BASE_DIR / "Static"

DEFAULT_MODEL = "gemini-3.8-flash"


#
# FastAPI App
# 

app = FastAPI()

# ملفات الـ CSS والـ JS (لازم قبل أي route)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# Initialize database
init_db()


# 
# Home Route
# =

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# 
# Conversations
#

@app.get("/conversations")
async def conversations():
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
            for item in items
        ]
    }


# 
# Chat History
# 

@app.get("/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    }


# 
# Document Upload
# 

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.",
                },
                status_code=400,
            )

        file_id = str(uuid.uuid4())
        safe_filename = Path(filename).name.replace(" ", "_")
        file_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id,
        )

        return JSONResponse(
            {
                "success": True,
                "message": (
                    f"Uploaded {result['filename']} "
                    f"and created {result['chunks']} chunks."
                ),
            }
        )

    except Exception as e:
        return JSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )


# 
# Streaming Helpers
# 

def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    Stream only normal AI text chunks.
    Tool messages and tool-call chunks are skipped.
    """
    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node", "")).lower()

    # نسمح بس بالـ node بتاع الموديل (chatbot)
    if node_name and node_name != "chatbot":
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    # chunks طلب الأدوات (بتتبعت أجزاء أثناء الـ streaming)
    if getattr(chunk, "tool_call_chunks", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    """
    Gemini 3.x بيرجّع content كقائمة blocks (فيها thinking و text).
    ناخد النص بس ونتجاهل التفكير.
    """
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") in ("thinking", "reasoning", "thought"):
                    continue

                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts)

    return ""


# 
# Chat Streaming
# 

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", DEFAULT_MODEL)

    if not user_message.strip():
        return JSONResponse({"error": "Message is required."}, status_code=400)

    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)

    set_current_thread_id(thread_id)

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 12,  # يمنع loop الأدوات اللانهائي
    }

    def event_generator():
        final_answer = ""

        try:
            inputs = {"messages": [HumanMessage(content=user_message)]}

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages",
            ):
                # تشخيص: شوف اللي بيوصل فعلًا (امسح السطر ده بعد ما كل حاجة تشتغل)
                print(
                    "CHUNK:",
                    type(chunk).__name__,
                    repr(getattr(chunk, "content", ""))[:200],
                    (metadata or {}).get("langgraph_node"),
                    flush=True,
                )

                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)
            else:
                # بدل الصمت: نبلّغ الواجهة إن مفيش رد
                yield sse_data(
                    {"error": "No response received from the model. Check the terminal for details."}
                )

            yield sse_data({"done": True})

        except Exception as e:
            print("STREAM ERROR:", repr(e), flush=True)
            yield sse_data({"error": str(e)})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 
# Run Application
# 

if __name__ == "__main__":
    uvicorn.run(
        "App.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )