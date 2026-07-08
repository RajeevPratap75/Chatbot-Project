import asyncio
import json
import logging
import os
import threading
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
import uvicorn
import httpx
try:
    from qdrant_client import QdrantClient
except ModuleNotFoundError:
    QdrantClient = None


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("college_assistant")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

QDRANT_URL = "https://944f67cc-5393-49f0-80e5-9d69bfa5f793.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_COLLECTION = "college_docs"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Qdrant client
qdrant_client = None
if QdrantClient:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Request model
class ChatRequest(BaseModel):
    message: str

class ConnectionManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("WebSocket connected. active_connections=%s", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info("WebSocket disconnected. active_connections=%s", len(self.active_connections))


manager = ConnectionManager()

def validate_message(message):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")
    return message.strip()

def parse_websocket_message(raw_message):
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return validate_message(raw_message)

    if not isinstance(payload, dict):
        raise ValueError("WebSocket payload must be a JSON object or plain text message.")

    return validate_message(payload.get("message"))

def search_documents(query_embedding):
    if qdrant_client:
        points = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=5
        ).points

        return [
            point.payload["text"]
            for point in points
            if point.payload and "text" in point.payload
        ]

    response = httpx.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        headers={"api-key": QDRANT_API_KEY},
        json={
            "vector": query_embedding,
            "limit": 5,
            "with_payload": True
        },
        timeout=60
    )
    response.raise_for_status()

    return [
        point["payload"]["text"]
        for point in response.json()["result"]
        if point.get("payload") and "text" in point["payload"]
    ]


def build_prompt(user_message, context):
    return f"""
You are St xavier's College Assistant.

Answer according to the uploaded documents if they match the user's question. If they do not match, answer according to the general knowledge of the LLM.

Rules:
- Use the uploaded document context as the primary source when it is relevant.
- If the context matches the user's question, answer from it.
- If the context does not match the user's question, answer using general LLM knowledge.

Format all responses using GitHub Flavored Markdown.
Rules:
- Use ## headings instead of # unless it's the main title.
- Use bullet lists when appropriate.
- Use numbered lists for procedures.
- Use tables for comparisons.
- Wrap all code inside fenced code blocks.
- Always specify the programming language.
- Never output raw HTML.

Context:
{context}

Question:
{user_message}

Answer:
"""

def retrieve_context(user_message):

    query_embedding = embedding_model.embed_query(user_message)
    search_result = search_documents(query_embedding)
    return "\n\n".join(search_result)


def generate_answer(prompt):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content

async def answer_question(user_message):
    user_message = validate_message(user_message)
    context = await asyncio.to_thread(retrieve_context, user_message)
    prompt = build_prompt(user_message, context)
    answer = await asyncio.to_thread(generate_answer, prompt)
    return answer

async def stream_answer(prompt, websocket, stop_event):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    done = object()

    def produce_tokens():
        try:
            stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=True,
            )

            for chunk in stream:
                if stop_event.is_set():
                    break

                delta = chunk.choices[0].delta.content
                if delta:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "content": delta})
        except Exception as exc:
            logger.exception("LLM streaming failed")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "stream_error", "detail": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    thread = threading.Thread(target=produce_tokens, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is done:
            break
        if item["type"] == "stream_error":
            return False
        await websocket.send_json(item)

    return True

# Home route
@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        answer = await answer_question(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("HTTP chat request failed")
        raise HTTPException(status_code=500, detail="The server could not answer right now.") from exc

    return {"reply": answer}

@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    stop_event = threading.Event()

    try:
        while True:
            raw_message = await websocket.receive_text()

            try:
                user_message = parse_websocket_message(raw_message)
            except ValueError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            logger.info("Processing WebSocket message")
            await websocket.send_json({"type": "status", "message": "retrieval_started"})

            try:
                context = await asyncio.to_thread(retrieve_context, user_message)
                prompt = build_prompt(user_message, context)
                await websocket.send_json({"type": "status", "message": "retrieval_complete"})
                await websocket.send_json({"type": "status", "message": "generation_started"})

                streamed = await stream_answer(prompt, websocket, stop_event)
                if not streamed:
                    await websocket.send_json({"type": "status", "message": "streaming_unavailable"})
                    answer = await asyncio.to_thread(generate_answer, prompt)
                    await websocket.send_json({"type": "final", "content": answer})

                await websocket.send_json({"type": "status", "message": "generation_finished"})
                await websocket.send_json({"type": "done"})
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("WebSocket message processing failed")
                await websocket.send_json({
                    "type": "error",
                    "message": "The server could not answer right now."
                })
    except WebSocketDisconnect:
        stop_event.set()
    except Exception:
        stop_event.set()
        logger.exception("Unexpected WebSocket error")
    finally:
        manager.disconnect(websocket)

# Run locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)