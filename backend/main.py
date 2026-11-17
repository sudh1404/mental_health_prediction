
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from app.predictor import StressPredictor

load_dotenv()

predictor = None

# Basic phrase screening is imperfect and is not a crisis detector.
CRISIS_PHRASES = [
    "kill myself",
    "end my life",
    "suicide",
    "suicidal",
    "want to die",
    "hurt myself",
    "self harm",
    "self-harm",
]

CRISIS_REPLY = (
    "I'm really sorry you're going through this. You deserve support, "
    "and you don't have to face this moment alone. If you may act on "
    "these thoughts or are in immediate danger, call India's emergency "
    "number 112 or go to the nearest emergency department. In India, "
    "you can also call Tele-MANAS at 14416 for mental-health support. "
    "If possible, reach out to someone you trust and ask them to stay "
    "with you. Are you in immediate danger right now?"
)

SYSTEM_PROMPT = """
You are SocialSense Companion, a supportive mental-wellness assistant.
Be warm, respectful, calm, and non-judgmental.

Listen first. Validate feelings without assuming you know exactly how
the person feels. Ask at most one gentle follow-up question at a time.
Offer small, practical coping steps when appropriate.

Do not diagnose mental-health conditions, prescribe medication, or
claim to replace a qualified professional. Do not promise confidentiality
or guarantee outcomes. Encourage professional or trusted-person support
when appropriate. Never shame, pressure, or dismiss the user.

If a user expresses possible self-harm or suicide risk, respond with
compassion, encourage immediate human support, and recommend emergency
help when there is immediate danger. Do not provide self-harm instructions.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    print("Starting SocialSense backend...")

    predictor = StressPredictor()

    print("Backend ready!")

    yield

    predictor = None


app = FastAPI(
    title="SocialSense API",
    description="SocialSense text analysis and wellbeing companion",
    version="2.0.0",
    lifespan=lifespan,
)


# CORS configuration for the React + Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=12,
    )


@app.get("/")
def home():
    return {
        "message": "SocialSense API is running",
        "features": ["text analysis", "wellbeing companion"],
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if predictor is not None else "loading",
        "chat_configured": bool(os.getenv("GROQ_API_KEY"))
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter some text.",
        )

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not ready.",
        )

    return predictor.predict(request.text)



@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message.",
        )

    lowered = message.lower()

    # Basic crisis phrase screening
    if any(phrase in lowered for phrase in CRISIS_PHRASES):
        return {
            "reply": CRISIS_REPLY,
            "safety_notice": True,
        }

    # Get Groq API key
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Chat is not configured. Add GROQ_API_KEY to the backend .env file.",
        )

    # Prepare conversation history
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for item in request.history[-10:]:
        if item.role in ("user", "assistant"):
            messages.append({
                "role": item.role,
                "content": item.content[:4000],
            })

    messages.append({
        "role": "user",
        "content": message,
    })

    # Connect to Groq API
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        completion = client.chat.completions.create(
            model=os.getenv(
                "GROQ_CHAT_MODEL",
                "openai/gpt-oss-20b"
            ),
            messages=messages,
            max_tokens=350,
        )

        reply = completion.choices[0].message.content

        return {
            "reply": reply or "I'm here to listen. Would you like to tell me a little more?",
            "safety_notice": False,
        }

    except OpenAIError as exc:
        print(
            f"Groq API Error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=502,
            detail="The companion is temporarily unavailable. Please try again shortly.",
        )
