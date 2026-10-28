
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.predictor import StressPredictor


# Load the models once when the backend starts.
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    print("Starting backend and loading ML models...")
    predictor = StressPredictor()
    print("Backend ready!")

    yield

    predictor = None


app = FastAPI(
    title="SocialSense API",
    description="Social media stress-related language classification",
    version="1.0.0",
    lifespan=lifespan,
)


# Allow the React development server to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=10000,
        description="Social media post to analyze",
    )


@app.get("/")
def home():
    return {
        "message": "SocialSense API is running",
        "models": [
            "Logistic Regression",
            "Linear SVM",
            "RoBERTa",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if predictor is not None else "loading"
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
