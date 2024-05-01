from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from fastai.learner import load_learner
import os
from tempfile import NamedTemporaryFile
import base64
import dotenv

dotenv.load_dotenv()

MODEL_FILE = os.environ.get("MODEL_FILE")
if MODEL_FILE is None:
    raise Exception("MODEL_FILE environment variable is required")
MODEL_NAME = os.environ.get("MODEL_NAME")
if MODEL_NAME is None:
    raise Exception("MODEL_NAME environment variable is required")

app = FastAPI()
model = load_learner(MODEL_FILE)


@app.get("/healthz", response_class=PlainTextResponse)
def get_healthz():
    return "OK"


class ClassifyRequest(BaseModel):
    image_base64: str


@app.post("/api/v0/classify")
def post_classify(req: ClassifyRequest):
    print("Received request to classify")
    with NamedTemporaryFile() as f:
        f.write(base64.b64decode(req.image_base64))
        f.seek(0)
        print("Predicting score")
        score = classify_image(f.name)
        print("Predicted ", str(score))
        return {"validity_score": score, "model": MODEL_NAME}


def classify_image(path: str) -> float:
    """
    Classify an image, returning the probability it is valid
    """
    _, _, probs = model.predict(path)
    return probs[1].item()
