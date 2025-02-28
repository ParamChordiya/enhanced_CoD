import logging
import csv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("Please install llama-cpp-python with: pip install llama-cpp-python")

from config import MODEL_PATH, LOG_FILE_PATH

# --------------------- PROMPT STRATEGIES ---------------------
PROMPTS = {
    "standard": (
        "Answer directly.\n"
        "Q: {question}\nA:"
    ),
    "cot": (
        "Think step by step. Provide your reasoning, then return the final answer "
        "after the sequence of steps. Use '####' before the final answer.\n\n"
        "Q: {question}\nA:"
    ),
    "cod": (
        "Use a minimal 'draft' style of reasoning (~5 words per step). "
        "After drafting, return the final answer after '####'.\n\n"
        "Q: {question}\nA:"
    ),
}


# --------------------- SCHEMA ---------------------
class QueryRequest(BaseModel):
    question: str
    method: Optional[str] = "cod"

# --------------------- FASTAPI INIT ---------------------
app = FastAPI(
    title="Chain-of-Draft Reasoning Optimizer",
    version="1.0.0"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------- MODEL LOADING ---------------------
logger.info(f"Loading model from: {MODEL_PATH}")
try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

# --------------------- CSV LOGGER SETUP ---------------------
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "method", "question", "response", 
                         "prompt_tokens", "completion_tokens", "total_tokens", 
                         "inference_time_s"])

@app.post("/ask")
def ask_model(request: QueryRequest):
    method = request.method.lower()
    if method not in PROMPTS:
        logger.warning(f"Invalid method {method}, using 'cod' instead.")
        method = "cod"

    prompt = PROMPTS[method].format(question=request.question)
    try:
        # Inference
        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.2,
            stop=["####"]
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    text_response = output["choices"][0]["text"].strip()

    usage = output.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    inference_time_s = output.get("time", None)  # If your llama_cpp version includes 'time' field

    # --------------------- LOG TO CSV ---------------------
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            timestamp,
            method,
            request.question,
            text_response,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            inference_time_s
        ])

    return {
        "response": text_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "inference_time_s": inference_time_s
    }
