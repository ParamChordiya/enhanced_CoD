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
