import os

MODEL_PATH = os.environ.get("MODEL_PATH", "models/llama-3-8B.Q4_K_M.gguf")
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "chain_of_draft_logs.csv")
