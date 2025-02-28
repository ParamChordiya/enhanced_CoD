import requests
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

BACKEND_URL = "http://127.0.0.1:8000/ask"
PLOTS_DIR = "plots"
RESULTS_CSV = "benchmark_logs.csv"  # or "frontend/benchmark_logs.csv"

METHODS = ["standard", "cot", "cod"]

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def load_dataset(file_path):
    """
    Load a JSON file containing items of form:
    [
      {
        "question": " ... ",
        "answer": " ... "  # correct answer if you have it
      },
      ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def evaluate_dataset(dataset_name, dataset):
    results = []
    for i, item in enumerate(dataset):
        question = item.get("question", "")
        correct_answer = item.get("answer", "").strip()  # if available

        for method in METHODS:
            payload = {"question": question, "method": method}
            start_time = time.time()
            try:
                resp = requests.post(BACKEND_URL, json=payload, timeout=60).json()
            except Exception as e:
                print(f"Error while calling method {method} for question {i}: {e}")
                continue
            end_time = time.time()

            response_text = resp.get("response", "").strip()
            # If you want to define accuracy by matching correct_answer or partial match:
            # e.g. 'correct' if correct_answer in response_text
            # or you can do a simple string comparison if it's straightforward
            is_correct = False
            if correct_answer:
                # naive check, you can do advanced checking
                is_correct = correct_answer.lower() in response_text.lower()

            # Collect token usage
            prompt_tokens = resp.get("prompt_tokens")
            completion_tokens = resp.get("completion_tokens")
            total_tokens = resp.get("total_tokens")
            inference_time_s = resp.get("inference_time_s") or (end_time - start_time)

            results.append({
                "dataset": dataset_name,
                "index": i,
                "question": question,
                "method": method,
                "response": response_text,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "response_time_s": inference_time_s
            })
    return results

def run_evaluation():
    # Load sample datasets
    gsm8k = load_dataset("datasets/gsm8k_sample.json")


    # Evaluate each
    all_results = []
    all_results += evaluate_dataset("gsm8k", gsm8k)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save appended logs in case you want everything in one file
    if os.path.exists(RESULTS_CSV):
        old_df = pd.read_csv(RESULTS_CSV)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)

    # Summaries
    print("====== SUMMARY ======")
    group = df.groupby(["dataset", "method"])
    acc = group["is_correct"].mean() * 100
    avg_time = group["response_time_s"].mean()
    avg_tokens = group["total_tokens"].mean()

    print("Accuracy (%) by dataset/method:\n", acc)
    print("\nAvg Response Time (s):\n", avg_time)
    print("\nAvg Total Tokens:\n", avg_tokens)

    # ---------- PLOTS ----------
    # 1) Accuracy Plot
    acc_plot = acc.unstack(level=-1).plot(kind="bar", figsize=(8,5), title="Accuracy by Method")
    acc_plot.set_ylabel("Accuracy (%)")
    acc_plot.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_by_method.png"))
    plt.close()

    # 2) Response Time Plot
    time_plot = avg_time.unstack(level=-1).plot(kind="bar", figsize=(8,5), title="Response Time by Method")
    time_plot.set_ylabel("Avg Time (s)")
    time_plot.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "time_by_method.png"))
    plt.close()

    # 3) Token Usage Plot
    tokens_plot = avg_tokens.unstack(level=-1).plot(kind="bar", figsize=(8,5), title="Avg Tokens by Method")
    tokens_plot.set_ylabel("Average Tokens Used")
    tokens_plot.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tokens_by_method.png"))
    plt.close()

    print(f"\nPlots saved in '{PLOTS_DIR}' folder. Full results in '{RESULTS_CSV}'.")

if __name__ == "__main__":
    run_evaluation()
