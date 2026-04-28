"""
Standalone Ragas evaluation runner.

Usage:
    python evaluate.py --csv test_dataset.csv --backend http://localhost:8000
    python evaluate.py --csv test_dataset.csv --local

Modes:
    --backend URL   : Calls the backend /query API (requires running server)
    --local         : Imports backend modules directly (run from backend/ dir)
"""

import argparse
import csv
import json
import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ragas_eval")


def load_csv(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"].strip(),
                "ground_truth": row["ground_truth"].strip(),
            })
    return rows


def query_via_api(question: str, backend_url: str) -> dict:
    resp = requests.post(
        f"{backend_url}/query",
        json={"question": question, "chat_history": []},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def run_api_evaluation(csv_path: str, backend_url: str, output_path: str):
    """Run evaluation by calling the backend API for each question."""
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    test_data = load_csv(csv_path)
    logger.info(f"Loaded {len(test_data)} test questions from {csv_path}")

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(test_data):
        question = item["question"]
        logger.info(f"[{i+1}/{len(test_data)}] Querying: {question}")
        try:
            result = query_via_api(question, backend_url)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            contexts = [s.get("title", "") + " " + s.get("url", "") for s in sources]
            if not contexts:
                contexts = ["No context retrieved."]
        except Exception as e:
            logger.error(f"API error: {e}")
            answer = f"Error: {e}"
            contexts = ["Error during retrieval."]

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    logger.info("Running Ragas evaluation...")

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    scores = {}
    for m in metrics:
        if m.name in result:
            scores[m.name] = round(result[m.name], 4)

    output = {
        "aggregate_scores": scores,
        "num_questions": len(questions),
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "per_question": result.to_pandas().to_dict(orient="records"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Aggregate scores: {scores}")
    return output


def run_local_evaluation(csv_path: str, output_path: str):
    """Run evaluation by importing backend modules directly."""
    backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
    sys.path.insert(0, backend_path)

    from evaluation import run_full_evaluation, generate_evaluation_report

    results = run_full_evaluation(csv_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    report = generate_evaluation_report(results)
    report_path = output_path.replace(".json", "_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Aggregate scores: {results['aggregate_scores']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on the Research Assistant")
    parser.add_argument("--csv", required=True, help="Path to test dataset CSV")
    parser.add_argument("--backend", type=str, default=None, help="Backend URL (API mode)")
    parser.add_argument("--local", action="store_true", help="Run locally importing backend modules")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    if args.local:
        run_local_evaluation(args.csv, args.output)
    elif args.backend:
        run_api_evaluation(args.csv, args.backend, args.output)
    else:
        logger.info("No mode specified. Defaulting to API mode with http://localhost:8000")
        run_api_evaluation(args.csv, "http://localhost:8000", args.output)


if __name__ == "__main__":
    main()
