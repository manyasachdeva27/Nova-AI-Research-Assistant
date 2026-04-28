import os
import csv
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from agent import run_agent

logger = logging.getLogger("research_assistant.evaluation")

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"


def _get_eval_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        base_url=OLLAMA_BASE_URL,
    )


def _get_eval_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_test_dataset(csv_path: str) -> List[dict]:
    """Load test questions and ground truth from a CSV file."""
    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"].strip(),
                "ground_truth": row["ground_truth"].strip(),
            })
    return rows


def run_pipeline_for_evaluation(test_data: List[dict]) -> dict:
    """Run the RAG pipeline on each test question and collect results for Ragas."""
    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []

    for i, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        logger.info(f"Evaluating question {i + 1}/{len(test_data)}: {question}")

        try:
            result = run_agent(question, chat_history=[])
            answer = result["answer"]
            raw_chunks = result.get("raw_chunks", [])
            contexts = [
                chunk.get("content", "") for chunk in raw_chunks if chunk.get("content")
            ]
            if not contexts:
                contexts = ["No context retrieved."]
        except Exception as e:
            logger.error(f"Pipeline error for question '{question}': {e}")
            answer = f"Error: {e}"
            contexts = ["Error during retrieval."]

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def evaluate_with_ragas(
    pipeline_results: dict,
    metrics: Optional[List[str]] = None,
) -> dict:
    """Run Ragas evaluation on pipeline results and return scores."""
    dataset = Dataset.from_dict(pipeline_results)

    available_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    if metrics:
        selected = [available_metrics[m] for m in metrics if m in available_metrics]
    else:
        selected = list(available_metrics.values())

    if not selected:
        selected = list(available_metrics.values())

    llm = _get_eval_llm()
    embeddings = _get_eval_embeddings()

    logger.info(f"Running Ragas evaluation with metrics: {[m.name for m in selected]}")

    result = ragas_evaluate(
        dataset=dataset,
        metrics=selected,
        llm=llm,
        embeddings=embeddings,
    )

    scores = {}
    for metric in selected:
        metric_name = metric.name
        if metric_name in result:
            scores[metric_name] = round(result[metric_name], 4)

    per_question = []
    result_df = result.to_pandas()
    for _, row in result_df.iterrows():
        q_result = {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "ground_truth": row.get("ground_truth", ""),
        }
        for metric in selected:
            metric_name = metric.name
            if metric_name in row:
                q_result[metric_name] = (
                    round(float(row[metric_name]), 4)
                    if row[metric_name] is not None
                    else None
                )
        per_question.append(q_result)

    return {
        "aggregate_scores": scores,
        "per_question_scores": per_question,
        "num_questions": len(pipeline_results["question"]),
        "metrics_used": [m.name for m in selected],
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def run_full_evaluation(
    csv_path: str,
    metrics: Optional[List[str]] = None,
) -> dict:
    """End-to-end: load dataset -> run pipeline -> evaluate with Ragas."""
    logger.info(f"Loading test dataset from {csv_path}")
    test_data = load_test_dataset(csv_path)
    logger.info(f"Loaded {len(test_data)} test questions")

    logger.info("Running RAG pipeline on test questions...")
    pipeline_results = run_pipeline_for_evaluation(test_data)

    logger.info("Running Ragas evaluation...")
    evaluation_results = evaluate_with_ragas(pipeline_results, metrics)

    logger.info(f"Evaluation complete. Aggregate scores: {evaluation_results['aggregate_scores']}")
    return evaluation_results


def generate_evaluation_report(results: dict) -> str:
    """Generate a markdown report from evaluation results."""
    scores = results.get("aggregate_scores", {})
    per_q = results.get("per_question_scores", [])
    timestamp = results.get("evaluated_at", "N/A")
    num_q = results.get("num_questions", 0)

    report = f"""# RAG Evaluation Report (Ragas)

## Overview
- **Questions evaluated:** {num_q}
- **Metrics used:** {', '.join(results.get('metrics_used', []))}
- **Evaluated at:** {timestamp}

## Aggregate Scores

| Metric | Score |
|--------|-------|
"""
    for metric_name, score in scores.items():
        emoji = "🟢" if score >= 0.7 else ("🟡" if score >= 0.4 else "🔴")
        report += f"| {emoji} {metric_name} | {score:.4f} |\n"

    report += "\n## Per-Question Breakdown\n\n"
    for i, q in enumerate(per_q, 1):
        report += f"### Question {i}\n"
        report += f"**Q:** {q.get('question', 'N/A')}\n\n"
        report += f"**Answer:** {q.get('answer', 'N/A')[:200]}...\n\n"
        report += f"**Ground Truth:** {q.get('ground_truth', 'N/A')[:200]}...\n\n"
        report += "| Metric | Score |\n|--------|-------|\n"
        for metric_name in results.get("metrics_used", []):
            val = q.get(metric_name)
            if val is not None:
                report += f"| {metric_name} | {val:.4f} |\n"
            else:
                report += f"| {metric_name} | N/A |\n"
        report += "\n"

    return report
