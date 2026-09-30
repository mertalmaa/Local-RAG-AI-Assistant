"""Retrieval ve isteğe bağlı LLM cevap kalitesini ölçen değerlendirme aracı."""

import argparse
import json
import time
from pathlib import Path

from ingest import BASE_DIR, load_embedding_model, prepare_database
from rag import (
    FALLBACK_ANSWER,
    ensure_foundry_ready,
    format_context,
    is_context_relevant,
    retrieve_chunks,
    stream_answer,
)


QUESTIONS_PATH = BASE_DIR / "evaluation_questions.json"
REPORT_PATH = BASE_DIR / "evaluation_report.json"


def load_test_cases(limit=None):
    test_cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return test_cases[:limit] if limit else test_cases


def contains_expected_terms(answer, expected_terms):
    normalized_answer = answer.casefold()
    return all(term.casefold() in normalized_answer for term in expected_terms)


def evaluate_case(test_case, collection, embedding_model, with_llm):
    started_at = time.perf_counter()
    chunks = retrieve_chunks(collection, embedding_model, test_case["question"])
    retrieval_seconds = time.perf_counter() - started_at

    retrieved_sources = [chunk["metadata"]["source"] for chunk in chunks]
    expected_source = test_case["expected_source"]
    context_is_relevant = is_context_relevant(chunks)
    retrieval_hit = (
        expected_source in retrieved_sources if test_case["answerable"] else None
    )

    result = {
        "id": test_case["id"],
        "question": test_case["question"],
        "answerable": test_case["answerable"],
        "expected_source": expected_source,
        "retrieval_hit": retrieval_hit,
        "context_is_relevant": context_is_relevant,
        "retrieval_seconds": round(retrieval_seconds, 3),
        "retrieved_chunks": [
            {
                "source": chunk["metadata"]["source"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "distance": round(chunk["distance"], 4),
            }
            for chunk in chunks
        ],
    }

    if with_llm:
        generation_started_at = time.perf_counter()
        if context_is_relevant:
            answer = "".join(
                stream_answer(test_case["question"], format_context(chunks))
            ).strip()
        else:
            answer = FALLBACK_ANSWER
        generation_seconds = time.perf_counter() - generation_started_at

        if test_case["answerable"]:
            answer_pass = contains_expected_terms(
                answer,
                test_case["expected_terms"],
            )
        else:
            answer_pass = answer.startswith(FALLBACK_ANSWER)

        result.update(
            {
                "answer": answer,
                "answer_pass": answer_pass,
                "generation_seconds": round(generation_seconds, 3),
            }
        )

    return result


def print_case_result(result, with_llm):
    if result["answerable"]:
        retrieval_status = "BAŞARILI" if result["retrieval_hit"] else "BAŞARISIZ"
    else:
        retrieval_status = "UYGULANMAZ"

    print(f"[{result['id']}] Retrieval: {retrieval_status}")
    print(f"  Soru: {result['question']}")
    print(f"  Süre: {result['retrieval_seconds']} saniye")
    print(
        "  Parçalar: "
        + ", ".join(
            f"{chunk['source']}#{chunk['chunk_index']}"
            for chunk in result["retrieved_chunks"]
        )
    )

    if with_llm:
        answer_status = "BAŞARILI" if result["answer_pass"] else "BAŞARISIZ"
        print(f"  Cevap: {answer_status} ({result['generation_seconds']} saniye)")
        print(f"  Model çıktısı: {result['answer']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Local RAG değerlendirmesi")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Retrieval yanında Foundry cevaplarını da değerlendirir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Yalnızca ilk N test vakasını çalıştırır.",
    )
    arguments = parser.parse_args()

    embedding_model = load_embedding_model()
    database_target = prepare_database(embedding_model)
    if arguments.with_llm:
        ensure_foundry_ready()

    test_cases = load_test_cases(arguments.limit)
    results = []

    for test_case in test_cases:
        result = evaluate_case(
            test_case,
            database_target,
            embedding_model,
            arguments.with_llm,
        )
        results.append(result)
        print_case_result(result, arguments.with_llm)

    answerable_results = [result for result in results if result["answerable"]]
    retrieval_passed = sum(result["retrieval_hit"] for result in answerable_results)

    summary = {
        "test_count": len(results),
        "answerable_test_count": len(answerable_results),
        "retrieval_passed": retrieval_passed,
        "retrieval_accuracy": round(
            retrieval_passed / len(answerable_results),
            3,
        )
        if answerable_results
        else None,
    }
    if arguments.with_llm:
        answer_passed = sum(result["answer_pass"] for result in results)
        summary.update(
            {
                "answer_passed": answer_passed,
                "answer_accuracy": round(answer_passed / len(results), 3),
            }
        )

    report = {
        "with_llm": arguments.with_llm,
        "summary": summary,
        "results": results,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Özet:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Rapor: {REPORT_PATH}")


if __name__ == "__main__":
    main()
