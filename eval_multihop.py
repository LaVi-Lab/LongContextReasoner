"""Generate judgements for multihop reasoning with conventional metrics like Exact Match.

Usage:
python eval_multihop.py --model-list [LIST-OF-MODEL-ID]
"""

import argparse
import string
import re
import os
import json

import numpy as np
from collections import defaultdict
from typing import List, Dict
from pytablewriter import MarkdownTableWriter
from rouge_score import rouge_scorer
from nltk.metrics.scores import f_measure
import fastchat
from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    get_model_list,
)


def check_data(questions, model_answers):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"


def extract_answer(cot: str, indicator: str) -> str:
    # TODO: allow specifying `stop generation token` (e.g., "\n") from outside
    # model is likely to synthesize fake examples after the first "\n", we use the first part
    answer = cot
    for term in ["\n"]:
        answer = answer.split(term)[0]
    answer = answer.split(indicator)[-1].strip()
    for term in ["."]:
        answer = answer.split(term)[0]
    return answer


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


# https://github.com/stanford-crfm/helm/blob/9be35a339347a9f2ad5644d7b72aede57486e3d4/src/helm/benchmark/metrics/basic_metrics.py#L183
def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(
        set(normalize_text(gold).split()), set(normalize_text(pred).split())
    )
    if ret is None:  # answer is the empty string after normalizing
        return 0.0
    return ret


def evaluate_single(refs: List[str], pred: str) -> Dict[str, float]:
    ANSWER_INDICATOR = "The answer is:"
    normalized_answers = [normalize_text(ref) for ref in refs]
    extracted_normalized_pred = normalize_text(extract_answer(pred, ANSWER_INDICATOR))

    results = {}

    if extracted_normalized_pred in normalized_answers:
        em = 1.0
    else:
        em = 0.0
    results["em"] = em

    f1 = np.max(
        [
            f1_score(gold=ref, pred=extracted_normalized_pred)
            for ref in normalized_answers
        ]
    )
    results["f1"] = f1

    rouge_types = ["rouge1", "rouge2", "rougeL"]
    for rouge_type in rouge_types:
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        score = np.max(
            [
                scorer.score(ref, extracted_normalized_pred)[rouge_type].fmeasure
                for ref in normalized_answers
            ]
        )
        results[rouge_type] = score

    return results


def make_table(result: Dict[str, Dict[str, float]], title="Overall_results"):
    writer = MarkdownTableWriter()
    metric_names = list(result[list(result.keys())[0]].keys())
    writer.headers = ["Model"] + metric_names
    values = []
    for k, v in result.items():
        row = [k]
        for metric_name in metric_names:
            row.append("{:.4f}".format(v[metric_name]))
        values.append(row)
    writer.value_matrix = values
    print(title + "\n")
    print(writer.dumps())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="musique",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--subtype_evaluation",
        action="store_true",
        help="evaluation on subtype",
    )
    args = parser.parse_args()

    question_file = os.path.join(
        os.path.dirname(fastchat.__file__),
        "llm_judge",
        "data",
        args.bench_name,
        "question.jsonl",
    )
    answer_dir = os.path.join(
        os.path.dirname(fastchat.__file__),
        "llm_judge",
        "data",
        args.bench_name,
        "model_answer",
    )

    # Load questions
    questions = load_questions(question_file, None, None)
    references = {
        question["question_id"]: question["reference"] for question in questions
    }
    question_idmap = {
        question["question_id"]: qid for qid, question in enumerate(questions)
    }

    # Load answers
    model_answers = load_model_answers(answer_dir)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    check_data(questions, model_answers)

    # Evaluate all answers
    model_results = {}
    subtype_model_results = defaultdict(lambda: {})
    for model in sorted(models):
        overall_result = defaultdict(lambda: 0)
        subtype_result = defaultdict(lambda: {})
        for qid, pred in model_answers[model].items():
            assert qid == pred["question_id"]
            current_q = questions[question_idmap[qid]]
            single_result = evaluate_single(
                references[qid], pred["choices"][0]["turns"][0]
            )
            for k, v in single_result.items():
                overall_result[k] += single_result[k]
                if args.subtype_evaluation:
                    subtype_result[current_q["qtype"]][k] = (
                        subtype_result[current_q["qtype"]].get(k, 0) + single_result[k]
                    )
            overall_result["count"] += 1
            if args.subtype_evaluation:
                subtype_result[current_q["qtype"]]["count"] = (
                    subtype_result[current_q["qtype"]].get("count", 0) + 1
                )

        normalized_overall_result = {}
        for k, v in overall_result.items():
            if k not in ["count"]:
                normalized_overall_result[k] = v / overall_result["count"]

        model_results[model] = normalized_overall_result

        # subtype
        if args.subtype_evaluation:
            normalized_subtype_result = defaultdict(lambda: {})

            for subtype, subresult in subtype_result.items():
                # k is different subtype
                for k, v in subresult.items():
                    if k not in ["count"]:
                        normalized_subtype_result[subtype][k] = v / subresult["count"]

                subtype_model_results[subtype][model] = normalized_subtype_result[
                    subtype
                ]

    # Show results
    make_table(model_results)
    if args.subtype_evaluation:
        for subtype in sorted(list(subtype_model_results.keys())):
            result = subtype_model_results[subtype]
            make_table(result, title=f"subtype: {subtype}")
