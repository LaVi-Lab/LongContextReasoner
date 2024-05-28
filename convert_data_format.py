import argparse
import os, json
from typing import List, Dict
from tqdm import tqdm
from rapidfuzz import fuzz
import _jsonnet
import re


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [
            json.loads(line.strip()) for line in file.readlines() if line.strip()
        ]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance, ensure_ascii=False) + "\n")


# from processed data
def convert_data_format(filename):
    print("[INFO] reading data from ", filename)
    data = read_jsonl(filename)
    new_data = []
    for d in data:
        assert len(d["answers_objects"]) == 1
        tmp = {
            "id": d["question_id"],
            "paragraphs": d["contexts"],
            "question": d["question_text"],
            "answer": d["answers_objects"][0]["spans"],
            "article_id": d["question_id"],
        }
        new_data.append(tmp)

    fname_list = filename.split("/")
    fname_list[-1] = "processed_" + fname_list[-1]
    output_fname = "/".join(fname_list)
    print("[INFO] Writing to:", output_fname)
    write_jsonl(new_data, output_fname)
    return new_data


# from raw data
# 2wikimultihopqa
def convert_2wikimultihopqa_raw_data(input_filename: str, output_filepath: str):
    print(f"[INFO] Reading data from: {input_filename}")
    with open(input_filename, "r") as file:
        raw_instances = json.load(file)

    processed_instances = []
    for raw_instance in raw_instances:
        raw_contexts = raw_instance["context"]

        supporting_titles = list(set([e[0] for e in raw_instance["supporting_facts"]]))

        processed_contexts = []
        for index, raw_context in enumerate(raw_contexts):
            title = raw_context[0]
            paragraph_text = " ".join(raw_context[1]).strip()
            is_supporting = title in supporting_titles
            processed_contexts.append(
                {
                    "idx": index,
                    "title": title.strip(),
                    "paragraph_text": paragraph_text,
                    "is_supporting": is_supporting,
                }
            )

        # context
        context = {}

        context["gold"] = [
            (p["title"], p["paragraph_text"])
            for p in processed_contexts
            if p["is_supporting"]
        ]

        context["provided"] = [
            (p["title"], p["paragraph_text"]) for p in processed_contexts
        ]

        processed_instance = {
            "id": raw_instance["_id"],
            "type": raw_instance["type"],
            "paragraphs": processed_contexts,
            "question": raw_instance["question"],
            "answer": [raw_instance["answer"]],
            "article_id": raw_instance["_id"],
            "context": context,
        }
        processed_instances.append(processed_instance)

    print(f"[INFO] Writing in: {output_filepath}")
    write_jsonl(processed_instances, output_filepath)
    return processed_instances


# hotpotqa
def convert_hotpotqa_raw_data(input_filename: str, output_filepath: str):
    print(f"[INFO] Reading data from: {input_filename}")
    with open(input_filename, "r") as file:
        raw_instances = json.load(file)

    max_num_tokens = 1000  # clip later.
    processed_instances = []

    for raw_instance in raw_instances:
        raw_contexts = raw_instance["context"]

        supporting_titles = [e[0] for e in raw_instance["supporting_facts"]]

        title_to_paragraph = {title: "".join(text) for title, text in raw_contexts}
        paragraph_to_title = {"".join(text): title for title, text in raw_contexts}

        gold_paragraph_texts = [
            title_to_paragraph[title] for title in supporting_titles
        ]
        gold_paragraph_texts = set(list(gold_paragraph_texts))

        paragraph_texts = ["".join(paragraph) for title, paragraph in raw_contexts]
        paragraph_texts = list(set(paragraph_texts))

        processed_contexts = [
            {
                "idx": index,
                "title": paragraph_to_title[paragraph_text].strip(),
                "paragraph_text": paragraph_text.strip(),
                "is_supporting": paragraph_text in gold_paragraph_texts,
            }
            for index, paragraph_text in enumerate(paragraph_texts)
        ]
        for context in processed_contexts:
            context["paragraph_text"] = " ".join(
                context["paragraph_text"].split(" ")[:max_num_tokens]
            )

        # context
        context = {}

        context["gold"] = [
            (p["title"], p["paragraph_text"])
            for p in processed_contexts
            if p["is_supporting"]
        ]

        context["provided"] = [
            (p["title"], p["paragraph_text"]) for p in processed_contexts
        ]

        processed_instance = {
            "id": raw_instance["_id"],
            "type": raw_instance["type"],
            "paragraphs": processed_contexts,
            "question": raw_instance["question"],
            "answer": [raw_instance["answer"]],
            "article_id": raw_instance["_id"],
            "context": context,
        }
        processed_instances.append(processed_instance)

    print(f"[INFO] Writing in: {output_filepath}")
    write_jsonl(processed_instances, output_filepath)
    return processed_instances


# musique
def convert_musique_raw_data(input_filename: str, output_filepath: str):
    print(f"[INFO] Reading data from: {input_filename}")
    raw_instances = read_jsonl(input_filename)
    processed_instances = []
    for raw_instance in raw_instances:
        processed_instance = {
            "id": raw_instance["id"],
            "type": raw_instance["id"][:4],  # 2hop,3hop,4hop
            "paragraphs": [
                {
                    "idx": index,
                    "paragraph_text": paragraph["paragraph_text"].strip(),
                    "title": paragraph["title"].strip(),
                    "is_supporting": paragraph["is_supporting"],
                }
                for index, paragraph in enumerate(raw_instance["paragraphs"])
            ],
            "question": raw_instance["question"],
            "answer": [raw_instance["answer"]],
            "article_id": raw_instance["id"],
        }

        # context
        context = {}
        tmp_ids = [
            x["paragraph_support_idx"] for x in raw_instance["question_decomposition"]
        ]

        context["gold"] = [
            (
                processed_instance["paragraphs"][xid]["title"],
                processed_instance["paragraphs"][xid]["paragraph_text"],
            )
            for xid in tmp_ids
        ]
        context["provided"] = [
            (p["title"], p["paragraph_text"]) for p in processed_instance["paragraphs"]
        ]

        processed_instance["context"] = context

        processed_instances.append(processed_instance)

    print(f"[INFO] Writing in: {output_filepath}")
    write_jsonl(processed_instances, output_filepath)
    return processed_instances


def get_subsampled(full_data: List, id_list: List, output_filepath: str):
    mapidx = {d["id"]: idx for idx, d in enumerate(full_data)}
    subsampled_data = [full_data[mapidx[i]] for i in id_list]
    write_jsonl(subsampled_data, output_filepath)
    print("[INFO] Writing subsampled in:", output_filepath)


def read_jsonnet(filepath: str):
    return json.loads(_jsonnet.evaluate_file(filepath))


def attach_annotation_data(dataset_dir: str, annotation_path: str):
    train_path = os.path.join(dataset_dir, f"processed_train.jsonl")
    ans_train = read_jsonl(train_path)

    annotations = read_jsonnet(annotation_path)

    id_to_annotation = {
        annotation["question_id"]: annotation for annotation in annotations
    }

    annotated_processed_data = []
    for instance in tqdm(ans_train):
        annotation = id_to_annotation.pop(instance["article_id"], None)

        if not annotation:
            continue

        assert instance["article_id"] == annotation["question_id"]
        question_id = instance["article_id"]

        question_match_score = fuzz.ratio(
            instance["question"], annotation["question_text"]
        )
        if question_match_score < 95:
            print(
                "WARNING the following questions may not be same. Check manually : "
                f'{instance["question"]} >>> {annotation["question_text"]}'
            )

        instance["question"] = annotation["question_text"]
        instance["anno_reasoning_steps"] = annotation["reasoning_steps"]
        anno_reasoning_steps = instance["anno_reasoning_steps"]

        answer_regex = r".*answer is: (.*)\."
        assert re.match(answer_regex, anno_reasoning_steps[-1]["cot_sent"])
        extracted_answer = re.sub(
            answer_regex, r"\1", anno_reasoning_steps[-1]["cot_sent"]
        )

        gold_answer = instance["answer"][0]
        if extracted_answer != gold_answer:
            print(
                f"WARNING: The extracted answer doesn't perfectly match the gold answer. "
                f"{extracted_answer} != {gold_answer}"
            )
        gold_answer = extracted_answer
        instance["answer"][0] = gold_answer

        context_paragraphs = instance["context"]["provided"]

        for paragraph in context_paragraphs:
            assert not paragraph[1].startswith("Title: ")
            assert not paragraph[1].startswith("Wikipedia Title: ")

        text_populated_reasoning_steps = []
        for reasoning_step in anno_reasoning_steps:
            # First, try to match it to the context_paragraphs.
            assert (
                len(reasoning_step["paragraphs"]) == 1
            )  # TODO: Make it single entry only.
            gold_paragraph = reasoning_step["paragraphs"][0]

            assert (
                "title" in gold_paragraph
            ), f"Field `title` missing in annotation for {question_id}"
            assert (
                "text_substring" in gold_paragraph
            ), f"Field `text_substring` missing in annotation for {question_id}"

            if not gold_paragraph["title"] or not gold_paragraph["text_substring"]:
                assert (
                    not gold_paragraph["title"] and not gold_paragraph["text_substring"]
                )
                gold_paragraph["paragraph_text"] = None
                text_populated_reasoning_steps.append(reasoning_step)
                continue

            matching_paragraphs = _find_matching_paragraphs(
                gold_paragraph["title"],
                gold_paragraph["text_substring"],
                context_paragraphs,
            )

            assert len(matching_paragraphs) < 2

            # Otherwise try to do a match based on retrieved paragraphs.
            # if not matching_paragraphs:
            #     retrieved_paragraphs = retriever.retrieve(gold_paragraph["title"], gold_paragraph["text_substring"])
            #     matching_paragraphs = _find_matching_paragraphs(
            #         gold_paragraph["title"], gold_paragraph["text_substring"], retrieved_paragraphs
            #     )

            if not matching_paragraphs:
                print("WARNING: Couldn't find any match for the annotated paragraph.")
                continue

            assert len(matching_paragraphs) == 1
            matching_paragraph = matching_paragraphs[0]

            assert gold_paragraph["title"].lower() == matching_paragraph[0].lower()
            gold_paragraph["paragraph_text"] = matching_paragraph[1]

            text_populated_reasoning_steps.append(reasoning_step)

        assert len(text_populated_reasoning_steps) == len(anno_reasoning_steps)
        instance["anno_reasoning_steps"] = text_populated_reasoning_steps
        annotated_processed_data.append(instance)

    save_path = os.path.join(dataset_dir, f"processed_train_ircot_annotated_20.jsonl")
    write_jsonl(annotated_processed_data, save_path)
    print("[INFO] Writing annotated data in:", save_path)


# reference: ircot/prompt_generator/attach_data_annotations.py
def _find_matching_paragraphs(
    query_title: str, query_text_substring: str, db_paragraphs: List[tuple]
) -> List[Dict]:
    assert isinstance(query_title, str)
    assert isinstance(query_text_substring, str)

    matching_paragraphs = []
    for paragraph in db_paragraphs:
        title_exact_match = query_title.lower().strip() == paragraph[0].lower().strip()
        paragraph_text_match_score = fuzz.partial_ratio(
            query_text_substring, paragraph[1]
        )

        if title_exact_match and paragraph_text_match_score > 95:
            matching_paragraphs.append(paragraph)

    return matching_paragraphs


DATAMAP = {
    "2wikimultihopqa": {
        "train": "train.json",
        "test": "test.json",
        "dev": "dev.json",
    },
    "hotpotqa": {
        "train": "hotpot_train_v1.1.json",
        "dev": "hotpot_dev_distractor_v1.json",
    },
    "musique": {
        "train": "musique_ans_v1.0_train.jsonl",
        "dev": "musique_ans_v1.0_dev.jsonl",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-datadir",
        type=str,
        help="The path to load raw multihop data (JSON-LINE).",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        help="The path to load annotation data from ircot (JSON-LINE).",
    )
    parser.add_argument(
        "--subsampled-path",
        type=str,
        default=None,
        help="The path to subsample raw multihop data (JSON-LINE).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The path to save multihop data (JSON-LINE).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if "2wikimultihopqa" in args.raw_datadir.lower():
        for set_name in ["train", "test", "dev"]:
            fname = DATAMAP["2wikimultihopqa"][set_name]

            processed_data = convert_2wikimultihopqa_raw_data(
                input_filename=os.path.join(args.raw_datadir, fname),
                output_filepath=os.path.join(
                    args.output_dir, f"processed_{set_name}.jsonl"
                ),
            )
            if set_name == "dev":
                get_subsampled(
                    full_data=processed_data,
                    id_list=[
                        sub_d["question_id"]
                        for sub_d in read_jsonl(args.subsampled_path)
                    ],
                    output_filepath=os.path.join(
                        args.output_dir, "processed_test_subsampled.jsonl"
                    ),
                )
        print("[INFO] Attaching annotation information...")
        attach_annotation_data(
            args.output_dir,
            annotation_path=args.annotation_dir + "/2wikimultihopqa.jsonnet",
        )

    elif "hotpotqa" in args.raw_datadir.lower():
        for set_name in ["train", "dev"]:
            fname = DATAMAP["hotpotqa"][set_name]
            processed_data = convert_hotpotqa_raw_data(
                input_filename=os.path.join(args.raw_datadir, fname),
                output_filepath=os.path.join(
                    args.output_dir, f"processed_{set_name}.jsonl"
                ),
            )
            if set_name == "dev":
                get_subsampled(
                    full_data=processed_data,
                    id_list=[
                        sub_d["question_id"]
                        for sub_d in read_jsonl(args.subsampled_path)
                    ],
                    output_filepath=os.path.join(
                        args.output_dir, "processed_test_subsampled.jsonl"
                    ),
                )
        print("[INFO] Attaching annotation information...")
        attach_annotation_data(
            args.output_dir,
            annotation_path=args.annotation_dir + "/hotpotqa.jsonnet",
        )

    elif "musique" in args.raw_datadir.lower():
        for set_name in ["train", "dev"]:
            fname = DATAMAP["musique"][set_name]
            processed_data = convert_musique_raw_data(
                input_filename=os.path.join(args.raw_datadir, fname),
                output_filepath=os.path.join(
                    args.output_dir, f"processed_{set_name}.jsonl"
                ),
            )
            if set_name == "dev":
                get_subsampled(
                    full_data=processed_data,
                    id_list=[
                        sub_d["question_id"]
                        for sub_d in read_jsonl(args.subsampled_path)
                    ],
                    output_filepath=os.path.join(
                        args.output_dir, "processed_test_subsampled.jsonl"
                    ),
                )
        print("[INFO] Attaching annotation information...")
        attach_annotation_data(
            args.output_dir,
            annotation_path=args.annotation_dir + "/musique.jsonnet",
        )
    else:
        raise NotImplementedError(
            "Only process 2wikimultihopqa, hotpotqa, and musique data."
        )
