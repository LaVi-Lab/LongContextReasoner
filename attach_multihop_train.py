"""Attach multihop reasoning data to instruction tuning data.

Usage:
python attach_multihop_train.py --original-data data/alpaca_data_fschat.json --multihop-data MuSiQue-Attribute/train.jsonl --attached-data data/alpaca_musique_coc.json --template data/prompts/coc.json
"""
import argparse
import json
import random
from copy import deepcopy
from tqdm import tqdm

from common import (
    alpaca_nonempty_input_fschat_template,
    PromptTemplate,
    from_sample_to_instance,
    get_prompt_from_instance,
    get_answer_from_instance,
    augment_sample,
    TASK2FUNC,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-data",
        default=None,
        type=str,
        help="The path to load the original instruction tuning data (JSON).",
    )
    parser.add_argument(
        "--random-n",
        default=None,
        type=int,
        help="How many random samples taken from the original data.",
    )
    parser.add_argument(
        "--multihop-data",
        default=None,
        type=str,
        help="The path to load the to-be-attached multihop reasoning data (JSON-LINE).",
    )
    parser.add_argument(
        "--attached-data",
        required=True,
        type=str,
        help="The path to store the attached data (JSON).",
    )
    parser.add_argument(
        "--template",
        default=None,
        nargs="+",
        type=str,
        help="The path(s) to load the prompt template (JSON).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The random seed.",
    )
    parser.add_argument(
        "--max-template-per-instance",
        default=0,
        type=int,
        help="How many different prompt templates will be applied to each instance. 0 means use all.",
    )
    parser.add_argument(
        "--max-context-per-instance",
        default=0,
        type=int,
        help="How many different context will be sampled for each instance. 0 means not sampling.",
    )
    parser.add_argument(
        "--auxiliary-tasks",
        default=None,
        type=str,
        nargs="+",
        help="Which auxilary tasks will be used.",
    )
    parser.add_argument(
        "--subsample-size",
        default=None,
        type=int,
        help="How many examples from auxilary tasks will be used.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Whether to perform deduplication at the end.",
    )
    parser.add_argument(
        "--random-multihop",
        default=None,
        type=int,
        help="How many random samples taken from the multihop data.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.original_data:
        print(f"Loading instruction tuning data from {args.original_data}")
        with open(args.original_data, "r", encoding="utf-8") as fin:
            original_data = json.load(fin)
            if args.random_n:
                original_data = random.sample(original_data, k=args.random_n)
        print(f"{len(original_data)} samples loaded")
    else:
        original_data = []

    new_data = []
    if args.multihop_data:
        print(f"Loading multihop reasoning training data from {args.multihop_data}")
        with open(args.multihop_data, "r", encoding="utf-8") as fin:
            multihop_data = [json.loads(line.strip()) for line in fin.readlines()]
            if args.random_multihop:
                multihop_data = random.sample(multihop_data, k=args.random_multihop)
        print(f"{len(multihop_data)} samples loaded")

        idx = len(original_data)

        if args.template:
            templates = []
            for path in args.template:
                print(f"Using prompt template from {args.template}")
                with open(path, "r", encoding="utf-8") as fin:
                    template = PromptTemplate(**json.load(fin))
                print(json.dumps(template.__dict__, indent=2, ensure_ascii=False))
                templates.append(template)

            pbar = tqdm(
                total=len(multihop_data)
                * max(1, args.max_context_per_instance)
                * (
                    min(len(templates), args.max_template_per_instance)
                    if args.max_template_per_instance > 0
                    else len(templates)
                ),
                desc="Attaching data",
                position=0,
                leave=True,
            )
            for raw_instance in multihop_data:
                if (
                    len(templates) > args.max_template_per_instance
                    and args.max_template_per_instance > 0
                ):
                    templates = random.sample(templates, args.max_template_per_instance)
                for template in templates:
                    if args.max_context_per_instance > 0:
                        augmented_instances = augment_sample(
                            raw_instance, args.max_context_per_instance
                        )
                    else:
                        augmented_instances = [raw_instance]
                    for sample in augmented_instances:
                        instance = from_sample_to_instance(sample)
                        input_str = get_prompt_from_instance(instance, template)
                        prompt = alpaca_nonempty_input_fschat_template.format(
                            instruction=template.instruction, input=input_str
                        )
                        answer = get_answer_from_instance(instance, template)
                        new_data.append(
                            {
                                "id": f"alpaca_{idx}",
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": prompt,
                                    },
                                    {
                                        "from": "gpt",
                                        "value": answer,
                                    },
                                ],
                            }
                        )
                        idx += 1
                        pbar.update(1)
            else:
                pbar.close()
                print("An example:")
                print(json.dumps(new_data[-1], indent=2))

        if args.auxiliary_tasks:
            for param in args.auxiliary_tasks:
                task, path = param.split(":")
                print(f"Using prompt template from {path}")
                with open(path, "r", encoding="utf-8") as fin:
                    template = PromptTemplate(**json.load(fin))
                print(json.dumps(template.__dict__, indent=2, ensure_ascii=False))

                pbar = tqdm(
                    desc=f"Adding {task} data",
                    position=0,
                    leave=True,
                )
                for raw_instance in multihop_data:
                    if args.max_context_per_instance > 0:
                        augmented_instances = augment_sample(
                            raw_instance, args.max_context_per_instance
                        )
                    else:
                        augmented_instances = [raw_instance]
                    for sample in augmented_instances:
                        if args.subsample_size and args.subsample_size > 0:
                            pairs = random.sample(
                                TASK2FUNC[task](sample, template), k=args.subsample_size
                            )
                        else:
                            pairs = TASK2FUNC[task](sample, template)
                        for prompt, answer in pairs:
                            new_data.append(
                                {
                                    "id": f"alpaca_{idx}",
                                    "conversations": [
                                        {
                                            "from": "human",
                                            "value": prompt,
                                        },
                                        {
                                            "from": "gpt",
                                            "value": answer,
                                        },
                                    ],
                                }
                            )
                            idx += 1
                            pbar.update(1)
                else:
                    pbar.close()
                    print(f"An example of {task}:")
                    print(json.dumps(new_data[-1], indent=2))

    attached_data = original_data + new_data

    # deduplicate
    if args.deduplicate:
        unique_data = set()
        for sample in attached_data:
            sample_copy = deepcopy(sample)
            del sample["id"]
            unique_data.add(json.dumps(sample_copy))
        print(f"There are {len(attached_data)} before deduplication")
        print(f"There are {len(unique_data)} after deduplication")
        attached_data = []
        for idx, sample in enumerate(unique_data):
            sample_copy = json.loads(sample)
            sample_copy["id"] = f"alpaca_{idx}"
            attached_data.append(sample_copy)

    print(f"Saving {len(attached_data)} attached data to {args.attached_data}")
    with open(args.attached_data, "w", encoding="utf-8") as fout:
        json.dump(attached_data, fout, indent=2, ensure_ascii=False)
