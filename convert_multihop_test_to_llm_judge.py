"""Convert multihop reasoning data to LLM Judge format.

Usage:
python scripts/multihop/convert_multihop_test_to_llm_judge.py --raw-data data/musique/processed_test_ircot_subsampled_500.jsonl --bench-name musique --template data/prompts/coc.json
"""

import argparse
import json
import os
import fastchat
from tqdm import tqdm


from common import (
    alpaca_nonempty_input_fschat_template,
    from_sample_to_instance,
    get_prompt_from_instance,
    PromptTemplate,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data", type=str, help="The path to load raw multihop data (JSON-LINE)."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        help="The name to store converted multihop data (JSON-LINE).",
    )
    parser.add_argument(
        "--template",
        required=True,
        type=str,
        help="The path to load the prompt template (JSON).",
    )
    args = parser.parse_args()

    print(f"Loading multihop reasoning test set from {args.raw_data}")
    with open(args.raw_data, "r", encoding="utf-8") as fin:
        multihop_data = [json.loads(line.strip()) for line in fin.readlines()]
    print(f"{len(multihop_data)} samples loaded")

    print(f"Using prompt template from {args.template}")
    with open(args.template, "r", encoding="utf-8") as fin:
        template = PromptTemplate(**json.load(fin))
    print(json.dumps(template.__dict__, indent=2, ensure_ascii=False))

    fschat_data = []
    for idx, instance in enumerate(tqdm(multihop_data), start=1):
        instance = from_sample_to_instance(instance)
        input_str = get_prompt_from_instance(instance, template)
        prompt = alpaca_nonempty_input_fschat_template.format(
            instruction=template.instruction, input=input_str
        )
        fschat_data.append(
            {
                "question_id": idx,
                "category": "reasoning",
                "turns": [prompt],
                "reference": [instance.answers[0]],
            }
        )

    path = os.path.join(
        os.path.dirname(fastchat.__file__),
        "llm_judge",
        "data",
        args.bench_name,
        "question.jsonl",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(fschat_data)} attached data to {path}")
    with open(path, "w", encoding="utf-8") as fout:
        for line in fschat_data:
            fout.write(json.dumps(line) + "\n")
