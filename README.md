# Making Long-Context Language Models Better Multi-Hop Reasoners

Official repo of "Making Long-Context Language Models Better Multi-Hop Reasoners" (ACL 2024)

## MuSiQue-Attribute Dataset

**MuSiQue-Attribute** is a subset of the original [MuSiQue](https://github.com/stonybrooknlp/musique) dataset with additional attribution annotations. There are 1,358 training examples in MuSiQue-Attribute. The data follows the same format of MuSiQue, with an additional `reasoning_steps` field. Below is an example of `reasoning_steps`:

```json
{
    ...,
    "reasoning_steps": [
        {
            "paragraphs": [
                {
                    "title": "CIMI-FM",
                    "text_substring": "She Did It"
                }
            ],
            "cot_sent": "The performer of \"She Did It\" is Eric Carmen",
        },
        {
            "paragraphs": [
                {
                    "title": "CIMI-FM",
                    "text_substring": "The Definitive Collection is a 1997 greatest hits compilation album of all the singles released by Cleveland, Ohio singer-songwriter Eric Carmen"
                }
            ],
            "cot_sent": "Eric Carmen was born in Cleveland, Ohio",
        },
        {
            "paragraphs": [
                {
                    "title": "Quebec Winter Carnival",
                    "text_substring": "Cleveland is a suburb of Chicago, located southwest of the city. It shares borders with the city in two areas, but is surrounded mostly by other suburbs"
                }
            ],
            "cot_sent": "The county that shares a border with Cuyahoga County, where Cleveland is located, is Lake County",
        },
    ]
}
```

The dataset is available at [`assets/MuSiQue-Attribute.zip`](assets/MuSiQue-Attribute.zip).

## Reproduction

Our code is based on [FastChat](https://github.com/lm-sys/FastChat).

### Installation

```sh
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
git checkout 722ab0299fd10221fa4686267fe068a688bacd4c
pip install --upgrade pip  # enable PEP 660 support
pip install -e ".[model_worker,llm_judge]"
pip install pytablewriter rouge_score nltk
cd ..
```

### Data Preparation

First, you should obtain all raw data for fine-tuning and evaluation:

1. Download the Alpaca-52K instruction tuning data from <https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json> and save it to `data/alpaca_data.json`.
2. Unzip MuSiQue-Attribute data into `MuSiQue-Attribute/train.jsonl`.
    ```sh
    unzip assets/MuSiQue-Attribute.zip
    ```
3. Download the subsampled dev & test sets from [IRCoT](https://github.com/StonyBrookNLP/ircot):
    ```sh
    git clone https://github.com/StonyBrookNLP/ircot.git
    ./ircot/download/processed_data.sh
    ```

Then, run the following command to convert Alpaca-52K to FastChat format.
```sh
python -m fastchat.data.convert_alpaca --in data/alpaca_data.json --out data/alpaca_data_fschat.json
```

Finally, run the following commands to get our fine-tuning data.
```sh
python attach_multihop_train.py --original-data data/alpaca_data_fschat.json --random-n 7200 --multihop-data MuSiQue-Attribute/train.jsonl --template prompts/ao.json --max-context-per-instance 2 --attached-data data/alpaca-7200-musique-ao-2.json
python attach_multihop_train.py --original-data data/alpaca-7200-musique-ao-2.json --multihop-data MuSiQue-Attribute/train.jsonl --template prompts/cot.json --max-context-per-instance 2 --attached-data data/alpaca-7200-musique-ao-2-cot-2.json
python attach_multihop_train.py --original-data data/alpaca-7200-musique-ao-2-cot-2.json --multihop-data MuSiQue-Attribute/train.jsonl --template prompts/coc.json --max-context-per-instance 2 --attached-data data/alpaca-7200-musique-ao-2-cot-2-coc-2.json
python attach_multihop_train.py --original-data data/alpaca-7200-musique-ao-2-cot-2-coc-2.json --multihop-data MuSiQue-Attribute/train.jsonl --auxiliary-tasks quotation_identification_all:prompts/qia.json --max-context-per-instance 1 --attached-data data/alpaca-7200-musique-ao-2-cot-2-coc-2-qia-1.json
```

For the evaluation data, run the following command to convert the subsampled, preprocessed MuSiQue test set data to LLM Judge format.
```sh
python convert_multihop_test_to_llm_judge.py --raw-data processed_data/musique/test_subsampled.jsonl --bench-name musique-coc --template prompts/coc.json
python convert_multihop_test_to_llm_judge.py --raw-data processed_data/musique/test_subsampled.jsonl --bench-name musique-cot --template prompts/cot.json
python convert_multihop_test_to_llm_judge.py --raw-data processed_data/musique/test_subsampled.jsonl --bench-name musique-ao --template prompts/ao.json
```
Test data will be stored into `fastchat/llm_judge/data/{bench-name}`.

### Fine-tuning

You can use the following command to replicate our fine-tuned model on 8 NVIDIA A100 80G:

```sh
bash scripts/train.sh data/alpaca-7200-musique-ao-2-cot-2-coc-2-qia-1.json <MODEL_NAME>
```

We also release the model weight of AttrLoRA in [`assets/AttrLoRA.zip`](assets/AttrLoRA.zip)

### Evaluation

To evaluate the model on MuSiQue, run the command:

```sh
bash eval.sh <MODEL_NAME>
```
