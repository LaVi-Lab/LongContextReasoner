import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable


alpaca_nonempty_input_fschat_template = "{instruction}\nInput:\n{input}"


@dataclass(frozen=True)
class Evidence:
    title: str
    sentence: str
    identifier: Optional[str] = None


@dataclass(frozen=True)
class Thought:
    rationale: str
    identifier: Optional[str] = None
    quotation: Optional[str] = None


@dataclass(frozen=True, eq=False)
class Instance:
    qid: str
    question: str
    answers: List[str]
    evidences: List[Evidence]
    cot: Optional[List[Thought]] = None  # [Thought_1, ..., Thought_n]


@dataclass(frozen=True)
class PromptTemplate:
    # prompt part
    instruction: str
    question_prefix: str
    question_suffix: str
    answer_prefix: str
    context_prefix: str
    context_suffix: str
    evidence_template_str: str
    question_first: bool
    # answer part
    step_template_str: str
    rationale_prefix: str
    rationale_suffix: str


def from_sample_to_instance(
    sample: dict,
) -> Instance:
    # convert sample to instance
    # TODO: we can choose to shuffle the order
    counter = 1
    evidences: List[Evidence] = []
    for paragraph in sample["paragraphs"]:
        # TODO: chunk paragraphs to shorter passages
        evidences.append(
            Evidence(
                title=paragraph["title"],
                sentence=paragraph["paragraph_text"],
                identifier=f"[{counter}]",
            )
        )
        counter += 1

    # checking...
    assert "id" in sample
    assert type(sample["answer"]) is list

    if "reasoning_steps" in sample:
        cot = []
        for step in sample["reasoning_steps"]:
            related_evidences = [
                evidence
                for evidence in evidences
                if evidence.title == step["paragraphs"][0]["title"]
                and step["paragraphs"][0]["text_substring"] in evidence.sentence
            ]
            if len(related_evidences) > 1:
                print(f"WARNING: {sample['id']} contains duplicated documents")
            cot.append(
                Thought(
                    rationale=step["cot_sent"],
                    identifier=related_evidences[0].identifier
                    if len(related_evidences) > 0
                    else None,
                    quotation=step["paragraphs"][0]["text_substring"],
                )
            )
    else:
        cot = None

    input_instance = Instance(
        qid=sample["id"],
        question=sample["question"],
        answers=sample["answer"],
        evidences=evidences,
        cot=cot,
    )

    return input_instance


def get_prompt_from_instance(
    instance: Instance,
    template: PromptTemplate,
) -> str:
    """This function constructs prompt for a single given instance."""
    context_strs: List[str] = []
    for evidence in instance.evidences:
        sentence = template.evidence_template_str.format(
            identifier=evidence.identifier,
            title=evidence.title,
            sentence=evidence.sentence,
        )
        context_strs.append(sentence)

    context = template.context_prefix + "".join(context_strs) + template.context_suffix

    if template.question_first:
        prompt = (
            template.question_prefix
            + instance.question
            + template.question_suffix
            + context
            + template.answer_prefix
        )
    else:
        # https://docs.anthropic.com/claude/docs/claude-2p1-guide#prompting-techniques-for-claude-21
        prompt = (
            context
            + template.question_prefix
            + instance.question
            + template.question_suffix
            + template.answer_prefix
        )
    return prompt


def get_answer_from_instance(
    instance: Instance,
    template: PromptTemplate,
) -> str:
    answer = instance.answers[0]
    if instance.cot:
        rationale = ""
        for thought in instance.cot:
            rationale += template.step_template_str.format(
                rationale=thought.rationale,
                quotation=thought.quotation,
                identifier=thought.identifier,
            )
        answer = (
            template.rationale_prefix + rationale + template.rationale_suffix + answer
        )
    return answer


def augment_sample(
    sample: dict,
    max_subsample_context: int,
) -> List[dict]:
    supporting = []
    distractors = []
    for paragraph in sample["paragraphs"]:
        if paragraph["is_supporting"]:
            supporting.append(paragraph)
        else:
            distractors.append(paragraph)

    augmented_samples = []
    for i in range(max_subsample_context):
        num_distractors = random.randint(0, len(distractors))
        sampled_distractors = random.sample(deepcopy(distractors), k=num_distractors)
        paragraphs = deepcopy(supporting) + sampled_distractors
        # even we sample two identical sets of paragraphs, shuffling will make a different
        random.shuffle(paragraphs)
        augmented_samples.append(
            {
                "id": sample["id"] + f"__{i}",
                "question": deepcopy(sample["question"]),
                "answer": deepcopy(sample["answer"]),
                "paragraphs": paragraphs,
                "reasoning_steps": deepcopy(sample["reasoning_steps"]),
            }
        )
    return augmented_samples


def quotation_identification(
    sample: dict, template: PromptTemplate
) -> List[Tuple[str, str]]:
    """This task asks the model to find one quotation from the passages for answering a question."""
    instance = from_sample_to_instance(sample)
    input_str = get_prompt_from_instance(instance, template)
    prompt = alpaca_nonempty_input_fschat_template.format(
        instruction=template.instruction, input=input_str
    )
    qa_pairs = []
    for cot in instance.cot:
        qa_pairs.append(
            (
                prompt,
                template.step_template_str.format(
                    identifier=cot.identifier, quotation=cot.quotation
                ),
            )
        )
    return qa_pairs


def quotation_identification_all(
    sample: dict, template: PromptTemplate
) -> List[Tuple[str, str]]:
    """This task asks the model to find all quotations from the passages for answering a question."""
    instance = from_sample_to_instance(sample)
    input_str = get_prompt_from_instance(instance, template)
    prompt = alpaca_nonempty_input_fschat_template.format(
        instruction=template.instruction, input=input_str
    )

    quotes = ""
    for thought in instance.cot:
        quotes += template.step_template_str.format(
            rationale=thought.rationale,
            quotation=thought.quotation,
            identifier=thought.identifier,
        )
    answer = template.rationale_prefix + quotes + template.rationale_suffix
    return [
        (prompt, answer),
    ]


TASK2FUNC: Dict[str, Callable[[dict, PromptTemplate], List[Tuple[str, str]]]] = {
    "quotation_identification": quotation_identification,
    "quotation_identification_all": quotation_identification_all,
}
