import logging
from typing import Dict, Any


def create_prompt(
    samples: Dict[str, Any],
    prompt: str,
    batch_size: int,
):
    """

    Args:
        samples: batch of data, maybe containing "text_input" with the question for each sample
        prompt: prompt to use for formatting the questions
        batch_size: batch size

    Returns:
        List of text inputs for the model
    """
    assert (
        "prompt" not in samples
    ), "Use field 'text_input' for per-datapoint prompts instead of 'prompt'"

    if "text_input" in samples:
        # assuming prompt in format "Question: {} Answer:"
        p = "{}"
        if p not in prompt:
            logging.warning(f"Prompt {prompt} has no placeholder {p}. Questions will be lost.")
        if isinstance(samples["text_input"][0], str):
            # single question per sample for predict_answer function
            text_batch = [prompt.format(question) for question in samples["text_input"]]
        elif isinstance(samples["text_input"][0], list):
            # multiple questions per sample for predict_multiquestion_answer function
            text_batch = [
                [prompt.format(question) for question in q_list] for q_list in samples["text_input"]
            ]
        else:
            raise ValueError(
                f"text_input {type(samples['text_input'][0])} not in the correct format. "
                f"possible formats: str and list"
            )

        return text_batch

    # assuming prompt is already the full prompt without placeholders
    assert "{}" not in prompt, (
        f"Prompt {prompt} has a placeholder but was not formatted. Either make your dataset "
        f"return 'text_input' or change the prompt to not contain a placeholder."
    )
    text_batch = [prompt] * batch_size
    return text_batch
