from ovqa.metrics.llm.context_vqa import CONTEXT_VQA

vqa_intro = """You are an annotator that judges the output of a QA system. Instructions:

1. Read the question, the correct answer and the predicted answer.
2. Select the score that best reflects how closely the predicted answer captures the same information as the correct answer.

Note: Please try to not think about what the image could have looked like. This is a text-only task. All you know about the image is the content of the reference.
In the rare cases where the question does not make sense, simply compare the predicted answer to the correct answer and ignore the question.

Scores:

1: completely wrong
2: mostly wrong
3: half right
4: mostly right
5: completely right
"""
vqa_prompt = "Question: {question}\nCandidate: {candidate}\nReference: {reference}\nVote: "

prompt_start = "Question: {question}\nCandidate: {candidate}\nReference: {reference}\n"
chat_add = (
    "The user will provide question, candidate and reference "
    "and you should output 'Vote: N' and replace N with your vote from 1 to 5."
)

MODES = {
    "vqa": {
        "intro": vqa_intro + "\n",
        "incontext_prompt": vqa_prompt + "{vote}\n\n",
        "output_prompt": vqa_prompt,
        "incontext_data": CONTEXT_VQA,
    },
    "vqachat": {
        "intro": "[INST] <<SYS>>\n" + vqa_intro + "<</SYS>>\n\n",
        "incontext_prompt_first": f"{prompt_start} [/INST] " + "Vote: {vote}\n</s>",
        "incontext_prompt": f"<s>[INST] {prompt_start} [/INST] " + "Vote: {vote}\n</s>",
        "output_prompt": f"<s>[INST] {prompt_start} [/INST] " + "Vote: ",
        "incontext_data": CONTEXT_VQA,
    },
    "vqachat2": {
        "intro": "[INST] <<SYS>>\n" + vqa_intro + f"\n{chat_add}\n" + "<</SYS>>\n\n",
        "incontext_prompt_first": f"{prompt_start} [/INST] " + "Vote: {vote}\n</s>",
        "incontext_prompt": f"<s>[INST] {prompt_start} [/INST] " + "Vote: {vote}\n</s>",
        "output_prompt": f"<s>[INST] {prompt_start} [/INST] " + "Vote: ",
        "incontext_data": CONTEXT_VQA,
    },
}
