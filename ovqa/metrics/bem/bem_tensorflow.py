"""
Tensorflow BEM Implementation

Download model:
https://tfhub.dev/google/answer_equivalence/bem

License: Apache-2.0

======================================================= TF Only

_CONDA_ENV=bem
conda deactivate
conda env remove -n ${_CONDA_ENV} -y
conda create -n ${_CONDA_ENV} python=3.9 -y
conda activate ${_CONDA_ENV}
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.14
pip install tensorflow-text==2.14 scipy flask

=======================================================

python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
python -c 'import torch; print(f"torch.cuda.is_available()={torch.cuda.is_available()}")'

"""
import numpy as np
import os

from scipy.special import softmax

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text

    tf_failed = None
except ImportError as e:
    tf_failed = e


class BEM:
    def __init__(self):
        if tf_failed is not None:
            raise RuntimeError(f"Cannot use BEM since tensorflow failed to import.") from tf_failed
        VOCAB_PATH = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt"  # @param {type:"string"}
        vocab_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=VOCAB_PATH,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            ),
            num_oov_buckets=1,
        )
        self.cls_id, self.sep_id = vocab_table.lookup(tf.convert_to_tensor(["[CLS]", "[SEP]"]))
        self.tokenizer = text.BertTokenizer(
            vocab_lookup_table=vocab_table,
            token_out_type=tf.int64,
            preserve_unused_token=True,
            lower_case=True,
        )
        self.bem = hub.load("https://tfhub.dev/google/answer_equivalence/bem/1")

    def bertify_example(self, example):
        question = self.tokenizer.tokenize(example["question"]).merge_dims(1, 2)
        reference = self.tokenizer.tokenize(example["reference"]).merge_dims(1, 2)
        candidate = self.tokenizer.tokenize(example["candidate"]).merge_dims(1, 2)

        input_ids, segment_ids = text.combine_segments(
            (candidate, reference, question), self.cls_id, self.sep_id
        )

        return {"input_ids": input_ids.numpy(), "segment_ids": segment_ids.numpy()}

    def pad(self, a, length=512):
        return np.append(a, np.zeros(length - a.shape[-1], np.int32))

    def bertify_examples(self, examples):
        input_ids = []
        segment_ids = []
        for example in examples:
            example_inputs = self.bertify_example(example)
            input_ids.append(self.pad(example_inputs["input_ids"]))
            segment_ids.append(self.pad(example_inputs["segment_ids"]))

        return {"input_ids": np.stack(input_ids), "segment_ids": np.stack(segment_ids)}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs_raw: list[dict[str, str]]) -> np.ndarray:
        """

        Args:
            inputs_raw: should look like
            [
                {
                    "question": "why is the sky blue",
                    "reference": "light scattering",
                    "candidate": "scattering of light",
                },
                {
                    "question": "why is the banana askew",
                    "reference": "it is growing towards the like",
                    "candidate": "because this protects them against elephants",
                }
            ]

        Returns:
            numpy array, shape (n_inputs,) of BEM scores, float in 0 to 1

        """
        inputs = self.bertify_examples(inputs_raw)
        # The outputs are raw logits.
        raw_outputs = self.bem(inputs)
        # They can be transformed into a classification 'probability' like so:

        # raw_outputs  # tensor shape (n_inputs, 2)
        # bem_score = softmax(np.squeeze(raw_outputs))[1]
        bem_score = softmax(raw_outputs, axis=1)[:, 1]
        return bem_score


def main():
    examples = [
        {
            "question": "why is the sky blue",
            "reference": "light scattering",
            "candidate": "scattering of light",
        },
        {
            "question": "why is the banana askew",
            "reference": "it is growing towards the sun",
            "candidate": "because this protects them against elephants",
        },
        {
            "question": "are there any elephants in the room",
            "reference": "yes",
            "candidate": "no",
        },
    ]

    bem = BEM()
    bem_score = bem(examples)
    print(f"BEM score: {bem_score}")

    os.system("nvidia-smi")
    breakpoint()
    print(f"Done")


if __name__ == "__main__":
    main()
