from ovqa.metrics.lerc.allennlp.dataset_reader import DatasetReader, Instance
from ovqa.metrics.lerc.allennlp.model import Model
from ovqa.metrics.lerc.allennlp.predictor import Predictor

from ovqa.metrics.lerc.lerc_model.lerc_dataset_reader import LERCDatasetReader
from ovqa.metrics.lerc.lerc_model.lerc_model import LERC
from ovqa.metrics.lerc.lerc_model.pretrain_model import PretrainLERC

_, _, _ = LERC, LERCDatasetReader, PretrainLERC  # do not remove this line or the model will break


@Predictor.register("lerc", exist_ok=True)
class LERCPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def _json_to_instance(self, inputs) -> Instance:
        inputs = {
            "context": inputs["context"],
            "question": inputs["question"],
            "reference": inputs["reference"],
            "candidate": inputs["candidate"],
        }
        return self._dataset_reader.text_to_instance(**inputs)


def get_pretrained_lerc(device="cuda"):
    """
    Usage:
        # The instance we want to get LERC score for in a JSON format
        input_json = {
            'context': 'context string',
            'question': 'question string',
            'reference': 'reference string',
            'candidate': 'candidate string'
        }

        output_dict = predictor.predict_json(input_json)
        score = output_dict["pred_score"]
        # normalize to give 0-1 instead of 1-5
        score = max(min((score - 1) / 4, 1), 0)

    Returns:

    """
    predictor = Predictor.from_path(
        archive_path="https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz",
        predictor_name="lerc",
        cuda_device=0 if device == "cuda" else -1,  # cpu
    )
    return predictor
