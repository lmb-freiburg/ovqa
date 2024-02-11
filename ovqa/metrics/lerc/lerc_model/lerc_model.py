import logging
import torch
from ovqa.metrics.lerc.allennlp.vocabulary import Vocabulary
from ovqa.metrics.lerc.allennlp.archival import load_archive
from ovqa.metrics.lerc.allennlp.model import Model
from ovqa.metrics.lerc.allennlp.initializers import InitializerApplicator
from transformers import BertModel
from typing import Dict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("lerc", exist_ok=True)
class LERC(Model):
    @property
    def embedding_dim(self):
        return self.bert.embeddings.word_embeddings.embedding_dim

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        pretrained_archive_path: str = None,
        vocab=Vocabulary(),
        initializer=InitializerApplicator(),
    ) -> None:
        super(LERC, self).__init__(vocab)
        if pretrained_archive_path:
            logger.info("Loading pretrained: %s", pretrained_archive_path)
            archive = load_archive(pretrained_archive_path)
            self.bert = archive.model.bert
        else:
            self.bert = BertModel.from_pretrained(bert_model)

        self.score_layer = torch.nn.Linear(self.embedding_dim, 1)
        self.loss = torch.nn.MSELoss()
        initializer(self)

    #
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        score: torch.Tensor = None,
        metadata: Dict = None,
    ) -> Dict:
        output_dict = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output = output_dict["last_hidden_state"]
        cls_output = output[:, 0, :]
        pred_score = self.score_layer(cls_output).squeeze(-1)

        output_dict = {"pred_score": pred_score, "metadata": metadata}

        if score is not None:
            score = score.float()
            output_dict["loss"] = self.loss(pred_score, score)
            output_dict["score"] = score

        return output_dict
