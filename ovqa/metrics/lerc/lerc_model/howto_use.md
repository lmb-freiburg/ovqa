```python
from ovqa.metrics.lerc.lerc_model.lerc_predictor import get_pretrained_lerc

lerc = get_pretrained_lerc()

# context in the original paper is a text passage containing information necessary to answer 
# the question. it can be set to the empty string if not available / applicable.
out_dict = lerc.predict_json(
    dict(context=context, question=question, reference=answer, candidate=pred)
)
score = out_dict["pred_score"]
# output should be around 1-5 so we can normalize it to 0-1 like this.
# note that its a regressor so it might output slightly outside the 1-5 range.
score = max(min((score - 1) / 4, 1), 0)
```
