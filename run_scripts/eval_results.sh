# --llm = add llama2 metric
# --include_models_file file.txt  # one line per model name to evaluate

# imagenet1k
python view_results_cls.py -d imagenet1k-square -s val --std_mode question
python view_results_cls.py -d imagenet1k -s val --std_mode question

# activitynet
python view_results_cls.py -d activitynet -s val --std_mode question

# coco
python view_results_cls.py -d coco -s val --std_mode question

# ovad
python view_results_ovad.py -d ovad_attributes -s val --std_mode question

# gqa
python view_results_vqa.py -d gqa -s balanced_testdev --split_model_names

# vqav2
python view_results_vqa.py -d vqav2 -s val --split_model_names

# retrieval
python view_results_ret.py -d imagenet1k-square -s val
python view_results_ret.py -d imagenet1k -s val
python view_results_ret.py -d activitynet -s val
python view_results_ret.py -d coco -s val
