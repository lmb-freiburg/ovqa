# ---------- dataset imagenet1k-square~val

# clip vit-l-14
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k-square~val/clip-openai~zs.yaml --skip_existing

# eva clip
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k-square~val/openclip-eva01~zs.yaml --skip_existing

# blip2 pretrained / coco
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k-square~val/blip2-nolm~pt~zs.yaml --skip_existing
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k-square~val/blip2-nolm~coco~zs.yaml --skip_existing

# ---------- dataset imagenet1k~val

# clip vit-l-14
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k~val/clip-openai~zs.yaml --skip_existing

# eva clip
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k~val/openclip-eva01~zs.yaml --skip_existing

# blip2
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k~val/blip2-nolm~pt~zs.yaml --skip_existing
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~imagenet1k~val/blip2-nolm~coco~zs.yaml --skip_existing
