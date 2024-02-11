# clip vit-l-14
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~activitynet~val/clip~openai~zs.yaml --skip_existing

# eva clip
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~activitynet~val/openclip-eva01~zs.yaml --skip_existing

# blip2
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~activitynet~val/blip2-nolm~pt~zs.yaml --skip_existing
python newevaluate.py -c ovqa/configs/retrieval_projects/retrieval~activitynet~val/blip2-nolm~coco~zs.yaml --skip_existing
