#for CONFIG_NAME in \
#        "blip1~ftvqa-trainset~default~none" \
#        "blip2-opt2.7b~pt~default~qa-short" \
#        "blip2-opt2.7b~pt~vqa~qa-short" \
#        "blip2-t5xl~pt~default~qa-short" \
#        "blip2-t5xl~pt~vqa~qa-short" \
#        "iblip-t5xl~pt-trainset~norep-b1~qa-short" \
#        "iblip-vic7b~pt-trainset~norep-b1~qa-short" \
#        "llava7b~default~llava-nosample~conv-multimodal" \
#        "x2vlm-base1b~ftvqa-trainset~default~none" \
#        "x2vlm-large1b~ftvqa-trainset~default~none" \
#        ; do
#    CONFIG_DIR="ovqa/configs/projects/vqav2~val"
#    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" --skip_existing $*
#done
#echo
#exit

python newevaluate.py -c ovqa/configs/projects/vqav2~val/blip1~ftvqa-trainset~default~none.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/blip2-opt2.7b~pt~default~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/blip2-opt2.7b~pt~vqa~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/blip2-t5xl~pt~default~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/blip2-t5xl~pt~vqa~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/iblip-t5xl~pt-trainset~norep-b1~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/iblip-vic7b~pt-trainset~norep-b1~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/llava7b~default~llava-nosample~conv-multimodal.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/x2vlm-base1b~ftvqa-trainset~default~none.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/vqav2~val/x2vlm-large1b~ftvqa-trainset~default~none.yaml --skip_existing
