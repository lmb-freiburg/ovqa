#for CONFIG_NAME in \
#        "blip1~ftvqa~default~none" \
#        "blip2-opt2.7b~pt~default~qa-short" \
#        "blip2-opt2.7b~pt~vqa~qa-short" \
#        "blip2-t5xl~pt~default~qa-short" \
#        "blip2-t5xl~pt~vqa~qa-short" \
#        "iblip-t5xl~pt~norep-b1~qa-short" \
#        "iblip-vic7b~pt~norep-b1~qa-short" \
#        "llava7b~default~llava-nosample~conv-multimodal" \
#        "x2vlm-base1b~ftvqa~default~none" \
#        "x2vlm-large1b~ftvqa~default~none" \
#        ; do
#    CONFIG_DIR="ovqa/configs/projects/gqa~balanced_testdev"
#    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" --skip_existing $*
#done
#echo
#exit

python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/blip1~ftvqa~default~none.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/blip2-opt2.7b~pt~default~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/blip2-opt2.7b~pt~vqa~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/blip2-t5xl~pt~default~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/blip2-t5xl~pt~vqa~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/iblip-t5xl~pt~norep-b1~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/iblip-vic7b~pt~norep-b1~qa-short.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/llava7b~default~llava-nosample~conv-multimodal.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/x2vlm-base1b~ftvqa~default~none.yaml --skip_existing
python newevaluate.py -c ovqa/configs/projects/gqa~balanced_testdev/x2vlm-large1b~ftvqa~default~none.yaml --skip_existing
