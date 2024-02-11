## setup OVQA_OUTPUT_DIR
#source <(python -m ovqa.bash_paths)
#if [ -z "$OVQA_OUTPUT_DIR" ]; then
#    echo "OVQA_OUTPUT_DIR not set, output directory will not be found"
#    exit 1
#fi
#
#DATASET="activitynet~val"
##for CONFIG_NAME in \
#for CONFIG_NAME in "blip1~ftvqa~default~none" \
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
#    echo ""
#    echo "# config ${CONFIG_NAME}"
#
#for PROMPT_NAME in "what-is-this" "what-is-happening-image" "what-act-is-this"; do
#    CONFIG_DIR="ovqa/configs/projects/${DATASET}"
#
#    SUFFIX="~${PROMPT_NAME}"
#    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
#        -d question_type="${PROMPT_NAME}" \
#        -o suffix_output_dir="${SUFFIX}" \
#        --skip_existing $*
#
#    F_DIR="${OVQA_OUTPUT_DIR}/${DATASET}/${CONFIG_NAME}${SUFFIX}"
#    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
#        -d question_type="${PROMPT_NAME}" \
#        -o suffix_output_dir="${SUFFIX}~followup" \
#        --add_config ovqa/configs/followup/followup_activitynet.yaml \
#        -o run.followup_prev_dir="${F_DIR}" \
#        --skip_existing $*
#
#done
#done
#echo
#exit

# config blip1~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip1~ftvqa~default~none~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip1~ftvqa~default~none~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip1~ftvqa~default~none.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip1~ftvqa~default~none.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip1~ftvqa~default~none~what-act-is-this --skip_existing

# config blip2-opt2.7b~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-opt2.7b~pt~default~qa-short~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-opt2.7b~pt~default~qa-short~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-opt2.7b~pt~default~qa-short~what-act-is-this --skip_existing

# config blip2-opt2.7b~pt~vqa~qa-short
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-opt2.7b~pt~vqa~qa-short~what-act-is-this --skip_existing

# config blip2-t5xl~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-t5xl~pt~default~qa-short~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-t5xl~pt~default~qa-short~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-t5xl~pt~default~qa-short~what-act-is-this --skip_existing

# config blip2-t5xl~pt~vqa~qa-short
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-t5xl~pt~vqa~qa-short~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-t5xl~pt~vqa~qa-short~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/blip2-t5xl~pt~vqa~qa-short~what-act-is-this --skip_existing

# config iblip-t5xl~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/iblip-t5xl~pt~norep-b1~qa-short~what-act-is-this --skip_existing

# config iblip-vic7b~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/iblip-vic7b~pt~norep-b1~qa-short~what-act-is-this --skip_existing

# config llava7b~default~llava-nosample~conv-multimodal
python newevaluate.py -c ovqa/configs/projects/activitynet~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/llava7b~default~llava-nosample~conv-multimodal~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/llava7b~default~llava-nosample~conv-multimodal~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/llava7b~default~llava-nosample~conv-multimodal~what-act-is-this --skip_existing

# config x2vlm-base1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/x2vlm-base1b~ftvqa~default~none~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/x2vlm-base1b~ftvqa~default~none~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/x2vlm-base1b~ftvqa~default~none~what-act-is-this --skip_existing

# config x2vlm-large1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-this -o suffix_output_dir=~what-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/x2vlm-large1b~ftvqa~default~none~what-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-happening-image -o suffix_output_dir=~what-is-happening-image~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/x2vlm-large1b~ftvqa~default~none~what-is-happening-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/activitynet~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-act-is-this -o suffix_output_dir=~what-act-is-this~followup --add_config ovqa/configs/followup/followup_activitynet.yaml -o run.followup_prev_dir=output/activitynet~val/x2vlm-large1b~ftvqa~default~none~what-act-is-this --skip_existing

