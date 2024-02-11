# setup $OVQA_OUTPUT_DIR
source <(python -m ovqa.bash_paths)
if [ -z "$OVQA_OUTPUT_DIR" ]; then
    echo "OVQA_OUTPUT_DIR not set, output directory will not be found"
    exit 1
fi

for DATASET in "imagenet1k-square~val" "imagenet1k~val"; do
    echo ""
    echo "# ---------- dataset ${DATASET}"

for CONFIG_NAME in \
        "blip1~ftvqa~default~none" \
        "blip2-opt2.7b~pt~default~qa-short" \
        "blip2-opt2.7b~pt~vqa~qa-short" \
        "blip2-t5xl~pt~default~qa-short" \
        "blip2-t5xl~pt~vqa~qa-short" \
        "iblip-t5xl~pt~norep-b1~qa-short" \
        "iblip-vic7b~pt~norep-b1~qa-short" \
        "llava7b~default~llava-nosample~conv-multimodal" \
        "x2vlm-base1b~ftvqa~default~none" \
        "x2vlm-large1b~ftvqa~default~none" \
        ; do
    echo ""
    echo "# config ${CONFIG_NAME}"

for PROMPT_NAME in "what-seen-image" "what-is-in-image" "whats-this"; do
    CONFIG_DIR="ovqa/configs/projects/${DATASET}"

    SUFFIX="~${PROMPT_NAME}"
    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
        -d question_type="${PROMPT_NAME}" \
        -o suffix_output_dir="${SUFFIX}" \
        --skip_existing $*

    F_DIR="${OVQA_OUTPUT_DIR}/${DATASET}/${CONFIG_NAME}${SUFFIX}"
    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
        -d question_type="${PROMPT_NAME}" \
        -o suffix_output_dir="${SUFFIX}~followup" \
        --add_config ovqa/configs/followup/followup_imagenet.yaml \
        -o run.followup_prev_dir="${F_DIR}" \
        --skip_existing $*

done
done
done
echo
exit

# ---------- dataset imagenet1k-square~val

# config blip1~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip1~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip1~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip1~ftvqa~default~none~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip1~ftvqa~default~none~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip1~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip1~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip1~ftvqa~default~none~whats-this --skip_existing

# config blip2-opt2.7b~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-opt2.7b~pt~default~qa-short~whats-this --skip_existing

# config blip2-opt2.7b~pt~vqa~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-opt2.7b~pt~vqa~qa-short~whats-this --skip_existing

# config blip2-t5xl~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-t5xl~pt~default~qa-short~whats-this --skip_existing

# config blip2-t5xl~pt~vqa~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short~whats-this --skip_existing

# config iblip-t5xl~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/iblip-t5xl~pt~norep-b1~qa-short~whats-this --skip_existing

# config iblip-vic7b~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/iblip-vic7b~pt~norep-b1~qa-short~whats-this --skip_existing

# config llava7b~default~llava-nosample~conv-multimodal
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/llava7b~default~llava-nosample~conv-multimodal~whats-this --skip_existing

# config x2vlm-base1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/x2vlm-base1b~ftvqa~default~none~whats-this --skip_existing

# config x2vlm-large1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k-square~val/x2vlm-large1b~ftvqa~default~none~whats-this --skip_existing

# ---------- dataset imagenet1k~val

# config blip1~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip1~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip1~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip1~ftvqa~default~none~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip1~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip1~ftvqa~default~none~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip1~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip1~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip1~ftvqa~default~none~whats-this --skip_existing

# config blip2-opt2.7b~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-opt2.7b~pt~default~qa-short~whats-this --skip_existing

# config blip2-opt2.7b~pt~vqa~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-opt2.7b~pt~vqa~qa-short~whats-this --skip_existing

# config blip2-t5xl~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-t5xl~pt~default~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-t5xl~pt~default~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~default~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-t5xl~pt~default~qa-short~whats-this --skip_existing

# config blip2-t5xl~pt~vqa~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/blip2-t5xl~pt~vqa~qa-short~whats-this --skip_existing

# config iblip-t5xl~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/iblip-t5xl~pt~norep-b1~qa-short~whats-this --skip_existing

# config iblip-vic7b~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/iblip-vic7b~pt~norep-b1~qa-short~whats-this --skip_existing

# config llava7b~default~llava-nosample~conv-multimodal
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/llava7b~default~llava-nosample~conv-multimodal~whats-this --skip_existing

# config x2vlm-base1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/x2vlm-base1b~ftvqa~default~none~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/x2vlm-base1b~ftvqa~default~none~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-base1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/x2vlm-base1b~ftvqa~default~none~whats-this --skip_existing

# config x2vlm-large1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-seen-image -o suffix_output_dir=~what-seen-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/x2vlm-large1b~ftvqa~default~none~what-seen-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=what-is-in-image -o suffix_output_dir=~what-is-in-image~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/x2vlm-large1b~ftvqa~default~none~what-is-in-image --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this --skip_existing
python newevaluate.py -c ovqa/configs/projects/imagenet1k~val/x2vlm-large1b~ftvqa~default~none.yaml -d question_type=whats-this -o suffix_output_dir=~whats-this~followup --add_config ovqa/configs/followup/followup_imagenet.yaml -o run.followup_prev_dir=output/imagenet1k~val/x2vlm-large1b~ftvqa~default~none~whats-this --skip_existing

