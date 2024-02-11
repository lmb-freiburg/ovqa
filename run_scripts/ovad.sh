#CONFIG_DIR="ovqa/configs/projects/ovad_attributes~val"
#FILES=$(ls ${CONFIG_DIR}/*.yaml) | sort
#
#for CONFIG_NAME in \
#        "blip1~ftvqa~default~none" \
#        "blip2-opt2.7b~pt~default~qa-short" \
#        "blip2-t5xl~pt~default~qa-short" \
#        "iblip-t5xl~pt~norep-b1~qa-short" \
#        "iblip-vic7b~pt~norep-b1~qa-short" \
#        "llava7b~default~llava-nosample~conv-multimodal" \
#        "x2vlm-base1b~ftvqa~default~none" \
#        "x2vlm-large1b~ftvqa~default~none" \
#        ; do
#    echo ""
#    echo "# config ${CONFIG_NAME}"
#
#for PROMPT_NAME in "new_first_question_type" "new_second_question_type" "new_third_question_type"; do
#
#    SUFFIX="~${PROMPT_NAME}"
#    echo python newevaluate.py -c "${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
#        -d prompt_type="${PROMPT_NAME}" \
#        -o suffix_output_dir="${SUFFIX}" \
#        --skip_existing $*
#
#done
#done
#echo
#exit

# config blip1~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip1~ftvqa~default~none.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip1~ftvqa~default~none.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip1~ftvqa~default~none.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config blip2-opt2.7b~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip2-opt2.7b~pt~default~qa-short.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip2-opt2.7b~pt~default~qa-short.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip2-opt2.7b~pt~default~qa-short.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config blip2-t5xl~pt~default~qa-short
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip2-t5xl~pt~default~qa-short.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip2-t5xl~pt~default~qa-short.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/blip2-t5xl~pt~default~qa-short.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config iblip-t5xl~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/iblip-t5xl~pt~norep-b1~qa-short.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config iblip-vic7b~pt~norep-b1~qa-short
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/iblip-vic7b~pt~norep-b1~qa-short.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config llava7b~default~llava-nosample~conv-multimodal
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/llava7b~default~llava-nosample~conv-multimodal.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config x2vlm-base1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/x2vlm-base1b~ftvqa~default~none.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/x2vlm-base1b~ftvqa~default~none.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/x2vlm-base1b~ftvqa~default~none.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing

# config x2vlm-large1b~ftvqa~default~none
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/x2vlm-large1b~ftvqa~default~none.yaml -d prompt_type=new_first_question_type -o suffix_output_dir=~new_first_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/x2vlm-large1b~ftvqa~default~none.yaml -d prompt_type=new_second_question_type -o suffix_output_dir=~new_second_question_type --skip_existing
python newevaluate.py -c ovqa/configs/projects/ovad_attributes~val/x2vlm-large1b~ftvqa~default~none.yaml -d prompt_type=new_third_question_type -o suffix_output_dir=~new_third_question_type --skip_existing
