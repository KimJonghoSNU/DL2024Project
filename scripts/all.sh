export CUDA_VISIBLE_DEVICES=6,7
# specify model name
model_path="/workspace/data03/pretrained_models/Meta-Llama-3-8B-Instruct"
model_name="Meta-Llama-3-8B-Instruct"

PROMPT_STYLES=("cot" "refine" "refine_rdkit" "refine_rdkit_daylight" "refine_daylight")
# ["cot", "refine", "refine_rdkit", "refine_rdkit_daylight", "refine_daylight"]
for prompt_style in ${PROMPT_STYLES[@]}; do
    python run.py \
        --model_path $model_path \
        --model_name $model_name \
        --output_path "output/" \
        --max_new_decoding_tokens 512 \
        --prompt_style $prompt_style \
        --temperature 0 \
        --top_k 1 \
        --dataset "mol2text"
done