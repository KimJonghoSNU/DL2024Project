export CUDA_VISIBLE_DEVICES=6,7
# specify model name
model_path="/workspace/data03/pretrained_models/Meta-Llama-3.1-8B-Instruct"
model_name="Meta-Llama-3.1-8B-Instruct"

python evaluation.py \
    --output_path "output/cot/Meta-Llama-3.1-8B-Instruct/20241123-014706.out" 

python evaluation.py \
    --output_path "output/refine_rdkit/Meta-Llama-3.1-8B-Instruct/20241123-024817.out" 

python evaluation.py \
    --output_path "output/refine_daylight/Meta-Llama-3.1-8B-Instruct/20241122-231229.out" 

python evaluation.py \
    --output_path "output/refine/Meta-Llama-3.1-8B-Instruct/20241123-020518.out" 

python evaluation.py \
    --output_path "output/refine_rdkit_daylight/Meta-Llama-3.1-8B-Instruct/20241122-203857.out" 


# python extract_final_answer.py \
#     --model_path $model_path \
#     --model_name $model_name \
#     --output_path "output/refine_rdkit/Meta-Llama-3.1-8B-Instruct/20241123-024817_histories.jsonl" \
#     --max_new_decoding_tokens 512 \
#     --temperature 0 \
#     --top_k 1 \
#     --dataset "mol2text"

# python extract_final_answer.py \
#     --model_path $model_path \
#     --model_name $model_name \
#     --output_path "output/refine_daylight/Meta-Llama-3.1-8B-Instruct/20241122-231229_histories.jsonl" \
#     --max_new_decoding_tokens 512 \
#     --temperature 0 \
#     --top_k 1 \
#     --dataset "mol2text"
    
# python extract_final_answer.py \
#     --model_path $model_path \
#     --model_name $model_name \
#     --output_path "output/refine/Meta-Llama-3.1-8B-Instruct/20241123-020518_histories.jsonl" \
#     --max_new_decoding_tokens 512 \
#     --temperature 0 \
#     --top_k 1 \
#     --dataset "mol2text"

# python extract_final_answer.py \
#     --model_path $model_path \
#     --model_name $model_name \
#     --output_path "output/refine_rdkit_daylight/Meta-Llama-3.1-8B-Instruct/20241122-203857_histories.jsonl" \
#     --max_new_decoding_tokens 512 \
#     --temperature 0 \
#     --top_k 1 \
#     --dataset "mol2text"
    