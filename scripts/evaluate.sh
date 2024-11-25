export CUDA_VISIBLE_DEVICES=6,7
# specify model name
model_path="/workspace/data03/pretrained_models/Meta-Llama-3.1-8B-Instruct"
model_name="Meta-Llama-3.1-8B-Instruct"

python evaluation.py \
    --prediction_file "output/cot/Meta-Llama-3.1-8B-Instruct/20241123-014706.out" 

python evaluation.py \
    --prediction_file "output/refine_rdkit/Meta-Llama-3.1-8B-Instruct/20241123-024817.out" 

python evaluation.py \
    --prediction_file "output/refine_daylight/Meta-Llama-3.1-8B-Instruct/20241122-231229.out" 

python evaluation.py \
    --prediction_file "output/refine/Meta-Llama-3.1-8B-Instruct/20241123-020518.out" 

python evaluation.py \
    --prediction_file "output/refine_rdkit_daylight/Meta-Llama-3.1-8B-Instruct/20241122-203857.out" 

