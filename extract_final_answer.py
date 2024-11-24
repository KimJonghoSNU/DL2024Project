from rdkit import Chem
# import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import time
from vllm import LLM, SamplingParams
import argparse
import os
import json

def load_as_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()
    data = data.split("```")
    data = [d.strip() for d in data]
    # 1st data = system, then user, assistant, user, assistant, ...
    json_data = []
    json_data.append({"role": "system", "content": data[0]})
    for i in range(1, len(data), 2):
        json_data.append({"role": "user", "content": data[i]})
        json_data.append({"role": "assistant", "content": data[i+1]})
    return json_data

def validate_smiles(smiles):
    """
    Validates a SMILES string and provides error feedback.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # mol.UpdatePropertyCache(strict=False)
    # Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
    try:
        problems = Chem.DetectChemistryProblems(mol)
        # https://stackoverflow.com/questions/75683009/how-to-catch-the-error-message-from-chem-molfromsmilesformula
    except:
        return "Your SMILES string is not valid. The parsing process failed."
    if len(problems) > 0:
        return "Your SMILES string failed sanity checks. " + " ".join([problem.Message() for problem in problems])
    else:
        return "Your SMILES string is valid."

def normalize_final_output(smiles):
    if ":" in smiles:
        smiles = smiles.split(":")[1]
    smiles = smiles.strip()
    return smiles


class Pipeline:
    def __init__(self,model_id,args):
        self.api = False
        self.local = False
        self.model_id = model_id
        self.args = args
        self.llm = LLM(model = args.model_path, dtype = "bfloat16", tensor_parallel_size=2, disable_custom_all_reduce=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.sampling_params = SamplingParams(temperature=0, top_k=-1, max_tokens = args.max_new_decoding_tokens,
                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")], n=1)
        
        self.temperature = args.temperature
        self.top_k = args.top_k
        print("Model loaded.")
    def get_respond(self, messages, print_output = False):
        outputs = self.llm.chat(messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=True)
        responds = [output.outputs[0].text for output in outputs]
        if print_output:
            for i in range(len(outputs)):
                print("Prompts: ", outputs[i].prompt)
                print("Responses: ", responds[i])
        # print("Prompts: ", outputs[0].prompt)
        # print("Responses: ", responds[0])
        
        return responds

def make_prompts(dataset, prompt_style = "cot", args = None, pipe = None):
    prompts = []
    for i, data in enumerate(dataset):
        # prompt = load_as_json("prompt/cot_1shot.txt")
        prompt = load_as_json("prompt/cot_1shot.txt")
        prompt.extend(data)
        prompts.append(prompt)
    outputs = pipe.get_respond(prompts)
    outputs = [normalize_final_output(o) for o in outputs]
    return outputs, prompts
    
parser = argparse.ArgumentParser(description='Multi debate')

parser.add_argument('--model_name', type=str, help='Name of GPT model used to test.')
parser.add_argument('--model_path', type=str, default = "", help='Path to GPT model.')
parser.add_argument('--output_path', type=str, help='Path for GPT model ouptuts.')
parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument('--top_k', type=int, default=None, help='Decoding top-k.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=512, help='Max new tokens for decoding output.')
parser.add_argument("--dataset", type=str, default="tracie", help = "Dataset to use.")

args = parser.parse_args()

# jsonl
dataset = [json.loads(line) for line in open(args.dataset, "r", encoding="utf-8")]

print(args)
print(f"Dataset size: {len(dataset)}")

pipe = Pipeline(args.model_name, args)
answer_list, prompts = make_prompts(dataset, args = args, pipe = pipe)
with open(args.output_path.replace("_histories.jsonl", ".out"), "w", encoding="utf-8") as f:
    for answer in answer_list:
        f.write(answer.replace("\n", " ") + "\n")
