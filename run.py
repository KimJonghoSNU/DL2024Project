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
    mol.UpdatePropertyCache(strict=False)
    # Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
    try:
        problems = Chem.DetectChemistryProblems(mol)
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
        self.llm = LLM(model = args.model_path, dtype = "bfloat16", tensor_parallel_size=1, disable_custom_all_reduce=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.sampling_params = SamplingParams(temperature=0, top_k=-1, max_tokens = args.max_new_decoding_tokens,
                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")], n=1)
        
        self.temperature = args.temperature
        self.top_k = args.top_k
        print("Model loaded.")
    def get_respond(self, messages):
        outputs = self.llm.chat(messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=True)
        responds = [output.outputs[0].text for output in outputs]
        print("Prompts: ", outputs[0].prompt)
        print("Responses: ", responds[0])
        
        return responds

def make_prompts(dataset, prompt_style = "cot", args = None, pipe = None):
    if prompt_style == "cot":
        prompts = []
        for i, data in enumerate(dataset):
            prompt = load_as_json("prompt/cot_1shot.txt")
            data[2] = data[2].replace("\n", " ")
            # prompt.extend([{"role": "user", "content": f"Now, convert the following description into its corresponding SMILES representation. Let's think step by step\nDescription: {data[2]}"}])
            prompt.extend([{"role": "user", "content": f"Description: {data[2]} Think step by step."}])
            prompts.append(prompt)
        outputs = pipe.get_respond(prompts)
        prompts = [p + [{"role": "assistant", "content": o}] for p, o in zip(prompts, outputs)]
        prompts = [p + [{"role": "user", "content": "Finalize your answer, only with the SMILES string, without any additional information."}] for p in prompts]
        outputs = pipe.get_respond(prompts)
        outputs = [normalize_final_output(o) for o in outputs]
        return outputs, prompts
    elif "refine" in prompt_style:
        prompts = []
        for i, data in enumerate(dataset): # data = [CID, SMILES, description]
            prompt = load_as_json("prompt/cot_1shot.txt")
            data[2] = data[2].replace("\n", " ")
            prompt.extend([{"role": "user", "content": f"Description: {data[2]}. Think step by step."}])
            prompts.append(prompt)
        outputs = pipe.get_respond(prompts)
        # TODO: parse SMILES from output
        prompts = [p + [{"role": "assistant", "content": o}] for p, o in zip(prompts, outputs)]
        # To extracted SMILES, temporaily generate the final prompt
        final_extractions = [p + [{"role": "user", "content": "Finalize your answer only with the SMILES string, without any additional information."}] for p in prompts]
        final_outputs = pipe.get_respond(final_extractions)
        final_outputs = [normalize_final_output(o) for o in final_outputs]
        # self-feedback
        self_feedback_prompt = {"role": "user", "content": "Double check that your answer is correct."}
        self_feedbacks = [self_feedback_prompt.copy() for _ in range(len(final_outputs))]
        # feedback from rdkit
        if "rdkit" in prompt_style:
            rdkit_feedbacks = [validate_smiles(o) for o in final_outputs]
            for i, o in enumerate(final_outputs):
                rdkit_feedback = rdkit_feedbacks[i]
                # if not "Valid SMILES format." in rdkit_feedback:
                self_feedbacks[i]["content"] = self_feedbacks[i]["content"] + "\nFeedback about SMILES validity: " + rdkit_feedback
                # prompts[i][-1]["content"] = prompts[i][-1]["content"] + f" The SMILES string you provided is not valid. {feedback}"
        if "daylight" in prompt_style:
            daylight_knowledge = open("data/daylight_filtered.txt", "r").read()
            # daylight is too long. make LLM to extract the relevant information.
            ##################
            prompts_for_daylight = prompts.copy()
            if "rdkit" in prompt_style: 
                prompts_for_daylight_toadd = []
                for i, o in enumerate(final_outputs):
                    rdkit_feedback = rdkit_feedbacks[i]
                    prompts_for_daylight_toadd.append(
                        [{"role": "user", 
                            "content": "Feedback about SMILES validity: " + rdkit_feedback + 
                            "\nExtract and summarize the relevant part to improve your SMILES answer from the following text: \n" + daylight_knowledge}])
                prompts_for_daylight = [p + p_add for p, p_add in zip(prompts_for_daylight, prompts_for_daylight_toadd)]
            else:
                prompts_for_daylight = [p + 
                                        [{"role": "user", 
                                          "content": "Extract and summarize the relevant part to validate your SMILES from the following text: \n" + 
                                          daylight_knowledge}] for p in prompts_for_daylight]
            daylight_outputs = pipe.get_respond(prompts_for_daylight)
            ##################
            for i, o in enumerate(daylight_outputs):
                if "rdkit" in prompt_style and "Valid SMILES format." in rdkit_feedbacks[i]:
                    continue # skip daylight feedback if SMILES is valid
                else:
                    self_feedbacks[i]["content"] = self_feedbacks[i]["content"] + "\nFeedback from daylight: " + o
        prompts = [p + [f] for p, f in zip(prompts, self_feedbacks)]
        outputs = pipe.get_respond(prompts)
        prompts = [p + [{"role": "assistant", "content": o}] for p, o in zip(prompts, outputs)]
        prompts = [p + [{"role": "user", "content": "Finalize your answer, only with the SMILES string, without any additional information."}] for p in prompts]
        outputs = pipe.get_respond(prompts)
        outputs = [normalize_final_output(o) for o in outputs]
        return outputs, prompts
    else:
        raise ValueError("Prompt style not recognized.")
    
parser = argparse.ArgumentParser(description='Multi debate')

parser.add_argument('--model_name', type=str, help='Name of GPT model used to test.')
parser.add_argument('--model_path', type=str, default = "", help='Path to GPT model.')
parser.add_argument('--output_path', type=str, help='Path for GPT model ouptuts.')
parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument('--top_k', type=int, default=None, help='Decoding top-k.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=512, help='Max new tokens for decoding output.')
parser.add_argument("--prompt_style", type=str, default="", help = "Prompting style for ICL.", choices = ["cot", "refine", "refine_rdkit", "refine_rdkit_daylight", "refine_daylight"])
parser.add_argument("--dataset", type=str, default="tracie", help = "Dataset to use.")

args = parser.parse_args()

pipe = Pipeline(args.model_name, args)
if "mol2text" in args.dataset:
    dataset_path = "data/mol2text/test.txt"
    # 1st line: header CID	SMILES	description
    with open(dataset_path, "r") as f:
        dataset = [line.split("\t") for line in f.readlines()[1:]]
    # For debug
    # dataset = dataset[:10]
elif "" in args.dataset:
    dataset = ()
    raise NotImplementedError
print(args)
print(f"Dataset size: {len(dataset)}")

output_path = args.output_path + f"/{args.prompt_style}/{args.model_name}/{time.strftime('%Y%m%d-%H%M%S')}.out"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
answer_list, histories = make_prompts(dataset, args.prompt_style, args = args, pipe = pipe)
# save histories
with open(output_path.replace(".out", "_histories.jsonl"), "w") as f:
    for history in histories:
        f.write(json.dumps(history[len(load_as_json("prompt/cot_1shot.txt")):]) + "\n")
with open(output_path, "w") as f:
    for answer in answer_list:
        f.write(answer.replace("\n", " ") + "\n")
