import argparse
from rdkit import Chem
import os

parser = argparse.ArgumentParser(description='Evaluation text2mol')

parser.add_argument('--prediction_file', type=str, help='Prediction file to evaluate.')
parser.add_argument('--label_file', type=str, help='Label file to evaluate.',
                    default="data/mol2text/test.txt")

args = parser.parse_args()
with open(args.prediction_file, "r", encoding="utf-8") as f:
    predictions = f.readlines()
    predictions = [line.strip() for line in predictions]

#  header CID	SMILES	description
with open(args.label_file, "r", encoding="utf-8") as f:
    dataset = [line.split("\t") for line in f.readlines()[1:]]
    
labels = [line[1] for line in dataset]

# Normalize the SMILES
correct = 0
incorrect = 0
parse_error = 0
total = 0

for pred, label in zip(predictions, labels):
    try:
        canon_pred = Chem.CanonSmiles(pred)
        canon_label = Chem.CanonSmiles(label)
        if canon_pred == canon_label:
            correct += 1
        else:
            incorrect += 1
    except:
        parse_error += 1
    total += 1

args.out_path = os.path.splitext(args.prediction_file)[0] + "_eval.txt"
    
with open(args.out_path, "w", encoding="utf-8") as f:
    f.write(",".join(["Accuracy", "Incorrect", "Parse Error"]) + "\n")
    f.write("{:.4f},{:.4f},{:.4f}\n".format(correct / total, incorrect / total, parse_error / total))
    # f.write("{:.4f}\n".format(correct / total))
    # f.write("{:.4f}\n".format(incorrect / total))
    # f.write("{:.4f}\n".format(parse_error / total))
    