import yaml
import argparse
import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


with open("./config.yaml") as f:
    threshold_dict = yaml.load(f, Loader=yaml.FullLoader)
    threshold_dict = threshold_dict["threshold"]


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tunib/electra-ko-base")
    parser.add_argument("--dir_path", type=str, default="/home/ckpt")
    parser.add_argument("--file_path", type=str, default="")
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--num_labels", type=int, default=7)
    parser.add_argument("--data_path", type=str, default="./data/test.csv")
    return parser


parser = _get_parser()
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, num_labels=args.num_labels
)

trained_model_dict = torch.load(f"{args.dir_path}/{args.file_path}")["state_dict"]

model_dict = dict()

for key in trained_model_dict:
    model_dict[key[6:]] = trained_model_dict[key]

model.load_state_dict(model_dict)
model.to(device)
model.eval()

df = pd.read_csv(args.data_path, encoding="utf-8", index_col=0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
comments = df["comment"]
comments = list(comments)
cols = threshold_dict.keys()
diction = {}

for col in cols:
    diction[col] = []
    diction[f"{col}_label"] = []

for idx, my_input in tqdm(enumerate(comments)):
    tokenized = tokenizer(
        my_input,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    ).to(device)
    output = model(**tokenized)

    logits = np.array(output.logits.cpu().detach()).squeeze()
    label = df.loc[idx][: args.num_labels].tolist()
    label = np.array(label)

    for i, col in enumerate(cols):
        diction[col].append(0 if threshold_dict[col] > logits[i] else 1)
        diction[f"{col}_label"].append(label[i])

f1_scores = []
precisions = []
recalls = []

for col in cols:
    f1 = f1_score(diction[col], diction[f"{col}_label"])
    prec = precision_score(diction[col], diction[f"{col}_label"])
    rec = recall_score(diction[col], diction[f"{col}_label"])
    f1_scores.append(f1)
    precisions.append(prec)
    recalls.append(rec)
    print(f"{col} f1: {f1}")
    print(f"{col} precision: {prec}")
    print(f"{col} recall: {rec}")

print(f"macro f1 score : {sum(f1_scores) / len(f1_scores)}")
print(f"macro precision score : {sum(precisions) / len(precisions)}")
print(f"macro recall score : {sum(recalls) / len(recalls)}")
