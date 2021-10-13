import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import metrics
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tunib/electra-ko-base")
    parser.add_argument("--dir_path", type=str, default="/home/ckpt")
    parser.add_argument("--file_path", type=str, default="")
    parser.add_argument("--num_labels", type=int, default=7)
    return parser


parser = _get_parser()
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

df = pd.read_csv('./data/test.csv', encoding='utf-8', index_col=0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

comments = df['comment']
comments = list(comments)
cols = ['stereotype','anti-stereotype','unrelated','profession','race','gender','religion']
diction = {}

for col in cols:
    diction[col] = []
    diction[f"{col}_label"] = []

for idx, my_input in tqdm(enumerate(comments)):
    tokenized = tokenizer(my_input, return_tensors="pt", max_length=512, padding="max_length", truncation=True).to(device)
    output = model(**tokenized)

    logits = np.array(output.logits.cpu().detach()).squeeze()
    label = df.loc[idx][:args.num_labels].tolist()
    label = np.array(label)

    for i, col in enumerate(cols):
        diction[col].append(logits[i])
        diction[f"{col}_label"].append(label[i])


for col in cols:
    pre, rec, threshold = metrics.precision_recall_curve(np.array(diction[f'{col}_label']), np.array(diction[f'{col}']))
    temp = []

    for i in range(min(len(threshold), len(rec), len(pre))):
        temp.append([threshold[i], (2*rec[i]*pre[i])/(rec[i] + pre[i])])

    t1, s1 = sorted(temp, key=lambda x: x[-1])[-1]

    print(f"{col}: {t1}")