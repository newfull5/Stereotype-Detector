from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd


class StereotypeDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_dir: str, stage: str, num_labels: int):
        super(StereotypeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.stage = stage
        self.num_labels = num_labels
        self.comments, self.labels = self.load_dataset(data_dir=data_dir, stage=stage)

    def load_dataset(self, data_dir, stage):
        dataframe = pd.read_csv(f"{data_dir}/{stage}.csv", index_col=0, encoding='utf-8')
        comments = list(dataframe["comment"])
        labels = list()
        max_length = 512

        for idx, comment in enumerate(comments):
            label = list(dataframe.iloc[idx][:self.num_labels])

            comments[idx] = self.tokenizer(
                comment,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            labels.append(torch.tensor(label).float())
        return comments, labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        return self.comments[index], self.labels[index]