from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd


class StereotypeDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_dir: str, stage: str):
        super(StereotypeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.stage = stage
        self.num_labels = 7
        self.comments, self.labels = self.load_dataset(data_dir=data_dir, stage=stage)

    def load_dataset(self, data_dir, stage):
        dataframe = pd.read_csv(f"{data_dir}/{stage}.csv")
        comments = list(dataframe["comment"])
        labels = list()
        max_length = max([len(self.tokenizer.encode(comment)) for comment in comments])

        for idx, comment in enumerate(comments):
            label = list(dataframe.iloc[idx][:self.num_labels])

            comments[idx] = self.tokenizer(
                comment,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            labels.append(label)

        return comments, labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        return self.comments[index], self.labels[index]