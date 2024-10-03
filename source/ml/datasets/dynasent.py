import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class DynaSent(Dataset):
    def __init__(self, data_round: int, data_split: str, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]):
        super().__init__()
        self.name = f'dynabench.dynasent.r{data_round}.all'
        self.tokenizer = tokenizer
        self.stoi = {'negative': 0, 'neutral': 1, 'positive': 2}

        data_dict = load_dataset(path='dynabench/dynasent', name=self.name, trust_remote_code=True)
        data = data_dict[data_split]
        encodings = self.tokenizer.batch_encode_plus(
            data['sentence'],
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True
        )

        self.labels = data['gold_label']
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        label = self.stoi[self.labels[idx]]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
