
from torch.utils.data import Dataset

class QueryPassageDataset(Dataset):
    def __init__(self, query_passage_dict_list, tokenizer, max_len=256):
        self.query_passage_dict_list = query_passage_dict_list
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.query_passage_dict_list)
    
    def __getitem__(self, index):

        query_passage_pair = self.query_passage_dict_list[index]

        query = query_passage_pair["query"]
        passage = query_passage_pair["passage"]

        q = self.tokenizer(query, truncation=True, padding='max_length',
                           max_length=self.max_len, return_tensors='pt')
        p = self.tokenizer(passage, truncation=True, padding='max_length',
                           max_length=self.max_len, return_tensors='pt')

        return {
            'query_input_ids': q['input_ids'].squeeze(0),
            'query_attention_mask': q['attention_mask'].squeeze(0),
            'doc_input_ids': p['input_ids'].squeeze(0),
            'doc_attention_mask': p['attention_mask'].squeeze(0)
        }