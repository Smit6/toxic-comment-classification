import torch


class ToxicDataset(torch.utils.data.Dataset):
  """
  The toxic dataset

  Parameters
  ----------
  data: pd.DataFrame
    The dataframe containing the data
  tokenizer: transformers.BertTokenizer
    The tokenizer to use
  max_len: int
    The maximum length of the sequence

  Returns
  -------
  dict
    The dictionary containing the data
  """
  def __init__(self, data, tokenizer, max_len):
    self.data = data
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, item):
    data = self.data.iloc[item]
    comment_text = data['comment_text']
    labels = data[self.data.columns[2:]]
    encoding = self.tokenizer.encode_plus(
      comment_text,
      max_length=self.max_len,
      add_special_tokens=True,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return dict(
      comment_text=comment_text,
      input_ids=encoding['input_ids'].flatten(),
      attention_mask=encoding['attention_mask'].flatten(),
      # labels=torch.FloatTensor(labels).half()
      labels=torch.tensor(labels, dtype=torch.float16)
    )


class TestDataset:
  """
  The test dataset

  Parameters
  ----------
  data: pd.DataFrame
    The dataframe containing the data
  tokenizer: transformers.BertTokenizer
    The tokenizer to use
  max_len: int
    The maximum length of the sequence

  Returns
  -------
  dict
    The dictionary containing the data
  """
  def __init__(self, data, tokenizer, max_len):
    self.data = data
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    data = self.data.iloc[item]
    comment_text = data['comment_text']
    encoding = self.tokenizer.encode_plus(
      comment_text,
      max_length=self.max_len,
      add_special_tokens=True,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return dict(
      comment_text=comment_text,
      input_ids=encoding['input_ids'].flatten(),
      attention_mask=encoding['attention_mask'].flatten(),
    )