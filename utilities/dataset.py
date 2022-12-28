import torch
import pandas as pd
from transformers import BertTokenizerFast as BertTokenizer
import pytorch_lightning as pl


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



class ToxicCommentsDataset(torch.utils.data.Dataset):
  '''
  The toxic comments dataset

  Parameters
  ----------
  data: pd.DataFrame
    The dataframe containing the data
  tokenizer: transformers.BertTokenizer
    The tokenizer to use
  max_token_len: int
    The maximum length of the sequence

  Attributes
  ----------
  tokenizer: transformers.BertTokenizer
    The tokenizer to use
  data: pd.DataFrame
    The dataframe containing the data
  max_token_len: int
    The maximum length of the sequence

  Returns
  -------
  dict
    The dictionary containing the data
  '''

  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_token_len: int = 128
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.comment_text
    LABEL_COLUMNS = self.data.columns[2:]
    labels = data_row[LABEL_COLUMNS]

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )




class ToxicCommentDataModule(pl.LightningDataModule):
  '''
  The data module for the toxic comment dataset

  Parameters
  ----------
  train_data: pd.DataFrame
    The dataframe containing the training data
  test_data: pd.DataFrame
    The dataframe containing the test data
  tokenizer: transformers.BertTokenizer
    The tokenizer to use
  batch_size: int
    The batch size
  max_len: int
    The maximum length of the sequence

  Attributes
  ----------
  train_data: pd.DataFrame
    The dataframe containing the training data
  test_data: pd.DataFrame
    The dataframe containing the test data
  tokenizer: transformers.BertTokenizer
    The tokenizer to use
  batch_size: int
    The batch size
  max_len: int
    The maximum length of the sequence

  Methods
  -------
  setup(self, stage=None)
    Sets up the data
  train_dataloader(self)
    Returns the training dataloader
  val_dataloader(self)
    Returns the validation dataloader
  test_dataloader(self)
    Returns the test dataloader

  Returns
  -------
  dict
    The dictionary containing the data
  '''

  def __init__(self, train_data, test_data, tokenizer, batch_size, max_len):
    super().__init__()
    self.train_data = train_data
    self.test_data = test_data
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.max_len = max_len
  
  def setup(self, stage=None):
    self.train_dataset = ToxicCommentsDataset(
      self.train_data,
      self.tokenizer,
      self.max_len
    )
    self.test_dataset = ToxicCommentsDataset(
      self.test_data,
      self.tokenizer,
      self.max_len
    )

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=6
    )
  
  def val_dataloader(self):
    return torch.utils.data.DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=6
    )
  
  def test_dataloader(self):
    return torch.utils.data.DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=6
    )