import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from transformers import BertTokenizerFast as BertTokenizer, get_linear_schedule_with_warmup, AdamW

# from pytorch_lightning.metrics.functional import auroc, accuracy
from torchmetrics import AUROC, Accuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader


from utilities.dataset import ToxicDataset, TestDataset
from utilities.model import ToxicClassifier
from utilities.train_eval import train, get_predictions


def load_df(filename):
  """
  Loads the data from the filename

  Parameters
  ----------
  filename: str
    The filename to load

  Returns
  -------
  df: pd.DataFrame
    The dataframe containing the data
  """
  df = pd.read_csv(os.path.join('data', filename))
  return df


def get_trainable_params(model, train_dataset, validation_dataset):
  """
  Returns the trainable parameters for the model

  Parameters
  ----------
  model: ToxicClassifier
    The model to train
  train_dataset: ToxicDataset
    The training dataset
  validation_dataset: ToxicDataset
    The validation dataset

  Returns
  -------
  optimizer: torch.optim.Adam
    The optimizer to use
  scheduler: torch.optim.lr_scheduler
    The scheduler to use
  train_data_loader: DataLoader
    The training data loader
  validation_data_loader: DataLoader
    The validation data loader
  EPOCHS: int
    The number of epochs to train for
  """
  EPOCHS = 5
  BATCH_SIZE = 16
  train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
  validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=6)

  optimizer = AdamW(params=model.parameters(), lr=2e-5)
  total_steps = len(train_data_loader) * EPOCHS

  print('total_steps:', total_steps)
  print('warmup_steps:', total_steps // 5)
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 5,
    num_training_steps=total_steps
  )

  return optimizer, scheduler, train_data_loader, validation_data_loader, EPOCHS



if __name__ == '__main__':
  """
  The main function
  """
  # Gather the data
  df = load_df('train.csv')

  # Tokenize the comments
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  encoding = tokenizer.encode_plus(
    df['comment_text'][0],
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
  )
  encoding.keys()

  # Split the data into train and validation sets
  train_df, validation_df = train_test_split(df, test_size=0.05, random_state=42)
  print('train_df.shape:', train_df.shape)
  print('validation_df.shape:', validation_df.shape)

  # Create the datasets
  train_dataset = ToxicDataset(train_df, tokenizer, 512)
  validation_dataset = ToxicDataset(validation_df, tokenizer, 512)

  # Select a device
  torch.cuda.empty_cache()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Device:', device)

  # Create the model
  model = ToxicClassifier(6)
  model = model.to(torch.float16)
  model = model.to(device)

  # Get the trainable parameters
  optimizer, scheduler, train_data_loader, validation_data_loader, EPOCHS = get_trainable_params(model, train_dataset, validation_dataset)

  # Train the model
  train(model, train_data_loader, validation_data_loader, optimizer, device, scheduler, EPOCHS)

  # Load the data for testing
  test_df = load_df('labeled_test.csv')
  # Create the test dataset
  test_dataset = ToxicDataset(validation_df, tokenizer, 512)
  # Create the test data loader
  test_data_loader = DataLoader(test_dataset, batch_size=16, num_workers=6)

  # Get the predictions
  model = ToxicClassifier(6).to(device)
  model.load_state_dict(torch.load('best_model_state.bin'))
  y_pred, y_test = get_predictions(model, test_data_loader, device)

  # Get Accuracy
  accuracy = Accuracy(task='multilabel', threshold=0.5, num_labels=6)
  print('Accuracy: ', accuracy(y_test, y_pred))

  np.savetxt('y_pred.csv', y_pred, delimiter=',')
  
