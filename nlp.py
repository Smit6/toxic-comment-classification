import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from transformers import BertTokenizerFast as BertTokenizer, get_linear_schedule_with_warmup, AdamW

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
  EPOCHS = 10
  BATCH_SIZE = 16
  train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=6)
  validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=6)

  optimizer = AdamW(params=model.parameters(), lr=2e-5)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
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
    max_length=256,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
  )
  encoding.keys()

  # Split the data into train and validation sets
  train_df, validation_df = train_test_split(df, test_size=0.1, random_state=42)
  print(train_df.shape, validation_df.shape)

  # Create the datasets
  train_dataset = ToxicDataset(train_df, tokenizer, 256)
  validation_dataset = ToxicDataset(validation_df, tokenizer, 256)

  # Select a device
  torch.cuda.empty_cache()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')
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
  test_df = pd.read_csv('test.csv')
  # Create the test dataset
  test_dataset = TestDataset(test_df, tokenizer, 256)
  # Create the test data loader
  test_data_loader = DataLoader(test_dataset, batch_size=16, num_workers=6)

  # Get the predictions
  model = torch.load('best_model_state.bin')
  y_pred, y_test = get_predictions(model, test_data_loader, device)
  y_pred = y_pred.numpy()


  # Get Accuracy
  print('Accuracy: ', accuracy_score(y_test, y_pred.round()))


  # Get the ROC AUC score
  auc = roc_auc_score(y_test, y_pred.round())
  print('ROC AUC score: ', auc)
  # Plot the ROC AUC curve
  fpr, tpr, _ = roc_curve(y_test, y_pred)
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('AUC-ROC curve')
  plt.legend(loc='lower right')
  plt.show()
  plt.clf()

  # Get the classification report
  print(classification_report(y_test, y_pred.round(), target_names=df.columns[2:]))

  # Plot the confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=df.columns[2:], yticklabels=df.columns[2:])
  plt.show()

