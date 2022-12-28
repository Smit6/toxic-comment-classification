import pandas as pd

import os

from transformers import BertTokenizerFast as BertTokenizer, get_linear_schedule_with_warmup, AdamW

import pytorch_lightning as pl

# from pytorch_lightning.metrics.functional import auroc, accuracy
from torchmetrics import AUROC, Accuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader


from utilities.dataset import ToxicDataset, TestDataset, ToxicCommentDataModule
from utilities.model import ToxicClassifier
from utilities.train_eval import train, get_predictions, train_validate
from utilities.train_eval_lightning import ToxicCommentTagger


def load_df(filename):
  '''
  Loads the data from the filename

  Parameters
  ----------
  filename: str
    The filename to load

  Returns
  -------
  df: pd.DataFrame
    The dataframe containing the data
  '''
  df = pd.read_csv(os.path.join('data', filename))
  return df


def get_trainable_params(model, train_dataset, validation_dataset):
  '''
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
  '''
  EPOCHS = 5
  BATCH_SIZE = 8
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


def get_data_module(train_dataset, validation_dataset, tokenizer, batch_size, max_len):
  '''
  Returns the data module

  Parameters
  ----------
  train_dataset: ToxicDataset
    The training dataset
  validation_dataset: ToxicDataset
    The validation dataset
  device: torch.device
    The device to use
  BATCH_SIZE: int
    The batch size to use

  Returns
  -------
  data_module: ToxicCommentDataModule
    The data module
  '''
  
  data_module = ToxicCommentDataModule(
    train_dataset,
    validation_dataset,
    tokenizer,
    batch_size=batch_size,
    max_len=max_len
  )
  
  return data_module

def get_trainer(gpus=1, max_epochs=5, patience=2):
  '''
  Returns the trainer

  Parameters
  ----------
  gpus: int
    The number of gpus to use
  max_epochs: int
    The number of epochs to train for
  patience: int
    The number of epochs to wait before stopping

  Returns
  -------
  trainer: pl.Trainer
  '''
  
  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min',
    verbose=True
  )

  logger = pl.loggers.TensorBoardLogger('lightning_logs', name='toxic_comments')

  early_stopping_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience
  )

  trainer = pl.Trainer(
    gpus=gpus,
    max_epochs=max_epochs,
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback]
  )

  return trainer



if __name__ == '__main__':
  '''
  The main function
  '''
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

  
  ################################
  # Train with pytorch lightning #
  ################################

  # Create the data module
  data_module = get_data_module(train_df, validation_df, tokenizer, 8, 512)

  # Create the trainer
  trainer = get_trainer(gpus=1, max_epochs=5, patience=2)

  steps_per_epoch = len(train_df) // 8
  n_train_steps = steps_per_epoch * 5
  n_warmup_steps = n_train_steps // 5

  # Create the model
  model = ToxicCommentTagger(
    n_classes=6,
    n_warmup_steps=n_warmup_steps,
    n_training_steps=n_train_steps
  )

  # Train the model
  trainer.fit(model, data_module)


  # Load the model
  trained_model  = ToxicCommentTagger.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    num_classes=6,
  )

  # Evaluate and freeze the model for predictions
  trained_model.eval()
  trained_model.freeze()

  # Move the model to the device
  trained_model = trained_model.to(device)

  predictions, real_values = get_predictions(
    model=trained_model,
    data_loader=data_module.validation_dataloader(),
    device=device
  )

  # Get the accuracy
  accuracy = Accuracy(task='multilabel', threshold=0.5, num_labels=6)
  print('Accuracy: ', accuracy(real_values, predictions))

  # ROC AUC
  aucroc = AUROC(task='multilabel', num_labels=6)
  for i, name in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    print(f'{name} AUC: {aucroc(predictions[:, i], real_values[:, i])}')

  # Classification report
  print(classification_report(real_values, predictions, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))






  # Train with pytorch

  ##################################
  # TO DO: Fix the lower precision #
  ##################################

  # Create the model
  # model = ToxicClassifier(6)
  # model = model.to(torch.float8)
  # model = model.to(device)

  # Get the trainable parameters
  # optimizer, scheduler, train_data_loader, validation_data_loader, EPOCHS = get_trainable_params(model, train_dataset, validation_dataset)

  # Train the model
  # train(model, train_data_loader, validation_data_loader, optimizer, device, scheduler, EPOCHS)
  # train_validate(model, train_data_loader, validation_data_loader, optimizer, device, scheduler, EPOCHS)  

  # # Load the data for testing
  # test_df = load_df('labeled_test.csv')
  # # Create the test dataset
  # test_dataset = ToxicDataset(validation_df, tokenizer, 512)
  # # Create the test data loader
  # test_data_loader = DataLoader(test_dataset, batch_size=8, num_workers=6)

  # # Get the predictions
  # model = ToxicClassifier(6).to(device)
  # model.load_state_dict(torch.load('best_model_state.bin'))
  # y_pred, y_test = get_predictions(model, test_data_loader, device)

  # # Get Accuracy
  # accuracy = Accuracy(task='multilabel', threshold=0.5, num_labels=6)
  # print('Accuracy: ', accuracy(y_test, y_pred))

  # np.savetxt('y_pred.csv', y_pred, delimiter=',')
  
