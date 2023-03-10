import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import torch

import time


def train_epoch(model, data_loader, optimizer, device, scheduler=None):
  """
  Trains the model for one epoch

  Parameters
  ----------
  model: ToxicClassifier
    The model to train
  data_loader: DataLoader
    The data loader to use
  optimizer: torch.optim.Adam
    The optimizer to use
  device: torch.device
    The device to use
  scheduler: torch.optim.lr_scheduler
    The scheduler to use

  Returns
  -------
  accuracy: float
    The accuracy of the model
  loss: float
    The loss of the model
  """
  model = model.train()
  losses = []

  start = time.time()

  for d in data_loader:
    input_ids = d['input_ids'].to(device)
    attention_mask = d['attention_mask'].to(device)
    labels = d['labels'].to(device)

    optimizer.zero_grad()
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs = torch.sigmoid(outputs)
    
    loss = torch.nn.BCELoss()(outputs, labels)
    losses.append(loss.item())
    loss.backward()
    
    optimizer.step()
    scheduler.step()
  
  end = time.time()
  print(f'Training time: {end - start}')

  return np.mean(losses), model




def eval_model(model, data_loader, device):
  """
  Evaluates the model

  Parameters
  ----------
  model: ToxicClassifier
    The model to evaluate
  data_loader: DataLoader
    The data loader to use
  device: torch.device
    The device to use

  Returns
  -------
  accuracy: float
    The accuracy of the model
  loss: float
    The loss of the model
  """
  model = model.eval()
  losses = []
  with torch.no_grad():
    for d in data_loader:
      input_ids = d['input_ids'].to(device)
      attention_mask = d['attention_mask'].to(device)
      labels = d['labels'].to(device)
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      outputs = torch.sigmoid(outputs)
      loss = torch.nn.BCELoss()(outputs, labels)
      losses.append(loss.item())
  return np.mean(losses)




def train(model, train_data_loader, validation_data_loader, optimizer, device, scheduler=None, EPOCHS=5):
  """
  Trains the model

  Parameters
  ----------
  model: ToxicClassifier
    The model to train
  train_data_loader: DataLoader
    The data loader to use for training
  validation_data_loader: DataLoader
    The data loader to use for validation
  optimizer: torch.optim.Adam
    The optimizer to use
  device: torch.device
    The device to use
  scheduler: torch.optim.lr_scheduler
    The scheduler to use
  EPOCHS: int
    The number of epochs to train for

  Returns
  -------
  history: dict
    The history of the training

  """
  history = defaultdict(list)
  best_loss = float('inf')

  for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_loss, model = train_epoch(
      model,
      train_data_loader,
      optimizer,
      device,
      scheduler
    )

    print(f'Train loss {train_loss}')

    val_loss = eval_model(
      model,
      validation_data_loader,
      device
    )

    print(f'Val   loss {val_loss}')
    print()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    if val_loss < best_loss:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_loss = val_loss


  plt.plot(history['train_loss'], label='train loss')
  plt.plot(history['val_loss'], label='validation loss')

  plt.title('Training history')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend()
  plt.savefig('train_history.png')
  plt.clf()

  return history




def get_predictions(model, data_loader, device):
  """
  Gets the predictions of the model

  Parameters
  ----------
  model: ToxicClassifier
    The model to use
  data_loader: DataLoader
    The data loader to use
  device: torch.device
    The device to use

  Returns
  -------
  predictions: torch.Tensor
    The predictions of the model
  real_values: torch.Tensor
    The real values of the model
  """
  model = model.eval()
  predictions = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d['comment_text']
      input_ids = d['input_ids'].to(device)
      attention_mask = d['attention_mask'].to(device)
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      outputs = torch.sigmoid(outputs)
      preds = torch.round(outputs)
      predictions.extend(preds)
      real_values.extend(d['labels'])
  predictions = torch.stack(predictions).cpu()
  real_values = torch.stack(real_values).cpu()
  return predictions, real_values


def train_validate(model, train_data_loader, validation_data_loader, optimizer, device, scheduler=None, EPOCHS=5):
  """
  Trains and validates the model

  Parameters
  ----------
  model: ToxicClassifier
    The model to train
  train_data_loader: DataLoader
    The data loader to use for training
  validation_data_loader: DataLoader
    The data loader to use for validation
  optimizer: torch.optim.Adam
    The optimizer to use
  device: torch.device
    The device to use
  scheduler: torch.optim.lr_scheduler
    The scheduler to use
  EPOCHS: int
    The number of epochs to train for

  Returns
  -------
  history: dict
    The history of the training
  predictions: torch.Tensor
    The predictions of the model
  real_values: torch.Tensor
    The real values of the model
  """
  criterion = torch.nn.BCELoss().to(device)

  for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    # Train the model
    model.train()
    losses = []

    start = time.time()

    for step, d in enumerate(train_data_loader):
      if step % 1000 == 0:
        elapsed = time.time() - start
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Loss: {:.5f}'.format(step, len(train_data_loader), elapsed, np.mean(losses)))
      input_ids = d['input_ids'].to(device)
      attention_mask = d['attention_mask'].to(device)
      labels = d['labels'].to(device)

      model.zero_grad()

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      outputs = torch.sigmoid(outputs)

      loss = criterion(outputs, labels)
      losses.append(loss.item())
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      optimizer.step()
      scheduler.step()

    print('=' * 10)
    print(' Epoch: {:>3,}  Train Loss: {:.5f}'.format(epoch, np.mean(losses)))

    # Validate the model
    model = model.eval()
    losses = []

    start = time.time()
    with torch.no_grad():
      for d in validation_data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.sigmoid(outputs)

        loss = torch.nn.BCELoss()(outputs, labels)
        losses.append(loss.item())
    
    print('  Validation Loss: {:.5f}'.format(np.mean(losses)))
    print('=' * 10)

