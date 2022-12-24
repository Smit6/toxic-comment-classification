from transformers import BertModel

import torch


class ToxicClassifier(torch.nn.Module):
  """
  The toxic classifier

  Parameters
  ----------
  n_classes: int
    The number of classes
  bert: BertForSequenceClassification
    The bert model
  """
  def __init__(self, n_classes):
    super(ToxicClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    return output
