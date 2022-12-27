# toxic-comment-classification
 
### Objective
Natural Language Processing system to predict the probability of the toxicity level of a comment.

### Dataset: [Toxic Comment](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

Wikipidia has provided this dataset through a Kaggle competition to explore advanced techniques to identify comments which can be considered toxic behavior. We must create a model which predicts a probability of each type of toxicity for each comment.


### Steps taken for building an optimal NLP system
1. Exploratory data analysis (EDA) - eda.ipynb
2. Embedding with BertTokenizer from Hugging-face
3. Dataloader for the datasets
4. Build multi-label classifier leveraging the BertModel from the Hugging-face
5. Train model
6. Test model
7. Measure the performance


#### 1. Exploratory data analysis (EDA)
- Gather
- Asses
- Clean
- Explore

#### 2. Embedding with BertTokenizer from Hugging-face
- Apply BertTokenizer to generate features from the comments' text.

#### 3. Dataloader for the datasets
- Prepare dataloaders for train, validation and test datasets.
- Batch size - 16

#### 4. Build multi-label classifier leveraging the BertModel from the Hugging-face

MODEL:
  BERT Model
  Liner(bert hidden size, n_classes=6)

#### 5. Train Model
- Optimizer - AdamW
- Scheduler - Linear scheduler with warmup to avoid nan issue
- Epochs - 10

#### 6. Test Model
- Evaluate the model on the test dataset and get the predicted labels

#### 7. Measure the performance
- Accuracy - 98.13%

- Classification Report

                      precision    recall  f1-score   support

        toxic              0.68      0.91      0.78       748
        severe_toxic       0.53      0.30      0.38        80
        obscene            0.79      0.87      0.83       421
        threat             0.23      0.38      0.29        13
        insult             0.79      0.70      0.74       410
        identity_hate      0.59      0.62      0.60        71

        micro avg          0.72      0.81      0.76      1743
        macro avg          0.60      0.63      0.60      1743
        weighted avg       0.72      0.81      0.75      1743
        samples avg        0.08      0.08      0.08      1743

The combination of accuracy and classification report gives us complete picture of the performance of the model.

### Conclusion
This natural language processing engine powered by transfer learned Bert model can classify the toxicity of the comment with great accuracy. We achieved ~98.13% accuracy, which is impressive.
