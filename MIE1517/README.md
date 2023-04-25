# Automated Essay Scoring with Multi-Aspect Ensemble Model

This project aims to build a multi-aspect automated essay scoring model using an ensemble of neural networks. The model takes essays as input and predicts scores for various writing aspects such as grammar, vocabulary, cohesion, and more.

The following is a link to the presenetation made for this project: [Video](https://youtu.be/fNbPEZ__sWo)


## Getting Started
### Prerequisites
To run this project, clone the repository and after install the required packages. 

### Dataset
The dataset used in this project is the Kaggle AES Competition dataset, which includes essays written by students in grades 8-12. The essays are scored on a scale of 0-5 for writing proficiency, with scores given for various writing aspects such as grammar, vocabulary, cohesion, and more [Dataset](https://www.kaggle.com/competitions/feedback-prize-english-language-learning).


### Model
The model used in this project is a ensemble of neural networks. Different transformers were used wit transfer learning such as BERT or Deberta


### Result 
The model achieved an average RMSE loss of 0.46, comparable to the Kaggle competition winner's score of 0.44. The model was able to distinguish between good and bad essays.


### Future Work 
Possible future improvements to the model include incorporating a variance term in the loss function to allow for more accurate fits to the data, and expanding the scope of the dataset to include a wider range of text types and writing levels. The model can also improve by fine-tuning the transformer model.