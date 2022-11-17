#Assignment 1 Classification using Convolutional Neural Networks

The goal of this project is to build a CNN for an animal classification task. The images of animal
comes from [Hit dataset](https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/exp5.html). The classification consists
of 20 classes (19 animals) and 1 human faces class.


### Installation
# To install the required environment for assignment 1, a standard way is to use the following commands:
pip install -r requirements.txt

# This will, by default, install a CPU version of PyTorch. 

### Training
# To train the binary classifier with the provided start code, you need to change the working directory to "AER1515_Assignment1/code/", and run "train.py".
# When the training is done, the program should generate a file called "model.pt" and show a plot of the training loss curve.

### Testing
# To test the trained CNN model, you need to run "test.py". The classification accuracy will be reported.

### Dataset
# All the training and testing data are included in the working directory of "AER1515_Assignment1/AnimalFace/".