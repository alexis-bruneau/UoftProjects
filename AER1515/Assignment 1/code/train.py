from matplotlib.rcsetup import validate_int
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torch.backends.cudnn as cudnn
random.seed(1000)
torch.manual_seed(1000)
np.random.seed(1000)
cudnn.deterministic = True
import torch.optim as optim
from glob import glob
import os
from animal_face_dataset import *
from Animal_Classification_Network import *
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split


def main():

    # hyper-parameters for network training
    ###############################################################################
    # TODO: Extend the binary classification to multi-class classification
    N_CLASSES = 20 # num of classes
    BATCH_SIZE = 64 # training batch size
    EPOCH_NUMBER = 45 # num of epochs
    VALIDATION_PER = 0.2 # Validation Percentage (you can play with this parameter)
    LEARNING_RATE = 1e-4 # Learning Rate
    IS_SHOW_IMAGES = False
    ###############################################################################



    # Load training dataset
    ###########################################################################################################
    # DO NOT MAKE ANY CHANGES IN THIS BLOCK
    label_map = {0: "Cat", 1: "Dog", 2: "Bear", 3: "Chicken", 4: "Cow", 5: "Deer", 6: "Duck", 7: "Eagle",
                 8: "Elephant", 9: "Human", 10: "Lion", 11: "Monkey", 12: "Mouse", 13: "Panda", 14: "Pigeon",
                 15: "Pig", 16: "Rabbit", 17: "Sheep", 18: "Tiger", 19: "Wolf"}
    main_path = "../AnimalFace/train/"
    paths = []
    labels = []

    for i in range(N_CLASSES):
        folder = label_map[i] + 'Head'
        path_i = os.path.join(main_path, folder, "*")
        for each_file in glob(path_i):
            paths.append(each_file)
            labels.append(i)
    dataset = AnimalDataset(paths, labels, (150, 150))
    ###########################################################################################################



    # Split the full train dataset into "Train Set" and "Validation Set"
    ###########################################################################################################
    # DO NOT MAKE ANY CHANGES IN THIS BLOCK
    dataset_indices = list(range(0, len(dataset)))
    train_indices, test_indices = train_test_split(dataset_indices, test_size=VALIDATION_PER, random_state=42)
    print("Number of train samples: ", len(train_indices))
    print("Number of validation samples: ", len(test_indices))

    # Training Set
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler)

    # Validation Set: We have not used it in the training yet
    # !!! Students are expected to use the validation set to monitor the training progress !!!
    test_sampler = SubsetRandomSampler(test_indices)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=test_sampler)
    ###########################################################################################################



    # Show image examples, require matplotlib package
    # Not show by default
    ####################################################
    if IS_SHOW_IMAGES:
        images, labels = iter(train_loader).next()
        fig, axis = plt.subplots(3, 5, figsize=(15, 10))
        for i, ax in enumerate(axis.flat):
            with torch.no_grad():
                npimg = images[i].numpy()
                npimg = np.transpose(npimg, (1, 2, 0))
                label = label_map[int(labels[i])]
                ax.imshow(npimg)
                ax.set(title=f"{label}")
        plt.show()
    ####################################################



    # Set up device (gpu or cpu), load CNN model, define Loss function and Optimizer
    #################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    #optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    #################################################################################



    # Training!!!
    ####################################################################################
    TRAIN_LOSS = []
    VALIDATION_LOSS = []
    min_valid_loss = np.inf
    for epoch in range(1, EPOCH_NUMBER + 1):
        epoch_loss = 0.0
        
        for data_, target_ in train_loader:
            # Load data and label
            target_ = target_.to(device)
            data_ = data_.to(device)

            # Clean up the gradients
            optimizer.zero_grad()

            # Get output from our CNN model and compute the loss
            outputs = model(data_)
            loss = criterion(outputs, target_)

            # Backpropagation and optimizing our CNN model
            loss.backward()
            optimizer.step()

            # Compute loss
            epoch_loss = epoch_loss + loss.item()

        # TODO: Add validation Loop here
        #################################
        # Your Code
        valid_loss = 0.0
        for data_, label in validation_loader:
            label = label.to(device)
            data_ = data_.to(device)
            
            target = model(data_)
            loss = criterion(target, label)
            valid_loss += loss.item()
      
        
    
        #################################

        # Append result to the lists for each epoch
        ##############################################################################
        TRAIN_LOSS.append(epoch_loss/len(train_loader))
        print(f"Epoch {epoch}, Training Loss: {epoch_loss/len(train_loader)}")
        
        # TODO: Append validation results to the lists for each epoch
        # Your Code
        VALIDATION_LOSS.append(valid_loss/len(validation_loader))
        print(f"Epoch {epoch}, Validation Loss: {valid_loss/len(validation_loader)}")
        ##############################################################################


    # Save the model
    # TODO: Instead save the model here,
    # TODO: you should save the model with the minimal validation loss
        if min_valid_loss > valid_loss:
            print(f'The Validation Loss decrease {min_valid_loss} ---> {valid_loss} Saving Model ...')
            #print(f'Validation Loss Decreased({min_valid_loss:.6f\}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            
            # Saving State Dict
            torch.save(model.state_dict(), 'model.pt')



    #torch.save(model.state_dict(), "model.pt")


    # TODO: Plot the training loss and validation loss in the same graph
    #################################################################################
    # Your Code
    plt.subplots(figsize=(6, 4))
    plt.plot(range(EPOCH_NUMBER), TRAIN_LOSS, color="blue", label="Training Set")
    plt.plot(range(EPOCH_NUMBER), VALIDATION_LOSS, color="red", label="Validation Loss")
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.show()
    #################################################################################

    return



if __name__ == '__main__':
    main()