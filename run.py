import albumentations as A
from dataset import SegDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils import *
from train import train_epoch
import torch.optim as optim

def train_unet(model,device):
    """
    Main training and evaluation function for a binary segmentation model.

    This function performs the following steps:
    1. Define an appropirate loss function for the problem(https://pytorch.org/docs/stable/nn.html#loss-functions) and
        optimizer (https://pytorch.org/docs/stable/optim.html).
    2. Iterate through training epochs:
        - Train the model on the training data using the defined loss function and optimizer.
        - Evaluate the model's accuracy, Dice Score, and Jaccard Index on the validation data.
        - Record accuracy, Dice Score, and Jaccard Index for each epoch in a global array(This will be used by you to plot the graphs later).
        - Save prediction examples to a specified folder.

    Args:
        None

    Returns:
        (Tuple[list, list, list]: A tuple containing lists of  for accuracy, Dice Score, and Jaccard Index for each epoch.

    Note:
        - This function assumes the existence of the following variables/constants:
            - model: The binary segmentation model to be trained and evaluated.
            - LEARNING_RATE: The learning rate used for the optimizer.
            - NUM_EPOCHS: The number of training epochs.
            - train_loader: DataLoader for training data.
            - val_loader: DataLoader for validation data.
            - device: The device (e.g., "cuda" or "cpu") on which the model is trained and evaluated.
        - You should provide appropriate values for these variables/constants before calling this function.

    """
    ####################################HYPERPARAMETERS#################################
    learning_rate = 1e-4
    num_epochs = 20
    img_height=img_width=512
    batch_size=8
    train_dir="/content/drive/MyDrive/CS 6476 Intro to Computer Vision/Assignment 4/Data/train"
    test_dir="/content/drive/MyDrive/CS 6476 Intro to Computer Vision/Assignment 4/Data/test"
    save_path="/content/drive/MyDrive/CS 6476 Intro to Computer Vision/Assignment 4/student_submission"
    ####################################################################################
  


    ###################################DATA AUGMENTATIONS###############################
    train_transform, val_transform=get_transforms(img_height, img_width)
    ###################################################################################

    ###################################DATALOADERS#####################################
    train_ds = SegDataset(
            dir=train_dir,
            transform=train_transform,
        )

    val_ds = SegDataset(
            dir = test_dir,
            transform=val_transform,
        )

    train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
        )


    val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
        )
    ###################################################################################
    #######################MAIN TRAINING LOOP#########################################

    '''
    Main training and evaluation loop for a binary segmentation model.

        In this part perform the following steps:
        1. Define an apporpirate loss function for the problem(https://pytorch.org/docs/stable/nn.html#loss-functions) and
        optimizer (https://pytorch.org/docs/stable/optim.html).
        2. Iterate through training epochs:
            - Train the model on the training data using the defined loss function and optimizer.
            - Save the model checkpoint, including the model's state_dict and optimizer's state_dict.
            - Evaluate the model's accuracy, Dice Score, and Jaccard Index on the validation data.
            - Record accuracy, Dice Score, and Jaccard Index for each epoch in a global array(This will be used by you to plot the graphs later).
            - Save prediction examples to a specified folder.
        
    '''
    ##START YOU CODE HERE

    # Define loss and optimizer
    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    # Iterate training epochs
    accuracies = []
    dices = []
    jaccards = []
    for epoch in range(num_epochs):
      print(f"Epoch {epoch}")
      model.train()
      t_loss = train_epoch(train_loader, model, optimizer, loss_fn,device)
      #save_predictions_as_imgs(train_loader, model, save_path, device='cpu')
      model.eval()
      accuracy, dice, jaccard = check_accuracy(train_loader, model, device='cuda')
      accuracies.append(accuracy)
      dices.append(dice)
      jaccards.append(jaccard)

    return(accuracies, dices, jaccards)

    

    ###END CODE HERE