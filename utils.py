#utility functions
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_height, img_width):
    ###################################DATA AUGMENTATIONS###############################
    train_transform = A.Compose(
            [
                ### START YOUR CODE HERE
                A.Resize(height=img_height, width=img_width),
                A.Normalize(mean=[0,0,0],std=[1,1,1]),
                A.Rotate(limit=90),
                A.Flip(p=0.5),
                ### END YOUR CODE HERE
                ToTensorV2(),   
            ],
        )

    val_transform = A.Compose(
            [
                ### START YOUR CODE HERE
                A.Resize(height=img_height, width=img_width),
                A.Normalize(mean=[0,0,0], std=[1,1,1]),
                ### END YOUR CODE HERE
                ToTensorV2(),
            ],
        )
    ###################################################################################

    return train_transform, val_transform
    
def check_accuracy(loader, model, device="cuda"):
  """
  Calculate the pixel accuracy, Jaccard Index (IoU), and Dice Score of a binary segmentation model on a given data loader.

  Args:
      loader : The data loader containing input images and corresponding ground truth masks.
      model : The binary segmentation model to evaluate in our case the UNET model you have trained.
      device (str, optional): The device on which to perform the evaluation, e.g., "cuda" or "cpu". Defaults to "cuda".

  Returns:
      Tuple[float, float, float]: A tuple containing the following metrics:
          - Accuracy (Pixel Accuracy): The ratio of correctly predicted pixels to the total number of pixels in the dataset.
          - Dice Score: Average Dice Score across the dataset.
          - Jaccard Index: Average Jaccard Index (IoU) across the dataset.

  Note:
      - The model output needs to be passed through sigmoid function and thresholding needs to be applied to obtain binary masks (0 or 1).
      - A threshold of 0.5 is applied to the probability maps to create the binary masks.
      - The Jaccard Index and Dice Score are calculated for each batch and then averaged across the entire dataset.
  """
 
  model.eval()
  #START YOUR CODE HERE
  
  accuracy = 0.0
  dice_score = 0.0
  jaccard_index = 0.0
  num_batches = len(loader)
  num_pixels = 0

  for batch in loader:
    images, masks = batch
    images, masks = images.to(device), masks.to(device)

    # Forward pass
    outputs = model.forward(images)
            
    # Apply sigmoid and thresholding to obtain binary masks
    predicted_masks = (torch.sigmoid(outputs) > 0.5).float()

    # Calculate pixel accuracy for the batch
    correct_pixels = (predicted_masks == masks).sum().item()
    num_pixels += masks.numel()
    accuracy += correct_pixels

    # Calculate Dice Score and Jaccard Index for the batch
    intersection = (predicted_masks * masks).sum().item()
    union = (predicted_masks + masks).sum().item()
    dice_score += (2 * intersection) / (union + 1e-5)
    jaccard_index += intersection / (union - intersection + 1e-5)

    # Calculate the average metrics
  batches = len(loader)
  accuracy /= num_pixels
  dice_score /= batches
  jaccard_index /= batches
    
  model.train()
  return (accuracy*100, dice_score, jaccard_index)
 #END YOU CODE HERE



def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    """
    Save a model checkpoint, including the model's state_dict and optimizer's state_dict, to a file.

    Args:
        state (dict): A dictionary containing the model's state_dict and optimizer's state_dict.
        filename (str, optional): The name of the file to save the checkpoint. Defaults to "unet_checkpoint.pth.tar".

    Returns:
        None

    Note:
        This function is used to save model checkpoints during training for later use or resuming training.
    """
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Load a model checkpoint, including the model's state_dict, from a checkpoint dictionary.

    Args:
        checkpoint (dict): A dictionary containing the model's state_dict.
        model: The model to which the state_dict should be loaded.

    Returns:
        None

    Note:
        This function is used to load a previously saved model checkpoint.
    """
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    """
    Save model predictions and ground truth images as PNG files to a specified folder.

    Args:
        loader: DataLoader providing batches of data.
        model: The model used for making predictions.
        folder (str): The folder where prediction images and ground truth images will be saved.
        device (str, optional): The device (e.g., "cuda" or "cpu") on which the model is evaluated. Defaults to "cuda".

    Returns:
        None

    Note:
        This function is typically used to save model predictions as images for visual inspection and analysis.
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}/{idx}.png")

    model.train()
