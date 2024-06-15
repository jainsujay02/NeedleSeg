import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import transforms
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import remove_small_objects, remove_small_holes, closing, square, dilation, erosion
import torch.nn.functional as F
from typing import Union
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Check for GPU availability and set the device accordingly
device = torch.device('cuda:0')
print("Using device:", device)

def load_data(image_dir, mask_dir, image_size=(512, 512), batch_size=4, num_samples=None, exclude_indices=None):
    images = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.jpg')])
    masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith('.png')])

    if exclude_indices is not None:
        images = [img for i, img in enumerate(images) if i not in exclude_indices]
        masks = [mask for i, mask in enumerate(masks) if i not in exclude_indices]

    if num_samples is not None:
        images = images[:num_samples]
        masks = masks[:num_samples]

    num_images = len(images)
    num_batches = int(np.ceil(num_images / batch_size))

    X_train = np.zeros((num_batches, batch_size, 1, *image_size), dtype=np.float32)
    Y_train = np.zeros((num_batches, batch_size, 1, *image_size), dtype=np.float32)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]
        batch_masks = masks[start_idx:end_idx]

        for i, (image_path, mask_path) in enumerate(zip(batch_images, batch_masks)):
            image = resize(imread(image_path, as_gray=True), image_size, mode='constant', preserve_range=True)
            mask = resize(imread(mask_path, as_gray=True), image_size, mode='constant', preserve_range=True)
            mask = (mask > 0).astype(np.float32)  # Threshold mask to be exactly 0 or 1
            X_train[batch_idx, i, 0, :, :] = image
            Y_train[batch_idx, i, 0, :, :] = mask

    # Convert to tensors and move to the specified device
    return torch.tensor(X_train).to(device), torch.tensor(Y_train).to(device)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.encoder_block(1, 64)
        self.encoder2 = self.encoder_block(64, 128)
        self.encoder3 = self.encoder_block(128, 256)
        self.encoder4 = self.encoder_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Adjust for proper upsampling
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_block(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, F.interpolate(enc4, dec4.size()[2:], mode='bilinear', align_corners=False)), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, F.interpolate(enc3, dec3.size()[2:], mode='bilinear', align_corners=False)), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, F.interpolate(enc2, dec2.size()[2:], mode='bilinear', align_corners=False)), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, F.interpolate(enc1, dec1.size()[2:], mode='bilinear', align_corners=False)), dim=1)
        dec1 = self.decoder1(dec1)

        # Final upsampling and convolution to get the same size as input
        dec1 = self.final_upsample(dec1)
        dec1 = F.interpolate(dec1, size=x.size()[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(self.final_conv(dec1))

# Instantiate model, move model to the correct device
model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

train_dir = 'path/to/trainImages'
mask_dir = 'path/to/trainMasks'
batch_size = 4

# Select 100 images for testing and exclude them from training
num_total_images = len([x for x in os.listdir(train_dir) if x.endswith('.jpg')])
test_indices = np.random.choice(num_total_images, 100, replace=False)

# Load full training data excluding the test indices
X_train, Y_train = load_data(train_dir, mask_dir, batch_size=batch_size, exclude_indices=test_indices)

# Load testing data
X_test, Y_test = load_data(train_dir, mask_dir, batch_size=batch_size, num_samples=100, exclude_indices=None)

class ModelCheckpoint:
    """Saves the best model observed during training based on validation loss."""
    def __init__(self, save_path='unet_model_state.pth'):
        self.save_path = save_path
        self.val_loss_min = np.Inf

    def update(self, val_loss, model, optimizer):
        if val_loss < self.val_loss_min:
            print(f'Saving new best model with improved validation loss: {val_loss:.6f}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, self.save_path)
            self.val_loss_min = val_loss

def train_validate_model(X_train, Y_train, num_epochs=20, k_folds=10):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    model_checkpoint = ModelCheckpoint()

    train_loss_over_epochs = []
    val_loss_over_epochs = []
    train_dice_over_epochs = []
    val_dice_over_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"Training fold {fold}/{k_folds}")

        # Splitting the data for this fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

        # Initialize the model and optimizer for each fold
        model = UNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = DiceLoss()

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_dice = 0.0
            for batch_idx in range(X_train_fold.shape[0]):
                optimizer.zero_grad()
                outputs = model(X_train_fold[batch_idx])
                loss = criterion(outputs, Y_train_fold[batch_idx])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_dice += 1 - loss.item()  # DICE coefficient is 1 - DICE loss

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_dice = 0.0
            with torch.no_grad():
                for batch_idx in range(X_val_fold.shape[0]):
                    outputs = model(X_val_fold[batch_idx])
                    val_loss = criterion(outputs, Y_val_fold[batch_idx])
                    val_running_loss += val_loss.item()
                    val_running_dice += 1 - val_loss.item()  # DICE coefficient is 1 - DICE loss

            val_loss_epoch = val_running_loss / X_val_fold.shape[0]
            train_loss_epoch = running_loss / X_train_fold.shape[0]
            train_dice_epoch = running_dice / X_train_fold.shape[0]
            val_dice_epoch = val_running_dice / X_val_fold.shape[0]
            train_loss_over_epochs.append(train_loss_epoch)
            val_loss_over_epochs.append(val_loss_epoch)
            train_dice_over_epochs.append(train_dice_epoch)
            val_dice_over_epochs.append(val_dice_epoch)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss_epoch}, Val Loss: {val_loss_epoch}')
            model_checkpoint.update(val_loss_epoch, model, optimizer)

        print(f"Fold {fold} completed.")

    # Plotting loss and DICE curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_over_epochs, label='Train Loss')
    plt.plot(val_loss_over_epochs, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.savefig('training_validation_loss_curve.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_dice_over_epochs, label='Train DICE')
    plt.plot(val_dice_over_epochs, label='Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('DICE')
    plt.legend()
    plt.title('Training and Validation DICE Curves')
    plt.savefig('training_validation_dice_curve.png')
    plt.show()

# Train the model using k-fold cross-validation
train_validate_model(X_train, Y_train, num_epochs=20, k_folds=10)

# Load the best model
checkpoint = torch.load('unet_model_state.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.to(device)  # Move the loaded model to the correct device

# Set the model to evaluation mode
model.eval()

# Prediction and post-processing function
def predict_and_process(test_dir, save_dir):
    test_images = sorted([os.path.join(test_dir, x) for x in os.listdir(test_dir) if x.endswith('.jpg')])
    ensure_dir(save_dir)
    for image_path in test_images:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        image = imread(image_path, as_gray=True)
        image = resize(image, (512, 512), mode='constant', preserve_range=True)
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)  #

        # Predict the mask with the model
        with torch.no_grad():  # Ensure no computation is done for gradients
            pred_mask = model(image)

        # Convert predictions to binary mask
        pred_mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255  # Threshold predictions

        # Post-process the binary mask
        pred_mask = remove_small_objects(pred_mask.astype(bool), min_size=64).astype(np.uint8) * 255
        pred_mask = remove_small_holes(pred_mask.astype(bool), area_threshold=64).astype(np.uint8) * 255
        pred_mask = closing(pred_mask.astype(bool), square(2)).astype(np.uint8) * 255  # Use morphological closing with a 5x5 square kernel

        # Additional morphological operations
        pred_mask = dilation(pred_mask, square(2))
        pred_mask = erosion(pred_mask, square(2))

        # Save the predicted mask
        Image.fromarray(pred_mask).save(f'{save_dir}/{base_name}_mask.png')

    print("Evaluation complete and masks saved.")

# Helper function to ensure directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to process the masks and create submission.csv
def process_images_to_submission(img_directory, save_directory='submissions', return_df=False):
    files = [f for f in os.listdir(img_directory) if f.endswith('.png')]
    if len(files) != 127:
        raise ValueError("Directory must contain exactly 127 .png files")
    files.sort(key=lambda x: int(x.split('_')[0]))  # Sort files numerically based on prefix

    data = []
    for file in files:
        img_path = os.path.join(img_directory, file)
        img = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)

        if not np.array_equal(img, img.astype(bool).astype(img.dtype) * 255):
            raise ValueError(f"Image {file} is not binary")
        if img.shape != (512, 512):
            raise ValueError(f"Image {file} is not of size 512x512 pixels")

        status = 1 if np.any(img == 255) else 0
        mask_indices = ' '.join(map(str, np.nonzero(img.flatten() == 255)[0])) if status else '-100'

        data.append({'imageID': int(file.split('_')[0]), 'status': status, 'mask': mask_indices})

    df = pd.DataFrame(data).set_index('imageID')
    ensure_dir(save_directory)
    df.to_csv(os.path.join(save_directory, 'submission.csv'))

    if return_df:
        return df

# Predict and process images
predicted_masks_dir = 'predicted_masks'
predict_and_process(train_dir, predicted_masks_dir)

# # Create submission file
# process_images_to_submission(predicted_masks_dir)
# print("Processing and saving of mask data completed.")

# Assuming a function to compute metrics
# Define the epsilon value to avoid division by zero
epsilon = 1e-7

# Function to compute metrics
def compute_metrics(pred, true):
    pred = pred > 0.5
    true = true > 0.5

    intersection = (pred & true).sum((1, 2, 3)).float()
    union = (pred | true).sum((1, 2, 3)).float()
    pred_sum = pred.sum((1, 2, 3)).float()
    true_sum = true.sum((1, 2, 3)).float()

    dice = (2 * intersection + epsilon) / (pred_sum + true_sum + epsilon)
    iou = (intersection + epsilon) / (union + epsilon)
    sensitivity = (intersection + epsilon) / (true_sum + epsilon)

    return dice.mean().item(), sensitivity.mean().item(), iou.mean().item()

# Evaluate the model
dices, sensitivities, ious = [], [], []
with torch.no_grad():
    for i in range(X_test.shape[0]):
        outputs = model(X_test[i])
        dice, sensitivity, iou = compute_metrics(outputs, Y_test[i])
        dices.append(dice)
        sensitivities.append(sensitivity)
        ious.append(iou)

mean_dice = np.mean(dices)
mean_sensitivity = np.mean(sensitivities)
mean_iou = np.mean(ious)

# Print computed metrics
print(f'Mean DICE: {mean_dice}')
print(f'Mean Sensitivity: {mean_sensitivity}')
