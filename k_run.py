import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import transforms
from skimage.io import imread
from skimage.transform import resize
import torch.nn.functional as F
from typing import Union
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image

from sklearn.model_selection import KFold
import numpy as np



# Check for GPU availability and set the device accordingly
device = torch.device('cuda:0')
print("Using device:", device)

def load_data(image_dir, mask_dir, image_size=(512, 512), batch_size=4):
    images = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.jpg')])
    masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith('.png')])
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
mask_dir = '/path/to/trainMasks'
batch_size = 4
X_train, Y_train = load_data(train_dir, mask_dir, batch_size=batch_size)
model_path = 'path/to/model/unet_model_state.pth'



class ModelCheckpoint:
    """Saves the best model observed during training based on validation loss."""
    def __init__(self, save_path='unet_model_state.pth'):
        """
        Args:
            save_path (str): Path for the checkpoint to be saved to.
                             Default: 'unet_model_state.pth'
        """
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

def train_validate_model(X_train, Y_train, num_epochs=10, k_folds=10):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    model_checkpoint = ModelCheckpoint(model_path)

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
            for batch_idx in range(X_train_fold.shape[0]):
                optimizer.zero_grad()
                outputs = model(X_train_fold[batch_idx])
                loss = criterion(outputs, Y_train_fold[batch_idx])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch_idx in range(X_val_fold.shape[0]):
                    outputs = model(X_val_fold[batch_idx])
                    val_loss = criterion(outputs, Y_val_fold[batch_idx])
                    val_running_loss += val_loss.item()

            val_loss_epoch = val_running_loss / X_val_fold.shape[0]
            print(f'Epoch {epoch+1}, Train Loss: {running_loss / X_train_fold.shape[0]}, Val Loss: {val_loss_epoch}')
            model_checkpoint.update(val_loss_epoch, model, optimizer)

        print(f"Fold {fold} completed.")

train_validate_model(X_train, Y_train, num_epochs=150, k_folds=10)

#  load the model
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.to(device)

# Set the model to evaluation mode
model.eval()

test_dir = 'path/to/testImages'
test_images = sorted([os.path.join(test_dir, x) for x in os.listdir(test_dir) if x.endswith('.jpg')])

for image_path in test_images:
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    image = imread(image_path, as_gray=True)
    image = resize(image, (512, 512), mode='constant', preserve_range=True)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions and move to GPU

    # Predict the mask with the model
    with torch.no_grad():  # Ensure no computation is done for gradients
        pred_mask = model(image)

    # Convert predictions to binary mask
    pred_mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255  # Threshold predictions and move to CPU for processing

    # Save the predicted mask
    Image.fromarray(pred_mask).save(f'predicted_masks/{base_name}_mask.png')

print("Evaluation complete and masks saved.")

def processImages(imgDirectory: str, saveDirectory: str = os.getcwd(), returnDF: bool = False) -> Union[pd.DataFrame, None]:
    files = [f for f in os.listdir(imgDirectory) if f.endswith('.png')]
    if len(files) != 127:
        raise ValueError("Directory must contain exactly 127 .png files")
    files.sort(key=lambda x: int(x.split('_')[0]))  # Sort files numerically based on prefix

    data = []
    for file in files:
        imgPath = os.path.join(imgDirectory, file)
        img = np.array(Image.open(imgPath).convert('L'), dtype=np.uint8)

        if not np.array_equal(img, img.astype(bool).astype(img.dtype) * 255):
            raise ValueError(f"Image {file} is not binary")
        if img.shape != (512, 512):
            raise ValueError(f"Image {file} is not of size 512x512 pixels")

        status = 1 if np.any(img == 255) else 0
        maskIndices = ' '.join(map(str, np.nonzero(img.flatten() == 255)[0])) if status else '-100'

        data.append({'imageID': int(file.split('_')[0]), 'status': status, 'mask': maskIndices})

    df = pd.DataFrame(data).set_index('imageID')
    df.to_csv(os.path.join(saveDirectory, 'submission.csv'))

    if returnDF: return df

# Assuming the directory for predicted masks and where to save submission
processImages('path/to/predicted_masks', 'submissions', returnDF=False)
print("Processing and submissions saved")