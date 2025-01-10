import os
import numpy as np
from torchvision import transforms
from PIL import Image
import random

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

categories = ['cats', 'dogs']

# Define individual augmentation transforms
augmentation_transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  # Max offset of 30 degrees
    transforms.ColorJitter(brightness=(0.3, 1.0)),  # Brightness minimum is 0.3
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ColorJitter(hue=0.5),
]

# Define normalization transform
normalize_transform = transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5

# Function to apply augmentation and save images
def augment_and_save_images(source_dir, augmented_dir, num_augmentations=5):
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
    
    for filename in os.listdir(source_dir):
        img_path = os.path.join(source_dir, filename)
        if os.path.isfile(img_path):
            # Read the image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            for i in range(num_augmentations):
                # Randomly select a subset of transformations and normalize image
                num_transforms = random.randint(1, len(augmentation_transforms))
                selected_transforms = random.sample(augmentation_transforms, k=num_transforms)
                transform = transforms.Compose(selected_transforms + [transforms.ToTensor(), normalize_transform])
                
                # Apply the transformations
                augmented_img = transform(img)
                
                # Convert the tensor back to a PIL image
                augmented_img = transforms.ToPILImage()(augmented_img)
                
                # Save the augmented image with a unique filename
                augmented_img.save(os.path.join(augmented_dir, filename))

# Example usage
if __name__ == '__main__':
    for category in categories:
        source_dir = os.path.join(base_dir, category, 'train')
        augmented_dir = os.path.join(base_dir, category, 'augmented')
        augment_and_save_images(source_dir, augmented_dir)