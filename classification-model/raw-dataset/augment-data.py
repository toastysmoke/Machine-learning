import os
import numpy as np
from torchvision import transforms
from PIL import Image

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

categories = ['cats', 'dogs']

# Define data augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor()
])

# Function to apply augmentation and save images
def augment_and_save_images(source_dir, augmented_dir, transform):
    for filename in os.listdir(source_dir):
        img_path = os.path.join(source_dir, filename)
        if os.path.isfile(img_path):
            # Read the image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Apply the transformations
            augmented_img = transform(img)
            
            # Convert the tensor back to a PIL image
            augmented_img = transforms.ToPILImage()(augmented_img)
            
            # Save the augmented image
            augmented_img.save(os.path.join(augmented_dir, filename))

for category in categories:
    # Get the path for dog and cat images
    path = os.path.join(base_dir, category, 'train')
    
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        continue
    
    augmented_path = os.path.join(base_dir, category, 'augmented')
    os.makedirs(augmented_path, exist_ok=True)
    
    augment_and_save_images(path, augmented_path, augmentation_transforms)