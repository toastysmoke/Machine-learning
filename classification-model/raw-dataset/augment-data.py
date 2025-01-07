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
    transforms.RandomRotation(60),  # Max offset of 60 degrees
    transforms.ColorJitter(brightness=(0.3, 1.0)),  # Brightness minimum is 0.3
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ColorJitter(hue=0.5),
]

# Function to apply augmentation and save images
def augment_and_save_images(source_dir, augmented_dir, num_augmentations=5):
    for filename in os.listdir(source_dir):
        img_path = os.path.join(source_dir, filename)
        if os.path.isfile(img_path):
            # Read the image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            for i in range(num_augmentations):
                # Randomly select a subset of transformations
                num_transforms = random.randint(1, len(augmentation_transforms))
                selected_transforms = random.sample(augmentation_transforms, k=num_transforms)
                transform = transforms.Compose(selected_transforms + [transforms.ToTensor()])
                
                # Apply the transformations
                augmented_img = transform(img)
                
                # Convert the tensor back to a PIL image
                augmented_img = transforms.ToPILImage()(augmented_img)
                
                # Save the augmented image with a unique filename
                new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                augmented_img.save(os.path.join(augmented_dir, new_filename))

for category in categories:
    # Get the path for dog and cat images
    path = os.path.join(base_dir, category, 'train')
    
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        continue
    
    augmented_path = os.path.join(base_dir, category, 'augmented')
    os.makedirs(augmented_path, exist_ok=True)
    
    augment_and_save_images(path, augmented_path)