import os
import shutil
from sklearn.model_selection import train_test_split

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

categories = ['cats', 'dogs']

for category in categories:
    # Get the path for dog and cat images
    path = os.path.join(base_dir, category)
    
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        continue
    
    images = os.listdir(path)
    
    # Split the images into training and testing sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create directories to store processed images in processed-training-dataset
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Move training images
    for img in train_images:
        img_path = os.path.join(path, img)
        if os.path.isfile(img_path):
            shutil.move(img_path, os.path.join(train_path, img))
            print(f"Moved {img} to {train_path}")
        else:
            print(f"File {img_path} is not a valid image.")
    
    # Move testing images
    for img in test_images:
        img_path = os.path.join(path, img)
        if os.path.isfile(img_path):
            shutil.move(img_path, os.path.join(test_path, img))
            print(f"Moved {img} to {test_path}")
        else:
            print(f"File {img_path} is not a valid image.")
    
    print(f"Moved {len(train_images)} training images and {len(test_images)} testing images for category '{category}'.")