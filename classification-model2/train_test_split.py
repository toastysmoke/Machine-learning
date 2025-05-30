import os
import sklearn
import numpy
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Create dataset class
class BrainCancerDataset(Dataset):

    def __init__(self, images, labels, transform=None):

        self.images = torch.from_numpy(images).float()
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_item = self.images[idx]
        label = self.labels[idx]
        return image_item, label
    




class BrainCancerModel(nn.Module):
    def __init__(self, class_num):
        super(BrainCancerModel, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2) 

        self.flattened_features_size = 256 * 32 * 32  # Size after flattening the output of the last pooling layer
        self.fc1 = nn.Linear(self.flattened_features_size, 256)
        self.fc2 = nn.Linear(256, class_num)

        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(-1, self.flattened_features_size)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x
    

def main():
    # Data directories
    no_tumor_directory = "brain_cancer_data/no_tumor"
    tumor_directory = "brain_cancer_data/tumor"

    image_paths = [no_tumor_directory, tumor_directory]

    # Initialize lists to hold the loaded images and their corresponding labels according to their index in the list
    loaded_images = []
    image_labels = []

    # no_tumor = 0
    # tumor = 1
    class_counter = 0

    # Loop through each directory and process the images
    for image_path in image_paths:

        files = os.listdir(image_path)

        for file in files:
            full_path = os.path.join(image_path, file)

            # Open the image file
            image = Image.open(full_path)
            image = image.convert("L")
            image = image.resize((512, 512))  # Resize the image to 512x512 pixels
            # Image already greyscale 512 * 512 and in jpg format
            # Convert the image to a numpy array, add color channel dimension of 1, and normalize pixel values
            image_array = numpy.array(image)
            image_array = image_array[numpy.newaxis, :, :]
            image_array = image_array.astype('float32') / 255.
            # Append the processed image array to the list
            loaded_images.append(image_array)
            # Append the corresponding label to the labels list
            image_labels.append(class_counter)
        
        print(f"Finished processing images in class label: {class_counter}")
        class_counter += 1

    # Convert the list of image arrays to a numpy array
    loaded_images = numpy.array(loaded_images)

    images_train, images_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
        loaded_images,
        image_labels,
        test_size=0.2,
        random_state=42,
        stratify=image_labels
    )

    # Specify data loader parameters
    batch_size = 32
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    # Create dataset instances for training and testing
    train_dataset = BrainCancerDataset(images_train, labels_train)
    test_dataset = BrainCancerDataset(images_test, labels_test)

    # Create DataLoader for training and testing dataset
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


    # Print the number of training and testing samples
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Training parameters
    num_epochs = 200
    learning_rate = 0.001
    class_num = 1

    # Initialize the model
    model = BrainCancerModel(class_num)

    # Define the loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)



    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ensure labels are of shape (batch_size, 1)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.2f}%")

    
    # Evaluation loop
    correct = 0
    total = 0
    test_loss = 0.0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ensure labels are of shape (batch_size, 1)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Testing Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()