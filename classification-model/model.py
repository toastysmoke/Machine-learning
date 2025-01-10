import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score
from PIL import Image

# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, folder, label, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                if self.transform:
                    img = self.transform(img)
                self.images.append(img)
                self.labels.append(label)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Define the enhanced CNN model
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 9 * 9, 512)  # Adjusted for 150x150 input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 9 * 9)  # Adjusted for 150x150 input size
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Define paths and other setup
    base_dir = os.path.dirname(__file__)
    categories = ['cats', 'dogs']
    data_dir = os.path.join(base_dir, 'raw-dataset')

    # Define transforms for the training and test data
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Prepare training data from the augmented dataset
    train_data = []
    for category in categories:
        label = categories.index(category)
        train_folder = os.path.join(data_dir, category, 'augmented')  # Use 'augmented' folder for augmented dataset
        train_data.append(ImageDataset(train_folder, label, transform=train_transform))

    train_dataset = torch.utils.data.ConcatDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Prepare test data
    test_data = []
    for category in categories:
        label = categories.index(category)
        test_folder = os.path.join(data_dir, category, 'test')
        test_data.append(ImageDataset(test_folder, label, transform=test_transform))

    test_dataset = torch.utils.data.ConcatDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model, loss function, and optimizer
    model = EnhancedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjusted learning rate

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Early stopping parameters
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Train the model
    num_epochs = 50  # Increased number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)  # Update the learning rate based on the average loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(base_dir, 'best_model.pth')))

    # Test the model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the final model
    model_path = os.path.join(base_dir, 'enhanced_cnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")