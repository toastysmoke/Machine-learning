import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the enhanced CNN model
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Pooling after first conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Pooling after second conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Pooling after third conv layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Pooling after fourth conv layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Pooling after fifth conv layer
        self.dropout_conv = nn.Dropout(0.3)  # Dropout after convolutional layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Adjusted for 150x150 input size
        self.dropout_fc1 = nn.Dropout(0.5)  # Dropout after first fully connected layer
        self.fc2 = nn.Linear(1024, 2)  # Output layer

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) 
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout_conv(x)  # Apply dropout after convolutional layers
        x = x.view(-1, 512 * 4 * 4)  # Adjusted for 150x150 input size
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)  # Apply dropout after first fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Load the model
model = EnhancedCNN()
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()  # Set the model to evaluation mode

# Prepare the input data
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize to the same size as during training
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with the same mean and std as during training
])

def predict(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    start_time = time.time()  # Start timing
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    end_time = time.time()  # End timing
    inference_time = end_time - start_time
    return predicted.item(), inference_time

# Example usage
image_path = os.path.join(os.path.dirname(__file__), 'image', 'cattodoggo.jpg')
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")
prediction, inference_time = predict(image_path)
if prediction == 0:
    print('Predicted class: cat')
else:
    print('Predicted class: dog')
print(f'Inference time: {inference_time:.6f} seconds')