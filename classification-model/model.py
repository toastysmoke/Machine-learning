import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Define paths
base_dir = os.path.dirname(__file__)
categories = ['cats', 'dogs']
data_dir = os.path.join(base_dir, 'raw-dataset')

# Function to load images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.reshape(150, 150, 1)  # Reshape for CNN input
            images.append(img)
            labels.append(label)
    return images, labels

# Prepare training data
X_train = []
y_train = []
for category in categories:
    label = categories.index(category)
    train_folder = os.path.join(data_dir, category, 'train')
    images, labels = load_images_from_folder(train_folder, label)
    X_train.extend(images)
    y_train.extend(labels)

X_train = np.array(X_train)
y_train = to_categorical(np.array(y_train), num_classes=2)

# Prepare test data
X_test = []
y_test = []
for category in categories:
    label = categories.index(category)
    test_folder = os.path.join(data_dir, category, 'test')
    images, labels = load_images_from_folder(test_folder, label)
    X_test.extend(images)
    y_test.extend(labels)

X_test = np.array(X_test)
y_test = to_categorical(np.array(y_test), num_classes=2)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model_path = os.path.join(base_dir, 'cnn_model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")