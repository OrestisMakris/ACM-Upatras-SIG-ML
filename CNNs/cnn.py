'''
# PyTorch CNN Tutorial: From Basics to Intermediate Level
# ======================================================
#
# This tutorial will guide you through implementing Convolutional Neural Networks (CNNs)
# in PyTorch, starting from basic concepts and gradually moving to more complex architectures.
#
# Contents:
# 1. Basic CNN components explanation
# 2. Simple CNN for MNIST
# 3. Intermediate CNN for CIFAR-10
# 4. Advanced CNN architecture (ResNet-like) for CIFAR-10
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# PART 1: UNDERSTANDING CNN COMPONENTS
# =============================================================================

'''
# 1.1 Convolutional Layers Explained
# ----------------------------------
# Convolutional layers are the core building blocks of CNNs.
# They apply a set of learnable filters (kernels) to input images.
# Each filter detects specific features (like edges, textures, patterns).
'''

def explain_convolution():
    # Create a simple 5x5 input (grayscale image)
    input_image = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Define a 3x3 kernel (edge detection)
    kernel = torch.tensor([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add output and input channel dimensions
    
    # Create a convolutional layer with the predefined kernel
    conv_layer = nn.Conv2d(1, 1, kernel_size=3, bias=False)
    conv_layer.weight.data = kernel
    
    # Apply convolution
    output = conv_layer(input_image)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_image.squeeze().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(kernel.squeeze().numpy(), cmap='gray')
    plt.title('Kernel (Edge Detection)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output.detach().squeeze().numpy(), cmap='gray')
    plt.title('Output Feature Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Conv2d operation explained:")
    print(f"Input shape: {input_image.shape} (batch_size, channels, height, width)")
    print(f"Kernel shape: {kernel.shape} (out_channels, in_channels, kernel_height, kernel_width)")
    print(f"Output shape: {output.shape}")
    print("\nThe kernel slides over the input image, performing element-wise")
    print("multiplication and summing the results to produce the output feature map.")

# Uncomment to run the convolution explanation:
# explain_convolution()

'''
# 1.2 Pooling Layers Explained
# ---------------------------
# Pooling layers reduce the spatial dimensions (width & height) of the input.
# They help to:
#   - Reduce computation
#   - Control overfitting
#   - Provide translation invariance
'''

def explain_pooling():
    # Create a simple 4x4 input feature map
    input_feature = torch.tensor([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15, 16]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Define max pooling and average pooling layers
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # Apply pooling
    max_output = max_pool(input_feature)
    avg_output = avg_pool(input_feature)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_feature.squeeze().numpy(), cmap='viridis')
    plt.title('Input Feature Map')
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{int(input_feature[0,0,i,j])}", 
                     ha="center", va="center", color="w")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(max_output.squeeze().numpy(), cmap='viridis')
    plt.title('After Max Pooling (2x2)')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(max_output[0,0,i,j])}", 
                     ha="center", va="center", color="w")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(avg_output.squeeze().numpy(), cmap='viridis')
    plt.title('After Average Pooling (2x2)')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{float(avg_output[0,0,i,j]):.1f}", 
                     ha="center", va="center", color="w")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Pooling operations explained:")
    print(f"Input shape: {input_feature.shape}")
    print(f"After pooling shape: {max_output.shape}")
    print("\nMax Pooling: Takes the maximum value from each window")
    print("Avg Pooling: Takes the average of all values in each window")

# Uncomment to run the pooling explanation:
# explain_pooling()

'''
# 1.3 Activation Functions
# -----------------------
# Activation functions introduce non-linearity to the network,
# allowing it to learn complex patterns.
'''

def explain_activations():
    # Generate input values
    x = torch.linspace(-5, 5, 1000)
    
    # Apply different activation functions
    relu = F.relu(x)
    leaky_relu = F.leaky_relu(x, 0.1)
    sigmoid = torch.sigmoid(x)
    tanh = torch.tanh(x)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x.numpy(), relu.numpy())
    plt.title('ReLU')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x.numpy(), leaky_relu.numpy())
    plt.title('Leaky ReLU (alpha=0.1)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x.numpy(), sigmoid.numpy())
    plt.title('Sigmoid')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x.numpy(), tanh.numpy())
    plt.title('Tanh')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Activation Functions Explained:")
    print("ReLU: f(x) = max(0, x) - Simple, computationally efficient, helps with vanishing gradient")
    print("Leaky ReLU: f(x) = max(αx, x) - Prevents 'dying ReLU' problem")
    print("Sigmoid: f(x) = 1/(1+e^-x) - Outputs between 0 and 1, used in binary classification")
    print("Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x) - Outputs between -1 and 1")

# Uncomment to run the activation functions explanation:
# explain_activations()

# =============================================================================
# PART 2: SIMPLE CNN FOR MNIST
# =============================================================================

'''
# Our first CNN will be trained on the MNIST dataset
# MNIST contains 28x28 grayscale images of handwritten digits (0-9)
'''

# Data Loading and Preprocessing for MNIST
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    
    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False
    )
    
    return train_loader, test_loader

# Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        # Input: 1x28x28 (MNIST images are grayscale, so 1 channel)
        # Output: 32 feature maps of size 26x26
        # (28 - 3 + 0)/1 + 1 = 26 (no padding, stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        
        # First max pooling layer
        # Input: 32x26x26
        # Output: 32x13x13
        # (26/2 = 13, rounded down)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        # Input: 32x13x13
        # Output: 64x11x11
        # (13 - 3 + 0)/1 + 1 = 11
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # Second max pooling layer
        # Input: 64x11x11
        # Output: 64x5x5
        # (11/2 = 5.5, rounded down to 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Input: 64 * 5 * 5 = 1600 (flattened feature maps)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Apply first convolution followed by ReLU activation
        x = F.relu(self.conv1(x))
        
        # Apply first pooling
        x = self.pool1(x)
        
        # Apply second convolution followed by ReLU activation
        x = F.relu(self.conv2(x))
        
        # Apply second pooling
        x = self.pool2(x)
        
        # Flatten the tensor
        x = x.view(-1, 64 * 5 * 5)
        
        # Apply first fully connected layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Apply final fully connected layer
        x = self.fc2(x)
        
        return x

# Training function
def train_model(model, train_loader, test_loader, epochs=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Lists to store metrics
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Evaluate on test set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy on test set after Epoch {epoch + 1}: {accuracy:.2f}%')
        
        train_losses.append(running_loss)
        test_accuracies.append(accuracy)
    
    return train_losses, test_accuracies

# Visualize results
def visualize_results(model, test_loader):
    model.eval()
    
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Get subset of images to display
    images = images.cpu()[:8]
    labels = labels[:8]
    predicted = predicted.cpu()[:8]
    
    # Plot the images and predictions
    fig = plt.figure(figsize=(12, 4))
    for i in range(8):
        ax = fig.add_subplot(1, 8, i + 1)
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f'P: {predicted[i]}\nT: {labels[i]}', 
                     color=('green' if predicted[i] == labels[i] else 'red'))
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

'''
# To run training on MNIST dataset:

# Load data
train_loader, test_loader = load_mnist()

# Create and train the model
model = SimpleCNN()
train_losses, test_accuracies = train_model(model, train_loader, test_loader, epochs=5)

# Visualize some predictions
visualize_results(model, test_loader)
'''

# =============================================================================
# PART 3: INTERMEDIATE CNN FOR CIFAR-10
# =============================================================================

'''
# CIFAR-10 is more challenging than MNIST:
# - 32x32 color images (3 channels)
# - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# - Requires more complex models
'''

# Data Loading and Preprocessing for CIFAR-10
def load_cifar10():
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Only normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=2
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=100, 
        shuffle=False, 
        num_workers=2
    )
    
    # Class names for visualization
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

# Define an intermediate CNN model for CIFAR-10
class IntermediateCNN(nn.Module):
    def __init__(self):
        super(IntermediateCNN, self).__init__()
        
        # First block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)  # More dropout for regularization
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Training CIFAR-10 model with learning rate scheduler
def train_cifar10_model(model, train_loader, test_loader, epochs=20):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.1, verbose=True
    )
    
    # Lists to store metrics
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Print statistics every 50 mini-batches
            if i % 50 == 49:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50:.3f}')
                running_loss = 0.0
        
        # Evaluate on test set after each epoch
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        print(f'Epoch {epoch + 1}: Validation Loss: {val_loss:.3f}, Accuracy: {accuracy:.2f}%')
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        train_losses.append(running_loss)
        test_accuracies.append(accuracy)
    
    return train_losses, test_accuracies

# Visualize CIFAR-10 predictions
def visualize_cifar10_results(model, test_loader, classes):
    model.eval()
    
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    images_subset = images[:16]
    labels_subset = labels[:16]
    
    with torch.no_grad():
        images_subset = images_subset.to(device)
        outputs = model(images_subset)
        _, predicted = torch.max(outputs, 1)
    
    # Plot the images and predictions
    images_subset = images_subset.cpu()
    predicted = predicted.cpu()
    
    fig = plt.figure(figsize=(12, 6))
    for i in range(16):
        ax = fig.add_subplot(2, 8, i + 1)
        # Convert tensor to numpy and transpose from [C, H, W] to [H, W, C]
        img = images_subset[i].numpy().transpose(1, 2, 0)
        
        # Unnormalize the image
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'P: {classes[predicted[i]]}\nT: {classes[labels_subset[i]]}', 
                     color=('green' if predicted[i] == labels_subset[i] else 'red'),
                     fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

'''
# To run training on CIFAR-10 dataset:

# Load data
train_loader, test_loader, classes = load_cifar10()

# Create and train the model
model = IntermediateCNN()
train_losses, test_accuracies = train_cifar10_model(model, train_loader, test_loader, epochs=10)

# Visualize some predictions
visualize_cifar10_results(model, test_loader, classes)
'''