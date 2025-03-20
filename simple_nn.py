import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic dataset (simple linear function with noise)
X = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
y = 2 * X + 1 + 0.2 * np.random.randn(100, 1).astype(np.float32)

# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear0 = nn.Linear(1, 1)  # Single fully connected layer
   

    def forward(self, x):
        x = self.linear0(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer

# Training loop with loss tracking
loss_values = []
epochs = 20

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients to zero
    y_pred = model(X_tensor)  # Forward pass (prediction)
    loss = criterion(y_pred, y_tensor)  # Compute loss
    loss.backward()  # Backpropagation (compute gradients)
    optimizer.step()  # Update weights using gradients
    
    loss_values.append(loss.item())  # Store loss value
    
    if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_values, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Visualize results
with torch.no_grad():  # Disable gradient computation for evaluation
    y_pred = model(X_tensor)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label='Real Data')  # Plot original data points
plt.plot(X, y_pred.numpy(), color='red', label='Model Prediction')  # Plot model predictions
plt.legend()
plt.show()  # Display plot