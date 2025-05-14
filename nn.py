from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset , DataLoader
import torch.nn as nn 
import torch.optim as optim


df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)



class_count = df['quality'].value_counts().sort_index()

print(class_count)


mapping = {3: 0, 4: 0, 5: 0,
           6: 1,
           7: 2, 8: 2, 9: 2}

df['quality'] = df['quality'].map(mapping)

class_count = df['quality'].value_counts().sort_index()

print(class_count)

plt.figure(figsize=(10, 6))
plt.bar(class_count.index.astype(str), class_count.values, color='skyblue')
plt.title('Distribution of Wine Quality Classes')
plt.xlabel('Quality Class')
plt.ylabel('Count')
plt.savefig('wine_quality_distribution.png', dpi = 100)
plt.show() 

# Correlation Heatmap
num_df = df.drop(columns=['quality'])
corr = num_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.savefig('wine_quality_correlation_heatmap.png', dpi = 100)
plt.show()

# Split the data into features and target
X = df.drop(columns=['quality'])
y = df['quality']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



class MpalaourasNN(nn.Module):
    def __init__(self , features_number , number_classes):
        super().__init__()
        self.mpalaouras = nn.Sequential(
            nn.Linear(features_number,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,number_classes)
        )
    def forward(self, x):
        return self.mpalaouras(x)


X_train=torch.tensor(X_train , dtype = torch.float32)
X_test=torch.tensor(X_test, dtype = torch.float32)
y_train=torch.tensor(y_train.values , dtype = torch.long)
y_test=torch.tensor(y_test.values , dtype =torch.long) 
print(y_test.shape[0])
print(X_train.shape[0])
train = TensorDataset(X_train,y_train)
test = TensorDataset(X_test, y_test)

train = DataLoader(train , batch_size=128)
test = DataLoader(test ,batch_size=128)
print(len(train))

number_classes = len(y_train.unique())

model = MpalaourasNN(X_train.shape[1],number_classes)
loss_function = nn.CrossEntropyLoss()
optimazer = torch.optim.Adam(model.parameters(), lr = 0.01)

val_loss_mean, val_loss_accur =[], []
train_losses,train_acc =[], []
# Fix train and validation loops

val_loss_mean, val_accuracies = [], []
train_losses, train_accuracies = [], []

for epoch in range(1, 100):
    # Training phase
    model.train()
    total_loss_epoch = 0
    correct_train = 0
    total_train = 0
    
    for inputs, targets in train:
        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        
        # Accuracy calculation
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions == targets).sum().item()
        total_train += targets.size(0)
        
        # Save loss
        total_loss_epoch += loss.item()
        
        # Backward pass
        optimazer.zero_grad()
        loss.backward()
        optimazer.step()
    
    # Calculate and store training metrics
    avg_train_loss = total_loss_epoch / len(train)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs_val, targets_val in test:
            # Forward pass
            outputs_val = model(inputs_val)
            val_loss += loss_function(outputs_val, targets_val).item()
            
            # Accuracy calculation
            predictions_val = torch.argmax(outputs_val, dim=1)
            correct_val += (predictions_val == targets_val).sum().item()
            total_val += targets_val.size(0)
    
    # Calculate and store validation metrics
    avg_val_loss = val_loss / len(test)
    val_accuracy = correct_val / total_val
    val_loss_mean.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    # Print metrics
    print("-------------------------------------------------")
    print(f"--------------------EPOCH({epoch})-----------------------")
    print("-------------------------------------------------")
    print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


















        
