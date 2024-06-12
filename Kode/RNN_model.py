#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Generating synthetic dataset
np.random.seed(42)
n_patients = 8000
gender = np.random.choice(['Male', 'Female'], size=n_patients)
age = np.random.randint(18, 90, size=n_patients)
diagnosis = np.random.choice(['Diagnosis A', 'Diagnosis B', 'Diagnosis C'], size=n_patients)
alcohol = np.random.randint(0, 20, size=n_patients)  # drinks per week
smoking = np.random.choice(['Non-smoker', 'Former smoker', 'Current smoker'], size=n_patients)
blood_pressure = np.random.randint(90, 180, size=n_patients)
mean_arterial_pressure = np.random.randint(70, 110, size=n_patients)
pulse = np.random.randint(50, 120, size=n_patients)
oxygen_saturation = np.random.randint(85, 100, size=n_patients)
weight = np.random.randint(50, 150, size=n_patients)  # in kilograms
ventilator_duration = np.random.randint(0, 48, size=n_patients)  # hours on ventilator
icu_duration = np.random.randint(1, 20, size=n_patients)  # days in ICU
death_within_year = np.random.choice([0, 1], size=n_patients, p=[0.9, 0.1])  # 10% mortality rate

data = pd.DataFrame({
    'Gender': gender,
    'Age': age,
    'Diagnosis': diagnosis,
    'Alcohol': alcohol,
    'Smoking': smoking,
    'Blood Pressure': blood_pressure,
    'Mean Arterial Pressure': mean_arterial_pressure,
    'Pulse': pulse,
    'Oxygen Saturation': oxygen_saturation,
    'Weight': weight,
    'Ventilator Duration': ventilator_duration,
    'ICU Duration': icu_duration,
    'Death Within Year': death_within_year
})

# Encode categorical variables
categorical_features = ['Gender', 'Diagnosis', 'Smoking']
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Split the dataset
X = data_encoded.drop(columns='Death Within Year')
y = data_encoded['Death Within Year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data to fit RNN input
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Adding sequence dimension
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Adding sequence dimension
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 30
learning_rate = 0.001

# Model, loss function, optimizer
model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_pred.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())

y_pred = np.array(y_pred)
y_pred_class = (y_pred > 0.5).astype(int)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred_class))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - RNN')
plt.show()

# %%
