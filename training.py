import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

MODELS_DIR = os.getenv("MODELS_DIR")
DATASET_PATH = os.getenv("DATASET_PATH")
model_path = os.path.join(MODELS_DIR, "model_checkpoint.pth")
config_path = os.path.join(MODELS_DIR, 'config.json')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

def create_regression_model(input_size, hidden_size1=64, hidden_size2=32):
    
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size1),
        nn.ReLU(),
        nn.Linear(hidden_size1, hidden_size2),
        nn.ReLU(),
        nn.Linear(hidden_size2, 1) # Output layer for a single regression value
    )
    return model

# --- 3. Training and Evaluation Function ---
def train_evaluate_and_save_model(model, train_loader, val_loader, scaler, config, base_save_dir=MODELS_DIR):

    
    os.makedirs(MODELS_DIR, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Training Loss: {avg_train_loss:.4f}')

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_val_loss += loss.item()
    
    avg_val_mse = total_val_loss / len(val_loader)
    print(f"\nTraining complete. Final Validation MSE: {avg_val_mse:.4f}")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    performance = {
        'final_training_loss': avg_train_loss,
        'validation_mse': avg_val_mse,
        'validation_rmse': np.sqrt(avg_val_mse)
    }
    with open(os.path.join(MODELS_DIR, 'performance.json'), 'w') as f:
        json.dump(performance, f, indent=4)

    print(f"All model artifacts saved successfully in {MODELS_DIR}")

if __name__ == '__main__':

    df = pd.read_csv(DATASET_PATH, sep=';')
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    input_size = X_train.shape[1]
    model_config = {
        'input_size': input_size,
        'hidden_layers': {'layer1': 64, 'layer2': 32},
        'learning_rate': 0.001,
        'epochs': 200,
        'batch_size': 32
    }
    model = create_regression_model(
        input_size=model_config['input_size'],
        hidden_size1=model_config['hidden_layers']['layer1'],
        hidden_size2=model_config['hidden_layers']['layer2']
    )
    
    train_evaluate_and_save_model(model, train_loader, val_loader, scaler, model_config)
    