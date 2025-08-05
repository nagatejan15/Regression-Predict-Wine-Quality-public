import numpy as np
import pandas as pd
import joblib
import os
import json
import torch
import torch.nn as nn
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
        nn.Linear(hidden_size2, 1) 
    )
    return model

def predict_single_example( prediction_x):
    
    try:
        #  Load Artifacts 
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        scaler = joblib.load(scaler_path)
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        #  Recreate Model using the functional approach 
        model = create_regression_model(
            input_size=config['input_size'],
            hidden_size1=config['hidden_layers']['layer1'],
            hidden_size2=config['hidden_layers']['layer2']
        )
        model.load_state_dict(model_state_dict)
        model.eval()

        #  Preprocess and Predict 
        example_np = np.array(prediction_x).reshape(1, -1)
        example_scaled = scaler.transform(example_np)
        example_tensor = torch.tensor(example_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(example_tensor)
            
        return prediction.item()

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Make sure the model directory is correct.")
        print(e)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return None
    

df = pd.read_csv(DATASET_PATH, sep=';')
x = df.drop('quality', axis=1)
y = df['quality']
x = x.iloc[30].values
y = y.iloc[30]


predicted_value = predict_single_example( x)

if predicted_value is not None:
    print(f"Input example data (raw): {x.tolist()}")
    print(f"True quality value: {y:.4f}")
    print(f"Predicted quality value: {predicted_value:.4f}")