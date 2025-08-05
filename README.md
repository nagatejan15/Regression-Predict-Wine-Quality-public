# **Wine Quality Prediction**

This project trains a neural network regression model to predict the quality of red wine based on its physicochemical properties. The model is built using PyTorch, and the project includes scripts for both training the model and making predictions.

## **Project Overview**

The goal of this project is to predict the quality of red wine, which is a score between 0 and 10\. A neural network model is trained on a dataset of red wine samples, learning the relationship between the chemical features of the wine and its quality rating.

The project consists of two main Python scripts:

* training.py: This script handles the loading and preprocessing of the dataset, training the regression model, evaluating its performance, and saving the trained model and other necessary artifacts.  
* prediction.py: This script loads the saved model and artifacts to make predictions on new, unseen wine data.

## **Project Structure**

* **dataset/**: Contains the raw data used for training and evaluation.  
* **model/**: This directory stores all the artifacts produced during model training.  
  * config.json: Stores the model's architecture and training hyperparameters.  
  * model\_checkpoint.pth: The saved state of the trained PyTorch model.  
  * performance.json: Contains the model's final training loss and validation metrics.  
  * scaler.pkl: The saved StandardScaler object used to normalize the input features.  
* **prediction.py**: Script to run inference using the trained model.  
* **requirements.txt**: A list of all the Python packages required to run the project.  
* **training.py**: Script to train the model from scratch.

## **Installation**

1. **Clone the repository:**
   ``` 
   git clone \<repository-url\>  
   cd Regression-Predict-Wine-Quality-public
2. **Create a virtual environment (recommended):**  
   ```python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. **Install the dependencies:**  
   ```
   pip install -r requirements.txt
   ```

4. **Create a .env file** in the root directory and add the following paths:  
   MODELS\_DIR=model  
   DATASET\_PATH=dataset/winequality-red.csv

## **Usage**

### **Training**

To train the model, run the training.py script from the root of the project directory:

```
python training.py
```

The script will perform the following steps:

1. Load the winequality-red.csv dataset.  
2. Scale the features using StandardScaler.  
3. Split the data into training and validation sets.  
4. Build and train the neural network model.  
5. Save the trained model, scaler, configuration, and performance metrics to the model/ directory.

### **Prediction**

To make a prediction on a new data sample, you can use the prediction.py script. The script is currently set up to predict the quality of a sample from the dataset. You can modify the script to input your own data.

To run the prediction script:

```
python prediction.py
```

The script will load the saved model and artifacts, preprocess the input data, and output the predicted wine quality.

## **Model Performance**

The performance of the trained model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the validation set. The final metrics are saved in model/performance.json.

* **Final Training Loss**: 0.177  
* **Validation MSE**: 0.399  
* **Validation RMSE**: 0.632

## **Dataset**

The dataset used for this project is the **Red Wine Quality** dataset, which can be found in the dataset/ directory. It contains 1,599 instances with 11 input features representing chemical properties of the wine and one output variable, quality, which is a score from 0 to 10\.

**Features:**

1. fixed acidity  
2. volatile acidity  
3. citric acid  
4. residual sugar  
5. chlorides  
6. free sulfur dioxide  
7. total sulfur dioxide  
8. density  
9. pH  
10. sulphates  
11. alcohol

**Target Variable:**

* quality (score between 0 and 10)