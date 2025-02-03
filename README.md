# Water Quality Model Assignment.
## Water Quality Classification using Neural Networks

## Group Members
- **Jules Gatete**
- **Pascal Mugisha**
- **Geofrey Tumwesigye**

## Project Overview
This project implements a neural network-based classification model to determine water potability using the provided dataset. Each group member applies unique optimization techniques, ensuring diverse approaches and meaningful comparisons.

## Dataset
We use the **Water Quality and Potability Dataset** available at Kaggle:
[Water Quality Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability?select=water_potability.csv)

## Repository Structure
```
├── data/
│   ├── water_potability.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── model_jules.py
│   ├── model_pascal.py
│   ├── model_geofrey.py
├── README.md
```

## Installation
To set up the project environment, run:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
- Load dataset and handle missing values
- Normalize features for better training performance
- Split data into 70% training, 15% validation, and 15% testing

## Model Training
Each group member defines a unique model with distinct **learning rate, dropout rate, and early stopping criteria**:
- **Jules Gatete**: Adam optimizer with L2 regularization
- **Pascal Mugisha**: RMSprop optimizer with Dropout
- **Geofrey Tumwesigye**: SGD optimizer with Momentum

Each model is trained and evaluated separately.

## Model Evaluation
Each member compares their model's performance based on:
- **Accuracy**
- **F1 Score**
- **Precision & Recall**
- **Loss Analysis**

A summary table documents each member's results:

| Engineer | Regularizer | Optimizer | Early Stopping | Dropout Rate | Accuracy | F1 Score | Recall | Precision |
|----------|------------|-----------|---------------|-------------|---------|---------|--------|----------|
| Jules Gatete | L2 | Adam | Yes | 0.2 | X.XX | X.XX | X.XX | X.XX |
| Pascal Mugisha | None | RMSprop | No | 0.3 | X.XX | X.XX | X.XX | X.XX |
| Geofrey Tumwesigye | None | SGD + Momentum | Yes | 0.1 | X.XX | X.XX | X.XX | X.XX |

## Video Submission
A video walkthrough explaining the models, optimizations, and results is included. 
[Video Link - To be added]

## Contribution and Documentation
- Each member contributed by defining a model, training, and evaluating it.
- Commits reflect meaningful contributions and progress.
- The final report includes methodology, findings, and key insights.

## Conclusion
This project successfully applies neural networks to classify water potability, showcasing the impact of different optimization strategies. The comparison provides insights into model performance and areas for improvement.





# Water Quality Classification using Neural Networks

## Group Members
- **Jules Gatete**
- **Pascal Mugisha**
- **Geofrey Tumwesigye**

## Project Overview
This project implements a neural network-based classification model to determine water potability using the provided dataset. Each group member applies unique optimization techniques, ensuring diverse approaches and meaningful comparisons.

## Dataset
We use the **Water Quality and Potability Dataset** available at Kaggle:
[Water Quality Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability?select=water_potability.csv)

## Repository Structure
```
├── data/
│   ├── water_potability.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── model_jules.py
│   ├── model_pascal.py
│   ├── model_geofrey.py
├── README.md
```

## Installation
To set up the project environment, run:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
- Load dataset and handle missing values
- Normalize features for better training performance
- Split data into 70% training, 15% validation, and 15% testing

## Model Training
Each group member defines a unique model with distinct **learning rate, dropout rate, and early stopping criteria**:
- **Jules Gatete**: Adam optimizer with L2 regularization
- **Pascal Mugisha**: RMSprop optimizer with Dropout
- **Geofrey Tumwesigye**: SGD optimizer with Momentum

Each model is trained and evaluated separately.

## Model Evaluation
Each member compares their model's performance based on:
- **Accuracy**
- **F1 Score**
- **Precision & Recall**
- **Loss Analysis**

A summary table documents each member's results:

| Engineer | Regularizer | Optimizer | Early Stopping | Dropout Rate | Accuracy | F1 Score | Recall | Precision |
|----------|------------|-----------|---------------|-------------|---------|---------|--------|----------|
| Jules Gatete | L2 | Adam | Yes | 0.2 | X.XX | X.XX | X.XX | X.XX |
| Pascal Mugisha | None | RMSprop | No | 0.3 | X.XX | X.XX | X.XX | X.XX |
| Geofrey Tumwesigye | None | SGD + Momentum | Yes | 0.1 | X.XX | X.XX | X.XX | X.XX |

## Video Submission
A video walkthrough explaining the models, optimizations, and results is included. 
[Video Link - To be added]

## Contribution and Documentation
- Each member contributed by defining a model, training, and evaluating it.
- Commits reflect meaningful contributions and progress.
- The final report includes methodology, findings, and key insights.

## Conclusion
This project successfully applies neural networks to classify water potability, showcasing the impact of different optimization strategies. The comparison provides insights into model performance and areas for improvement.



# Water Quality Classification using Neural Networks

## Group Members
- **Jules Gatete**
- **Pascal Mugisha**
- **Geofrey Tumwesigye**

## Project Overview
This project implements a neural network-based classification model to determine water potability using the provided dataset. Each group member applies unique optimization techniques, ensuring diverse approaches and meaningful comparisons.

## Dataset
We use the **Water Quality and Potability Dataset** available at Kaggle:
[Water Quality Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability?select=water_potability.csv)

## Repository Structure
```
├── data/
│   ├── water_potability.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_jules.py
│   ├── model_pascal.py
│   ├── model_geofrey.py
│   ├── evaluate_models.py
├── README.md
```

## Installation
To set up the project environment, run:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)
    X = df.drop(columns=['Potability'])
    y = df['Potability']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Model Implementations
Each member trains a unique model with different optimization strategies:

### **Geofrey Tumwesigye** - Adam Optimizer with L2 Regularization
```python
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_model_jules(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

### **Pascal Mugisha** - RMSprop Optimizer with Dropout
```python

def build_model_pascal(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

### **Jules Gatete** - SGD with Momentum
```python

def build_model_geofrey(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.SGD(momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

## Training & Evaluation
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }
```

## Final Results
| Engineer | Regularizer | Optimizer | Dropout Rate | Accuracy | F1 Score | Recall | Precision |
|----------|------------|-----------|-------------|---------|---------|--------|----------|
| Jules Gatete | L2 | Adam | 0.2 | X.XX | X.XX | X.XX | X.XX |
| Pascal Mugisha | None | RMSprop | 0.3 | X.XX | X.XX | X.XX | X.XX |
| Geofrey Tumwesigye | None | SGD + Momentum | 0.1 | X.XX | X.XX | X.XX | X.XX |

## Video Submission
A video walkthrough explaining the models, optimizations, and results is included. 
[Video Link - To be added]

## Contribution and Documentation
- Each member contributed by defining a model, training, and evaluating it.
- Commits reflect meaningful contributions and progress.
- The final report includes methodology, findings, and key insights.

## Conclusion
This project successfully applies neural networks to classify water potability, showcasing the impact of different optimization strategies. The comparison provides insights into model performance and areas for improvement.

