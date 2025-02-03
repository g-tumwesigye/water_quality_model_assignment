# water_quality_model_assignment

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
