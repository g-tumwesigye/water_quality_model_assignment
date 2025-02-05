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

## Repository structure
│ title: **water_quality_model_assignment**:
│ link: [water_quality_model_assignment](https://github.com/g-tumwesigye/water_quality_model_assignment)
│ Branches:**4** 

### Branches
```
├── main/
│   ├── [Water_Qaulity_Analysis_PLD4.ipynb](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/main/Water_Qaulity_Analysis_PLD4.ipynb)
│   ├── [README.md](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/main/README.md)


├── Jules_branch/
│   ├── [Water_Quality_Model.ipynb](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/Jules_branch/Water_Quality_Model.ipynb)
│   ├── [README.md](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/Jules_branch/README.md)


├── MPascal/
│   ├── [water_quality_model_assignment](https://github.com/g-tumwesigye/water_quality_model_assignment/tree/MPascal)
│   ├── [README.md](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/MPascal/README.md)

├── geofrey_branch/
│   ├── [the_model.ipynb](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/geofrey_branch/the_model.ipynb)
│   ├── [README.md](https://github.com/g-tumwesigye/water_quality_model_assignment/blob/geofrey_branch/README.md)


```
## Data Preprocessing
- Load dataset and handle missing values
- Normalize features for better training performance
- Splitted data 

## Model Training
Each group member defines a unique model with distinct **learning rate, dropout rate, and early stopping criteria**:
- **Jules Gatete**: Adam optimizer with L2 regularization
- **Pascal Mugisha**: RMSprop optimizer with Dropout, and L2_regularization.
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
| Jules Gatete | L2 (0.0002) | Adam | Yes(5) | 0.1 | | 67% | 0.583 | 0.593 | 0.572|
| Pascal Mugisha |L2 (0.01) | SGD | Yes (10) | 0.3 | 66.2% | 0.301 | 0.187 | 0.778 |
| Geofrey Tumwesigye | L2 (0.007) | RMSprop |Yes (25) |0.2 | 66.87% | 0.583 |0.594 | 0.572 |

## Video Submission
A video walkthrough explaining the models, optimizations, and results is included. 
[Video Link - To be added]

## Contribution and Documentation
- Each member contributed by defining a model, training, and evaluating it.
- Commits reflect meaningful contributions and progress.
- The final report includes methodology, findings, and key insights.

## Conclusion
This project successfully applies neural networks to classify water potability, showcasing the impact of different optimization strategies. The comparison provides insights into model performance and areas for improvement.
