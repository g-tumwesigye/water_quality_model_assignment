## **Water Quality Classification - Jules Model**

# ** Project Overview**
This project focuses on classifying water potability using machine learning techniques. We optimized a neural network model to balance recall and precision, ensuring improved generalization and stability. 

---

## ** Model Insights**
- **Class Weights** → `Class 1 (Potable)` weight increased to `1.1x` to improve recall.
- **Learning Rate** → `0.0002` for smoother convergence and reduced overfitting.
- **Dropout Rate** → Lowered to `0.1` to **retain more features** without overfitting.
- **Threshold Selection** →  **0.5** for balanced recall-precision trade-off.

### ** Performance Evaluation**
| **Metric**  | **Final Model** |
|------------|---------------|
| **Training Accuracy** | 0.748 |
| **Testing Accuracy**  | 0.640 |
| **F1-Score (Class 1)** | 0.574 |
| **Recall (Class 1)** | 0.619 |
| **Precision (Class 1)** | 0.535 |

---

## ** Challenges & Key Learnings**
- **Class Imbalance:** Required **class weight tuning** to improve recall without sacrificing precision.
- **Over-Regularization:** Dropout was **too high initially**, leading to excessive information loss.
- **Threshold Optimization:** Tried **multiple thresholds (0.4, 0.5, 0.6)** to find the best trade-off between recall & precision.
- **Hyperparameter Tuning:** Adjusted **learning rate, dropout, and batch size** for optimal convergence.

---

## ** Training Summary Table**
| **Train Instance** | **Engineer Name** | **Regularizer** | **Optimizer** | **Early Stopping** | **Dropout Rate** | **Accuracy** | **F1-Score** | **Recall** | **Precision** |
|-------------------|----------------|-------------|------------|---------------|--------------|-----------|------------|--------|------------|
| Final Model | [Jules Gatete] | L2 (0.0001) | Adam (lr=0.0002) | Patience=5 | 0.1 | 0.640 | 0.574 | 0.619 | 0.535 |

---

## ** Conclusion & Next Steps**
The final model **improves recall** for potable water **without sacrificing too much precision**. Future improvements could involve:
- **Feature Engineering** → Exploring interactions between water quality parameters.
- **Alternative Models** → Trying Random Forest or XGBoost for enhanced interpretability.
- **Active Learning** → Iteratively refining model based on real-world misclassifications.
