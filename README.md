# 🚰 Water Quality Model

## 📌 Project Overview
This project aims to classify water samples as **potable (safe to drink) or non-potable** based on various chemical properties. A **Neural Network** built with **TensorFlow/Keras** is used for classification, with extensive hyperparameter tuning to achieve the best recall-precision balance.

## 📂 Dataset
The dataset, `water_potability.csv`, consists of **3,276 water samples** with **9 features**:
- **pH**
- **Hardness**
- **Solids**
- **Chloramines**
- **Sulfate**
- **Conductivity**
- **Organic Carbon**
- **Trihalomethanes**
- **Turbidity**

### **🛠 Data Preprocessing**
- **Handled missing values** by filling them with column means.
- **Standardized features** using `StandardScaler()` to ensure equal weight for all variables.
- **Split dataset** into **70% training, 15% validation, and 15% testing**.

---

## 🏗️ Model Architecture
A **Deep Neural Network (DNN)** was implemented with the following layers:
- **Dense (64 neurons, ReLU, L2 Regularization = 0.007)**
- **Dropout (0.2)**
- **Dense (32 neurons, ReLU, L2 Regularization = 0.007)**
- **Dropout (0.2)**
- **Dense (16 neurons, ReLU, L2 Regularization = 0.007)**
- **Dropout (0.2)**
- **Dense (1 neuron, Sigmoid activation for binary classification)**

### **🛠 Optimization Techniques**
✅ **Optimizer:** `RMSprop (LR = 0.002)`  
✅ **Loss Function:** `Binary Crossentropy`  
✅ **Regularization:** `L2 (0.007)`  
✅ **Dropout:** `0.2` to prevent overfitting  
✅ **Class Weighting:** `{0:1, 1:2}` (to balance imbalanced data)  
✅ **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor = 0.7, patience = 7)  
✅ **Early Stopping:** `patience = 25` (restores best weights)

---

## 📊 Model Performance (After Fine-Tuning)
| Metric       | Value  |
|-------------|--------|
| **Test Accuracy**  | **66.46%** ✅ |
| **F1 Score**       | **0.5714** ✅ |
| **Recall**         | **0.5729** ✅ |
| **Precision**      | **0.5699** ✅ |

📌 **Final model achieved a balanced recall-precision trade-off**, ensuring both potable and non-potable classifications are well-represented.

---

## 📈 Visualizations
✅ **Feature Distributions**  
✅ **Class Distribution Plot (Potable vs. Non-Potable)**  
✅ **Feature Correlation Heatmap**  
✅ **Training vs. Validation Loss Graph**  
✅ **Training vs. Validation Accuracy Graph**  
✅ **Confusion Matrix for Classification Evaluation**  

---

## 📝 How to Run
1️⃣ Install dependencies:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
   python train_model.py
   python evaluate_model.py
```bash


## 👤 Author: **Geofrey Tumwesigye** 
