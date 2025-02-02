# 🚀 Water Quality Model

## 📌 Project Overview  
  
The goal is to **predict water potability** using a **neural network model built with TensorFlow/Keras**.

---

## 👤 **Team Members**
- **Pascal Mugisha**
- **Jules Gatete**
- **Geofrey Tumwesigye** 

---
  

**📄 Files in this branch:**
- 📌 `README.md`
- 📌 `the_model.ipynb`

---

## 🛠️ **The  Model Configuration**
| Parameter       | Choice |
|----------------|--------|
| **Regularization** | Dropout (0.2) |
| **Optimizer** | RMSprop |
| **Learning Rate** | 0.0005 |
| **Early Stopping** | Patience = 20 |
| **Class Weighting** | `{0:1, 1:1.5}` |

### 📊 **Final Model Performance**
| Metric        | Value |
|--------------|--------|
| **Test Accuracy** | **64.23%** |
| **F1 Score** | **0.5417** |
| **Recall** | **54.17%** |
| **Precision** | **54.17%** |

---

## 🏗 **How to Run the Model**
1️⃣ **Clone the Repository**  
```bash
git clone <https://github.com/g-tumwesigye/water_quality_model_assignment/tree/geofrey_branch>
cd water_quality_model_assignment
git checkout geofrey_branch

2️⃣ **Install Dependencies**  
```bash
pip install tensorflow pandas numpy scikit-learn

3️⃣ **Run Jupyter Notebook**
```bash
jupyter notebook the_model.ipynb
```bash

4️⃣ **Execute all cells** to train and evaluate the model.

## 📌 **Project Structure**
```bash
water_quality_model_assignment/
│── README.md  
│── the_model.ipynb  # Model implementation
│── (Other team members' branches exist separately)
```bash


## 📝 **Actions**
✅ Implemented **RMSprop optimizer**  
✅ Tuned **Dropout & Class Weights** for better balance between recall and precision.  
✅ Applied **Early Stopping (patience=20)** for optimal training.  
✅ Achieved **64.23% Accuracy** with improved recall & precision.



