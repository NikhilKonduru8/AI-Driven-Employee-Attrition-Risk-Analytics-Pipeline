# AI-Driven Employee Attrition & Predictive Risk Analytics Pipeline

An end-to-end machine learning and deep learning pipeline designed to predict employee attrition risk using engineered corporate HR metrics. This repository demonstrates standard production workflows, including synthetic enterprise data generation, advanced non-linear feature engineering, structural regularization, and comprehensive visual evaluation.

## 🚀 Architectural Overview

Instead of applying basic models to generic datasets, this architecture models complex real-world corporate behaviors by generating interactive features and optimizing a deep neural network to predict risk probability. 

### Key Capabilities:
* Advanced Feature Engineering: Computes a custom non-linear BurnoutIndex combining overtime hours, employee satisfaction scores, and relative monthly compensation.
* Production Deep Learning Architecture: Built using TensorFlow/Keras with structural BatchNormalization and Dropout layers to ensure robust regularization and smooth gradient convergence.
* Adaptive Training Controls: Implements dynamic learning rate scheduling via ReduceLROnPlateau and automated execution bounds using EarlyStopping.
* Enterprise Evaluation Matrix: Features automated plotting for loss convergence curves, AUC tracking, ROC analysis, and detailed confusion matrix heatmaps.

---

## 📊 Feature & Data Schema

The data engine simulates a scaling organization with 2,500 unique structural personnel records across the following core variables:

- MonthlyIncome: Base monthly salary scaling from $3,500 to $18,000.
- OvertimeHours: Monthly accumulated overtime hours (0 to 60 hours).
- Age: Employee age tracking (21 to 65 years).
- YearsAtCompany: Total retention timeline within the enterprise.
- SatisfactionScore: Surveyed satisfaction matrix index graded from 1 (Low) to 5 (High).
- BurnoutIndex: Composite engineered metric calculated as (OvertimeHours * (6 - SatisfactionScore)) / (MonthlyIncome / 1000)

---

## 🛠️ Project Structure & Execution Pipeline

1. Preprocessing & Data Wrangling
* Synthesizes multivariate personnel distributions.
* Injects realistic statistical noise to simulate complex corporate dynamics.
* Utilizes a stratified split (80/20) to perfectly preserve target label ratios across both training and validation bounds.
* Scales feature matrices using a production-grade StandardScaler.

2. Deep Learning Neural Network Topology
* Input Layer: 6 Engineered Features
* Dense Layer: 64 Units (ReLU activation)
* Batch Normalization Layer
* Dropout Layer: Rate = 0.3
* Dense Layer: 32 Units (ReLU activation)
* Batch Normalization Layer
* Dropout Layer: Rate = 0.2
* Dense Layer: 16 Units (ReLU activation)
* Output Layer: 1 Unit (Sigmoid activation) -> Attrition Probability Score

3. Optimization Parameters
* Loss Function: Binary Cross-Entropy
* Optimizer: Adam Optimizer (Initial learning rate = 0.005)
* Class Weight Balancer: Computed dynamically to mitigate target class imbalances by balancing negative samples against positive samples.

---

## 📈 Installation & Setup

Prerequisites:
Ensure your environment meets the structural dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

Execution:
Simply pull down the repository and execute the driver file or copy it into a Google Colab instance:
python technical_test_pipeline.py

---

## 🏆 Performance & Visual Dashboard

Upon completion, the pipeline automatically processes the test partition and instantiates a 4-panel graphic analysis matrix detailing:
1. Confusion Matrix Heatmap: Outlines exact True Positives, False Positives, and classification accuracy.
2. ROC Curve: Computes model sensitivity versus specificity, reporting the final AUC Score.
3. Loss Convergence Chart: Tracks Training vs. Validation loss across training epochs to actively prove the absence of overfitting.
4. AUC Performance Track: Displays performance trajectory updates after every iteration cycle.
