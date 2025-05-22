# Cerebral Palsy Risk Factor Analysis

## Overview

This repository presents a multi-method approach to identify and evaluate clinical and perinatal risk factors associated with pediatric cerebral palsy (CP). Two modeling pipelines were used: machine learning classifiers (Random Forest, XGBoost, SVM) and a feedforward neural network (FNN). Models were assessed for classification performance and interpretability.

## Objectives

To distinguish CP from non-CP cases by identifying high-impact risk factors through machine learning methods, supporting early diagnosis and intervention.

## Methods

### Machine Learning Models

* **Algorithms:** Random Forest, XGBoost, SVM
* **Preprocessing:** Imputation, encoding, scaling
* **Validation:** 5-fold cross-validation, hyperparameter tuning
* **Evaluation:** AUC-ROC, feature importance, Venn analysis

### Deep Learning (FNN)

* A feedforward neural network trained over 100 epochs
* Model metrics: Accuracy, Precision, Recall, F1 Score, PR-AUC, ROC-AUC
* Visualization: Confusion matrix, SHAP summary plot, ROC and PR curves

## Results

* All models showed high precision, recall, and PR-AUC
* SHAP and feature selection revealed key risk contributors
* Combinatorial interaction visualizations enabled risk stratification

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow shap imbalanced-learn
```

## Running

```bash
# Clone the repo
git clone https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis.git
cd Cerebral-Palsy-CP-Risk-Factors-Analysis

# Run ML models (RF/XGB/SVM/FNN)
python RF_XGB_SVM_CODE.py

```

## License

This project is licensed under the MIT License.

## Contributions

Contributions are welcome. Contact us to collaborate on pediatric neurology and risk modeling.

## Contact

* PI: [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa)
* Contributor: [Foysal Ahammad](mailto:foah48505@hbku.edu.qa)
