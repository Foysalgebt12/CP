
import os
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score, recall_score,
    precision_score, accuracy_score, roc_curve
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_and_preprocess_data(filepath):
    data = pd.read_excel(filepath)
    numerical_features = ['Number of Prgenancy', 'Mother Age', 'Gestation Age of baby (weeks)',
                          'Number of Childerens', 'Birth Weight']
    categorical_features = ['Gender', 'Gravidity', 'Parity', 'Mother Health', 'Geastation Status',
                            'Term', 'Delivery Mode', 'BWC', 'Admit to NICU', 'Past Surgeries', 'G tube',
                            'Consanguine Marriage', 'Physcial Therapy']
    imputer = SimpleImputer(strategy='mean')
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    y = (data['CP/Non-CP'] == 'CP').astype(int)
    X = data.drop('CP/Non-CP', axis=1)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    X_preprocessed = preprocessor.fit_transform(X)
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
    f_values, p_values = f_classif(X_preprocessed, y)
    selected_indices = np.where(p_values < 0.05)[0]
    X_selected = X_preprocessed[:, selected_indices]
    selected_features = [feature_names[i] for i in selected_indices]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42), selected_features

def build_model(input_dim):
    model = Sequential([
        Flatten(input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_and_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def main():
    filepath = "CP.xlsx"
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    (X_train, X_test, y_train, y_test), feature_names = load_and_preprocess_data(filepath)
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                        batch_size=64, verbose=0)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\\n", classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    shap.summary_plot(shap_values.values, X_test_dense, feature_names=feature_names, show=False, plot_type="bar")
    plt.savefig(f"{output_dir}/shap_summary_plot.png", dpi=300); plt.close()

    # ROC + PR curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axs[0].set_title('ROC Curve'); axs[0].legend()
    axs[1].plot(recall, precision, color='darkorange', lw=2, label=f'PR AUC = {pr_auc:.4f}')
    axs[1].set_title('Precision-Recall Curve'); axs[1].legend()
    plot_and_save(fig, f"{output_dir}/roc_pr_combined_plot.png")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual_Non-CP", "Actual_CP"], columns=["Predicted_Non-CP", "Predicted_CP"])
    cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black', cbar=False)
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plot_and_save(fig, f"{output_dir}/confusion_matrix_plot.png")

    # Metric Summary
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"],
        "Value": [acc, prec, rec, f1, roc_auc, pr_auc]
    })
    metrics_df.to_csv(f"{output_dir}/model_metrics.csv", index=False)

    # Performance Plot
    fig = plt.figure(figsize=(6, 4))
    sns.barplot(x="Metric", y="Value", data=metrics_df, palette="magma")
    plt.ylim(0, 1); plt.title("Model Performance Metrics")
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.4f')
    plot_and_save(fig, f"{output_dir}/model_performance_plot.png")

if __name__ == "__main__":
    main()



