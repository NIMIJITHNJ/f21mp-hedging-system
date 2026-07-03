import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATA_DIR = "data/processed"
OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_REPORT_DIR = "outputs/reports"
OUTPUT_FIGURE_DIR = "outputs/figures"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)


def load_data():
    print("Loading processed data...")

    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv", index_col=0)
    X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv", index_col=0)
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv", index_col=0)

    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv", index_col=0).squeeze()
    y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv", index_col=0).squeeze()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv", index_col=0).squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, model_name, X_val, y_val, X_test, y_test):
    print(f"\nEvaluating {model_name}...")

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    val_report = classification_report(
        y_val,
        val_pred,
        target_names=["Low", "Medium", "High"],
        digits=4,
        zero_division=0
    )

    test_report = classification_report(
        y_test,
        test_pred,
        target_names=["Low", "Medium", "High"],
        digits=4,
        zero_division=0
    )

    print(f"\n{model_name} validation accuracy: {val_acc:.4f}")
    print(f"{model_name} test accuracy      : {test_acc:.4f}")

    print("\nValidation report:")
    print(val_report)

    print("\nTest report:")
    print(test_report)

    report_path = f"{OUTPUT_REPORT_DIR}/{model_name.lower().replace(' ', '_')}_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name} Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Test accuracy      : {test_acc:.4f}\n\n")
        f.write("Validation classification report:\n")
        f.write(val_report)
        f.write("\n\nTest classification report:\n")
        f.write(test_report)

    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} - Test Confusion Matrix")
    plt.tight_layout()

    fig_path = f"{OUTPUT_FIGURE_DIR}/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {fig_path}")

    return {
        "model": model_name,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    model_path = f"{OUTPUT_MODEL_DIR}/random_forest_model.pkl"
    joblib.dump(model, model_path)

    print(f"Saved Random Forest model: {model_path}")
    return model


def train_xgboost(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        print("\nXGBoost is not installed. Install it using:")
        print("pip install xgboost")
        return None

    print("\nTraining XGBoost...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    model_path = f"{OUTPUT_MODEL_DIR}/xgboost_model.pkl"
    joblib.dump(model, model_path)

    print(f"Saved XGBoost model: {model_path}")
    return model


def save_feature_importance(model, model_name, feature_names):
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    csv_path = f"{OUTPUT_REPORT_DIR}/{model_name.lower().replace(' ', '_')}_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"{model_name} - Top 15 Feature Importances")
    plt.tight_layout()

    fig_path = f"{OUTPUT_FIGURE_DIR}/{model_name.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved feature importance: {csv_path}")
    print(f"Saved feature importance plot: {fig_path}")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape  : {X_val.shape}")
    print(f"Test shape : {X_test.shape}")

    print("\nLabel distribution:")
    print("Train:")
    print(y_train.value_counts().sort_index())
    print("Val:")
    print(y_val.value_counts().sort_index())
    print("Test:")
    print(y_test.value_counts().sort_index())

    results = []

    rf_model = train_random_forest(X_train, y_train)
    results.append(
        evaluate_model(rf_model, "Random Forest", X_val, y_val, X_test, y_test)
    )
    save_feature_importance(rf_model, "Random Forest", X_train.columns)

    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        results.append(
            evaluate_model(xgb_model, "XGBoost", X_val, y_val, X_test, y_test)
        )
        save_feature_importance(xgb_model, "XGBoost", X_train.columns)

    results_df = pd.DataFrame(results)
    results_path = f"{OUTPUT_REPORT_DIR}/classical_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nClassical model comparison:")
    print(results_df)
    print(f"\nSaved comparison: {results_path}")


if __name__ == "__main__":
    main()