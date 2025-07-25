from pathlib import Path

from utils.explainable.shap_explainer import init_shap
from utils.explainable.train_model import train_model


def async_get_explainable(app):
    import threading
    import base64
    import os
    import glob

    def execute_train_model():
        try:
            current_dir = Path(__file__).parent
            static_dir = current_dir.parent.parent / "static"
            for file_path in glob.glob(str(static_dir / "*.png")):
                try:
                    os.remove(file_path)
                    print(f"Đã xóa: {file_path}")
                except Exception as e:
                    print(f"Lỗi khi xóa {file_path}: {e}")
            for file_path in glob.glob(str(static_dir / "*.pkl")):
                try:
                    os.remove(file_path)
                    print(f"Đã xóa: {file_path}")
                except Exception as e:
                    print(f"Lỗi khi xóa {file_path}: {e}")

            res = train_model(app)
            init_shap(res['model'], res['X_test'])
            tree_based = get_tree_based(res['model'], res['X_test'], res['y_test'])
            for key, b64_img in tree_based.items():
                with open(f"static/{key}.png", "wb") as f:
                    f.write(base64.b64decode(b64_img))
        except Exception as e:
            print(f"SHAP error: {e}")

    threading.Thread(target=execute_train_model).start()


def get_tree_based(model, X_test, y_test):
    import os, glob, io, base64
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import plot_tree
    from sklearn.metrics import r2_score
    print("Start tree-based model")

    cols = X_test.columns
    # Feature Importance
    fi = pd.Series(model.feature_importances_, index=cols).sort_values()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=fi.values, y=fi.index, palette='viridis')
    plt.title('Feature Importance')
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    feature_chart = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Model performance: scatter thực tế vs dự đoán
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 4))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, label="Predictions")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual (R² = {r2_score(y_test, y_pred):.2f})")
    plt.legend()
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    perf_chart = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Decision Tree
    plt.figure(figsize=(20, 12))
    plot_tree(model.estimators_[0], feature_names=cols, filled=True, max_depth=3)
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    tree_chart = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    print("Done tree-based model")
    return {
        'feature_chart': feature_chart,
        'model_performance': perf_chart,
        'tree_structure': tree_chart
    }
