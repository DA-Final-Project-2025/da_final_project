import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap

def generate_summary_stats(df):
    return df.describe().to_html(classes='table table-bordered')

def generate_plots(df):
    plots = []
    numeric_cols = df.select_dtypes(include='number').columns[:2]  # ví dụ lấy 2 cột đầu
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        filename = f'static/{col}_plot.png'
        plt.savefig(filename)
        plt.close()
        plots.append(filename)
    return plots

def train_and_predict(df):
    df = df.dropna()
    le = LabelEncoder()
    if 'target' not in df.columns:
        df['target'] = (df['price'] > df['price'].median()).astype(int)

    X = df.drop(columns=['target', 'price'], errors='ignore')
    X = X.select_dtypes(include=['number'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test[:50])

    # ✅ FIX: For binary classification
    shap.plots.beeswarm(shap_values[..., 0], show=False)

    shap_plot_path = "static/shap_plot.png"
    plt.savefig(shap_plot_path)
    plt.close()

    return {"Random Forest": round(acc * 100, 2)}, shap_plot_path