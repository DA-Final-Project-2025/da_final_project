from utils.explainable.shap import get_shap


def async_get_explainable(app):
    import threading
    import base64
    import os
    import glob

    def get_explainable():
        # 🧹 Xóa tất cả ảnh PNG cũ trong thư mục static
        for file_path in glob.glob("static/*.png"):
            try:
                os.remove(file_path)
                print(f"Đã xóa: {file_path}")
            except Exception as e:
                print(f"Lỗi khi xóa {file_path}: {e}")

        # 👇 Sinh ảnh mới
        tree_based = get_tree_based(app)
        lime = get_lime(app)
        result = {**tree_based, **lime}
        for key, b64_img in result.items():
            with open(f"static/{key}.png", "wb") as f:
                f.write(base64.b64decode(b64_img))

    def execute_shap():
        try:
            get_shap(app)
        except Exception as e:
            print(f"SHAP error: {e}")

    threading.Thread(target=get_explainable).start()
    threading.Thread(target=execute_shap).start()


def get_tree_based(app):
    import os, glob, io, base64
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print("Start handle tree-based model")
    # Load file CSV mới nhất
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        return {'error': 'Không tìm thấy file dữ liệu'}

    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)

    if 'quantity_sold' not in df.columns:
        return {'error': "Thiếu cột 'quantity_sold' để làm nhãn"}

    # Chọn các đặc trưng phù hợp
    categorical_features = ['brand', 'fulfillment_type', 'category']
    for col in categorical_features:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    numeric_features = [
        'original_price', 'price', 'review_count',
        'rating_average', 'favourite_count',
        'number_of_images', 'vnd_cashback',
        'has_video'
    ]
    df = df.dropna(subset=numeric_features + ['quantity_sold'])  # loại bỏ dòng thiếu

    X = df[numeric_features + categorical_features]
    y = df['quantity_sold']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Hồi quy với Decision Tree
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Feature Importance
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values()
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
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    plt.xlabel('Thực tế (quantity_sold)')
    plt.ylabel('Dự đoán')
    plt.title(f'Model Performance\nRMSE: {rmse:.2f} - MAE: {mae:.2f}')
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    perf_chart = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Decision Tree
    plt.figure(figsize=(10, 6))
    plot_tree(model, feature_names=X.columns, filled=True)
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

def get_lime(app):
    import os, glob, io, base64
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LinearRegression
    from lime.lime_tabular import LimeTabularExplainer

    print("Start handle LIME")

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        return {'error': 'Không tìm thấy file dữ liệu'}

    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)

    if 'quantity_sold' not in df.columns:
        return {'error': "Thiếu cột 'quantity_sold' để làm nhãn"}

    categorical_features = ['brand', 'fulfillment_type', 'category']
    for col in categorical_features:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    numeric_features = [
        'original_price', 'price', 'review_count',
        'rating_average', 'favourite_count',
        'number_of_images', 'vnd_cashback',
        'has_video'
    ]
    df = df.dropna(subset=numeric_features + ['quantity_sold'])

    X = df[numeric_features + categorical_features]
    y = df['quantity_sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện model đơn giản
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 1️⃣ LIME Local Explanation (cho sample đầu tiên của X_test)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X.columns.tolist(),
        mode='regression'
    )
    instance_idx = 0
    explanation = explainer.explain_instance(
        data_row=X_test.iloc[instance_idx].values,
        predict_fn=model.predict
    )
    fig_local = explanation.as_pyplot_figure()
    lime_local_base64 = fig_to_base64(fig_local)

    # 2️⃣ Feature Contribution (tổng trọng số mô hình)
    coef = model.coef_
    fig_feat = plt.figure()
    plt.barh(X.columns, coef, color='coral')
    plt.xlabel("Weight")
    plt.title("Feature Contribution (Linear Model)")
    feature_contribution_base64 = fig_to_base64(fig_feat)

    # 3️⃣ LIME Instance Explanation (cho sample #3 nếu có)
    instance_idx_3 = min(2, len(X_test) - 1)
    explanation_3 = explainer.explain_instance(
        data_row=X_test.iloc[instance_idx_3].values,
        predict_fn=model.predict
    )
    fig_instance = explanation_3.as_pyplot_figure()
    lime_instance_base64 = fig_to_base64(fig_instance)

    print("Done LIME")
    return {
        'lime_local_explanation': lime_local_base64,
        'feature_contribution_chart': feature_contribution_base64,
        'lime_instance_explanation': lime_instance_base64
    }