import joblib
import os, glob, io, base64
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def get_shap(app):
    print("Start handle SHAP")

    # Load file CSV mới nhất
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        return {'error': 'Không tìm thấy file dữ liệu'}

    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)

    if 'quantity_sold' not in df.columns:
        return {'error': "Thiếu cột 'quantity_sold' để làm nhãn"}

    # Encode và chuẩn bị dữ liệu
    # Biến mục tiêu
    target = 'quantity_sold'

    # Loại các cột không có ý nghĩa dự đoán
    drop_cols = ['Unnamed: 0', 'id', 'name', 'description', 'current_seller']

    # Mã hóa các cột dạng object (categorical)
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    for col in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col] = df_encoded[col].astype(int)

    # Lấy danh sách các đặc trưng đầu vào
    features = [col for col in df.columns if col not in drop_cols + [target]]

    df = df_encoded.dropna(subset=features + ['quantity_sold'])
    # Tách dữ liệu huấn luyện
    X = df_encoded[features]
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static'))
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(shap_values, os.path.join(save_dir, 'shap_values.pkl'))
    joblib.dump(X_test, os.path.join(save_dir, 'X_test_shap.pkl'))

    print("Done SHAP")
    return None


def plot_to_base64(func_plot):
    buf = io.BytesIO()
    plt.figure()  # Mở một figure mới
    func_plot()   # Gọi hàm vẽ
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64


# SHAP summary plot
def plot_summary(shap_values, X_test):
    shap.summary_plot(shap_values, X_test, show=False)


# SHAP dependence plot
def plot_specific_feature(feature, shap_values, X_test):
    shap.dependence_plot(feature, shap_values.values, X_test, show=False)


# SHAP waterfall plot (first instance)
def plot_waterfall(shap_values):
    shap.plots.waterfall(shap_values[0], show=False)


# Load SHAP values & plots
def plot_shap(feature):
    base_dir = os.path.dirname(__file__)
    static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

    shap_path = os.path.join(static_dir, 'shap_values.pkl')
    x_test_path = os.path.join(static_dir, 'X_test_shap.pkl')

    if not os.path.exists(shap_path) or not os.path.exists(x_test_path):
        return {
            'error': 'SHAP files not found'
        }

    shap_values = joblib.load(shap_path)
    X_test = joblib.load(x_test_path)

    return {
        'shap_summary_plot': plot_to_base64(lambda: plot_summary(shap_values, X_test)),
        'shap_specific_feature': plot_to_base64(lambda: plot_specific_feature(feature, shap_values, X_test)),
        'shap_waterfall_chart': plot_to_base64(lambda: plot_waterfall(shap_values))
    }

def plot_shap_specific_feature(feature):
    base_dir = os.path.dirname(__file__)
    static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

    shap_path = os.path.join(static_dir, 'shap_values.pkl')
    x_test_path = os.path.join(static_dir, 'X_test_shap.pkl')

    if not os.path.exists(shap_path) or not os.path.exists(x_test_path):
        return ''
    shap_values = joblib.load(shap_path)
    X_test = joblib.load(x_test_path)

    return plot_to_base64(lambda: plot_specific_feature(feature, shap_values, X_test))
