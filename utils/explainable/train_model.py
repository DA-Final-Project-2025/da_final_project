import glob
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_model(app):
    print("Start train model")
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static'))
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model,  os.path.join(save_dir, 'rf_model.pkl'))
    joblib.dump(X_train, os.path.join(save_dir, 'x_train.pkl'))
    joblib.dump(X_test, os.path.join(save_dir, 'x_test.pkl'))
    joblib.dump(y_test, os.path.join(save_dir, 'y_test.pkl'))
    joblib.dump(features, os.path.join(save_dir, 'features.pkl'))
    return {
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'features': features
    }

def get_trained_model():
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        path = os.path.join(static_dir, 'rf_model.pkl')
        return joblib.load(path)
    except Exception as e:
        return {'error': f'Loading rf_model failed: {str(e)}'}

def get_x_train():
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        path = os.path.join(static_dir, 'x_train.pkl')
        return joblib.load(path)
    except Exception as e:
        return {'error': f'Loading x_train failed: {str(e)}'}

def get_x_test():
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        path = os.path.join(static_dir, 'x_test.pkl')
        return joblib.load(path)
    except Exception as e:
        return {'error': f'Loading x_test failed: {str(e)}'}

def get_y_test():
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        path = os.path.join(static_dir, 'y_test.pkl')
        return joblib.load(path)
    except Exception as e:
        return {'error': f'Loading y_test failed: {str(e)}'}

def get_features():
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        path = os.path.join(static_dir, 'features.pkl')
        return joblib.load(path)
    except Exception as e:
        return {'error': f'Loading features failed: {str(e)}'}