from flask import Flask, request, render_template, redirect, url_for, flash
import os
import time
import pandas as pd
from utils.analysis import describe_numeric, generate_boxplot_svgs, generate_feature_distribution_svgs
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NUMERIC_COLS = [
    "original_price", "price", "review_count", "rating_average",
    "favourite_count", "number_of_images", "vnd_cashback", "quantity_sold"
]

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Trang chủ: upload file
@app.route('/')
def index():
    return render_template('index.html')


# Xử lý upload và phân tích, chuyển về dashboard
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('Không tìm thấy file!')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('Chưa chọn file!')
        return redirect(url_for('index'))
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f'Lỗi đọc file: {e}')
        return redirect(url_for('index'))
    # Lưu tên file mới nhất vào session nếu muốn
    return redirect(url_for('dashboard'))

# Thống kê mô tả
@app.route('/stats')
def stats():
    import glob
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        flash('Chưa có file dữ liệu, vui lòng upload trước!')
        return redirect(url_for('index'))
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    nrows = len(df)
    stats_data = describe_numeric(df, NUMERIC_COLS)
    return render_template('stats.html', filename=os.path.basename(filepath), nrows=nrows, stats=stats_data, columns=NUMERIC_COLS)

# Boxplot
@app.route('/boxplot')
def boxplot():
    import glob
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        flash('Chưa có file dữ liệu, vui lòng upload trước!')
        return redirect(url_for('index'))
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    nrows = len(df)
    boxplots = generate_boxplot_svgs(df, NUMERIC_COLS)
    return render_template('boxplot.html', filename=os.path.basename(filepath), nrows=nrows, boxplots=boxplots, columns=NUMERIC_COLS)

@app.route('/feature_types')
def feature_types():
    import glob
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        flash('Chưa có file dữ liệu, vui lòng upload trước!')
        return redirect(url_for('index'))
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    # Bỏ cột Unnamed: 0 nếu có
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    # Không plot biểu đồ cho id
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    num_numeric = len(numeric_cols)
    num_categorical = len(categorical_cols)
    # Chỉ truyền các cột cần plot vào hàm vẽ
    plot_cols = numeric_cols + categorical_cols
    feature_svgs = generate_feature_distribution_svgs(df[plot_cols])
    return render_template(
        'feature_types.html',
        filename=os.path.basename(filepath),
        num_numeric=num_numeric,
        num_categorical=num_categorical,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        feature_svgs=feature_svgs
    )

# Dashboard (nếu cần)
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Dataset info
@app.route('/dataset_info')
def dataset_info():
    return render_template('dataset_info.html')

if __name__ == '__main__':
    app.run(debug=True)