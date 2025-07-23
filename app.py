from flask import Flask, request, render_template, redirect, url_for, flash
import os
import time
import pandas as pd
from utils.analysis import describe_numeric, generate_boxplot_svgs, generate_feature_distribution_svgs
from utils.correlation import generate_correlation_plots, generate_scatter_plot, generate_density_plot, generate_violin_plot
from utils.explainable.async_get_explainable import async_get_explainable
from utils.explainable.shap import plot_shap, plot_shap_specific_feature

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
        async_get_explainable(app)
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


@app.route('/explainable')
def explainable():
    import glob
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        flash('Chưa có file dữ liệu, vui lòng upload trước!')
        return redirect(url_for('index'))
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)

    # Biến mục tiêu
    target = 'quantity_sold'

    # Loại các cột không có ý nghĩa dự đoán
    drop_cols = ['Unnamed: 0', 'id', 'name', 'description', 'current_seller']

    features = [col for col in df.columns if col not in drop_cols + [target]]
    return render_template('explainable/index.html', features=features)

@app.route('/shap/<feature>')
def shap(feature):
    return plot_shap(feature)

@app.route('/shap/specific/<feature>')
def shap_specific_feature(feature):
    return plot_shap_specific_feature(feature)

# Dashboard (nếu cần)
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Dataset info
@app.route('/dataset_info')
def dataset_info():
    return render_template('dataset_info.html')


# Route cho giao diện correlation plots
@app.route('/correlation')
def correlation():
    import glob
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        flash('Chưa có file dữ liệu, vui lòng upload trước!')
        return redirect(url_for('index'))
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    numeric_cols = [
        "original_price", "price", "review_count", "rating_average",
        "favourite_count", "number_of_images", "vnd_cashback", "quantity_sold"
    ]
    categorical_cols = [
        "name", "fulfillment_type", "brand", "pay_later", "current_seller",
        "has_video", "category"
    ]
    # Lấy giá trị cột từ từng dropdown
    scatter_x = numeric_cols[0] if numeric_cols else None
    scatter_y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)
    density_x = numeric_cols[0] if numeric_cols else None
    density_y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)
    violin_cat = categorical_cols[0] if categorical_cols else None
    violin_val = numeric_cols[0] if numeric_cols else None
    
    plot_dir = 'static/correlation'
    scatter_path, density_2d_path, density_1d_path, violin_path = generate_correlation_plots(
        df,
        plot_dir,
        x_col=scatter_x,
        y_col=scatter_y,
        density_x=density_x,
        density_y=density_y,
        violin_cat=violin_cat,
        violin_val=violin_val
    )
    nrows = len(df)
    return render_template(
        'correlation/correlation.html',
        filename=os.path.basename(filepath),
        nrows=nrows,
        scatter_plot_url='/' + scatter_path,
        density_2d_plot_url='/' + density_2d_path,
        density_1d_plot_url='/' + density_1d_path if density_1d_path else None,
        violin_plot_url='/' + violin_path,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scatter_x=scatter_x,
        scatter_y=scatter_y,
        density_x=density_x,
        density_y=density_y,
        violin_cat=violin_cat,
        violin_val=violin_val
    )

# AJAX endpoint để vẽ lại scatter plot
@app.route('/correlation/scatter_plot', methods=['POST'])
def scatter_plot_ajax():
    import glob
    from flask import jsonify
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        return jsonify({'error': 'No data file found'}), 400
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    numeric_cols = [
        "original_price", "price", "review_count", "rating_average",
        "favourite_count", "number_of_images", "vnd_cashback", "quantity_sold"
    ]
    
    data = request.get_json()
    scatter_x = data.get('scatter_x', numeric_cols[0] if numeric_cols else None)
    scatter_y = data.get('scatter_y', numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None))
    
    plot_dir = 'static/correlation'
    scatter_path = generate_scatter_plot(
        df,
        plot_dir,
        x_col=scatter_x,
        y_col=scatter_y
    )
    return jsonify({
        'scatter_plot_url': '/' + scatter_path
    })

# AJAX endpoint để vẽ lại density plots
@app.route('/correlation/density_plot', methods=['POST'])
def density_plot_ajax():
    import glob
    from flask import jsonify
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        return jsonify({'error': 'No data file found'}), 400
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    numeric_cols = [
        "original_price", "price", "review_count", "rating_average",
        "favourite_count", "number_of_images", "vnd_cashback", "quantity_sold"
    ]
    
    data = request.get_json()
    density_x = data.get('density_x', numeric_cols[0] if numeric_cols else None)
    density_y = data.get('density_y', numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None))
    
    plot_dir = 'static/correlation'
    (density_2d_path, density_1d_path) = generate_density_plot(
        df,
        plot_dir,
        density_x=density_x,
        density_y=density_y
    )
    
    return jsonify({
        'density_2d_plot_url': '/' + density_2d_path,
        'density_1d_plot_url': '/' + density_1d_path if density_1d_path else None,
    })

# AJAX endpoint để vẽ lại violin plots
@app.route('/correlation/violin_plot', methods=['POST'])
def violin_plot_ajax():
    import glob
    from flask import jsonify
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
    if not files:
        return jsonify({'error': 'No data file found'}), 400
    filepath = max(files, key=os.path.getctime)
    df = pd.read_csv(filepath)
    numeric_cols = [
        "original_price", "price", "review_count", "rating_average",
        "favourite_count", "number_of_images", "vnd_cashback", "quantity_sold"
    ]
    categorical_cols = [
        "name", "fulfillment_type", "brand", "pay_later", "current_seller",
        "has_video", "category"
    ]
    data = request.get_json()
    violin_cat = data.get('violin_cat', categorical_cols[0] if categorical_cols else None)
    violin_val = data.get('violin_val', numeric_cols[0] if numeric_cols else None)
    
    print(f"Violin category: {violin_cat}, Violin value: {violin_val}")
    plot_dir = 'static/correlation'
    violin_path = generate_violin_plot(
        df,
        plot_dir,
        violin_cat=violin_cat,
        violin_val=violin_val
    )
    
    return jsonify({
        'violin_plot_url': '/' + violin_path
    })

if __name__ == '__main__':
    app.run(debug=True)