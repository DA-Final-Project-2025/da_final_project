from flask import Flask, request, render_template, redirect, url_for
import os
import time
import pandas as pd
from utils.analysis import generate_summary_stats, generate_plots, train_and_predict

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "Không có file được tải lên"

    file = request.files['file']
    if file.filename == '':
        return "Chưa chọn file"

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    if 'price' not in df.columns or 'brand' not in df.columns:
        return "File thiếu cột 'price' hoặc 'brand'", 400

    price_range = request.form.get('price_range')
    if price_range == "low":
        df = df[df['price'] < 500000]
    elif price_range == "mid":
        df = df[(df['price'] >= 500000) & (df['price'] <= 2000000)]
    elif price_range == "high":
        df = df[df['price'] > 2000000]

    brand = request.form.get('brand')
    if brand:
        df = df[df['brand'].str.contains(brand, case=False, na=False)]

    summary = generate_summary_stats(df)
    plots = generate_plots(df)
    accuracy_dict, shap_plot = train_and_predict(df)

    return render_template(
        'result.html',
        summary=summary,
        plots=plots,
        accuracy=accuracy_dict,
        shap_plot='shap_plot.png'
    )

if __name__ == '__main__':
    app.run(debug=True)