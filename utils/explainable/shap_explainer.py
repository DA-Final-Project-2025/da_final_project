import joblib
import os, glob, io, base64
import shap
import matplotlib.pyplot as plt
from utils.explainable.train_model import get_x_test

def init_shap(model, X_test):
    print("Start init shap")
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static'))
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(shap_values, os.path.join(save_dir, 'shap_values.pkl'))

    print("Done init shap")
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
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        shap_path = os.path.join(static_dir, 'shap_values.pkl')

        if not os.path.exists(shap_path):
            return {
                'error': 'SHAP files not found'
            }

        shap_values = joblib.load(shap_path)
        X_test = get_x_test()

        return {
            'shap_summary_plot': plot_to_base64(lambda: plot_summary(shap_values, X_test)),
            'shap_specific_feature': plot_to_base64(lambda: plot_specific_feature(feature, shap_values, X_test)),
            'shap_waterfall_chart': plot_to_base64(lambda: plot_waterfall(shap_values))
        }
    except Exception as e:
        print(f"[SHAP plotting error] {type(e).__name__}: {e}")
        return {'error': f'SHAP plotting failed: {str(e)}'}

def plot_shap_specific_feature(feature):
    try:
        base_dir = os.path.dirname(__file__)
        static_dir = os.path.abspath(os.path.join(base_dir, '../../static'))

        shap_path = os.path.join(static_dir, 'shap_values.pkl')

        if not os.path.exists(shap_path):
            return ''
        shap_values = joblib.load(shap_path)
        X_test = get_x_test()

        return plot_to_base64(lambda: plot_specific_feature(feature, shap_values, X_test))
    except Exception as e:
        print(f"[SHAP plot_shap_specific_feature error] {type(e).__name__}: {e}")
        return ''
