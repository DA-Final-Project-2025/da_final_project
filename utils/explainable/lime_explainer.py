import io, base64
from lime.lime_tabular import LimeTabularExplainer
from utils.explainable.train_model import get_x_train, get_features, get_x_test, get_trained_model, get_y_test
import matplotlib.pyplot as plt

def explain_instance(index):
    try:
        index = int(index)
        model = get_trained_model()
        X_train = get_x_train()
        X_test = get_x_test()
        Y_test = get_y_test()
        features = get_features()
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=features,
            mode='regression'
        )

        data_row = X_test.iloc[index].values

        # 5. Giải thích tại điểm i
        exp = explainer.explain_instance(
            data_row=data_row,
            predict_fn=model.predict,
            num_features=len(features)
        )

        buf = io.BytesIO()
        plt.figure()  # Mở một figure mới
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return {
            "instance_details": get_instance_details(X_test, Y_test, index),
            "plot": image_base64
        }
    except Exception as e:
        print(f"[explain_instance error] {type(e).__name__}: {e}")
        return {'error': f'Explain_instance failed: {str(e)}'}

def get_instance_details(X, Y, index):
    features = [
        'original_price', 'price', 'fulfillment_type', 'brand',
        'review_count', 'rating_average', 'favourite_count',
        'pay_later', 'date_created', 'number_of_images',
        'vnd_cashback', 'has_video', 'category'
    ]

    instance = X.iloc[index][features].to_dict()
    instance['quantity_sold'] = Y.iloc[index]  # Or X.iloc[index]['quantity_sold'] if it's part of X

    # Formatting function
    def format_value(key, value):
        if key in ['original_price', 'price', 'vnd_cashback']:
            return f"{int(value):,}₫"
        elif key == 'pay_later':
            return "Có hỗ trợ" if value else "Không hỗ trợ"
        elif key == 'has_video':
            return "Có" if value else "Không"
        elif key == 'quantity_sold':
            return f"{int(value):,}"
        else:
            return value

    return {key: format_value(key, value) for key, value in instance.items()}