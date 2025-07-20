import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import io

def describe_numeric(df, columns):
    stats = {}
    for col in columns:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std(),
                'count': df[col].count()
            }
    return stats

def generate_boxplot_svgs(df, columns):
    # Định nghĩa đơn vị cho từng thuộc tính
    units = {
        "original_price": "VNĐ",
        "price": "VNĐ",
        "review_count": "lượt",
        "rating_average": "sao",
        "favourite_count": "lượt",
        "number_of_images": "ảnh",
        "vnd_cashback": "VNĐ",
        "quantity_sold": "lượt"
    }
    svgs = {}
    for col in columns:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(5, 7))
            box = sns.boxplot(y=df[col], ax=ax, color='#e3f2fd', linewidth=2, fliersize=6)
            max_val = df[col].max()
            ax.scatter(0, max_val, color='red', s=120, label='Max', zorder=5)
            unit = units.get(col, "")
            ax.set_ylabel(f"{col} ({unit})", fontsize=14, color='#0d6efd')
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            # Annotation giá trị max
            ax.annotate(f'Max: {max_val:,.2f}', xy=(0, max_val), xytext=(0.2, max_val),
                        textcoords='data', color='red', fontsize=12, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.legend()
            for spine in ax.spines.values():
                spine.set_visible(False)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='svg')
            plt.close(fig)
            svgs[col] = buf.getvalue().decode('utf-8')
    return svgs