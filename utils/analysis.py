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

def generate_feature_distribution_svgs(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    svgs = {}

    # Numeric: Histogram
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.hist(df[col].dropna(), bins=30, color='#90caf9', edgecolor='black')
        ax.set_title(f'Histogram: {col}', fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='svg')
        plt.close(fig)
        svgs[col] = buf.getvalue().decode('utf-8')

    # Categorical: Pie (<=10) hoặc Bar (>10)
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))  # tăng chiều ngang cho legend
        vc = value_counts.copy()
        if len(vc) > 5:
            others = vc[5:].sum()
            vc = vc[:5]
            if others > 0:
                vc['Khác'] = others
        colors = sns.color_palette('pastel', len(vc))
        wedges, texts, autotexts = ax.pie(
            vc,
            autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10}
        )
        ax.set_ylabel('')
        ax.set_title(f'Pie chart: {col}', fontsize=11)
        # Thêm legend bên phải
        ax.legend(wedges, vc.index, title="Nhóm", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        plt.close(fig)
        svgs[col] = buf.getvalue().decode('utf-8')

    return svgs