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

numeric_cols = [
    "original_price", "price", "review_count", "rating_average",
    "favourite_count", "number_of_images", "vnd_cashback", "quantity_sold"
]

categorical_cols = [
    "name", "fulfillment_type", "brand", "pay_later", "current_seller",
    "has_video", "category"
]

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def wrap_or_trim(label, wrap_at=15, max_len=20):
    text = str(label)
    if len(text) > max_len:
        text = text[:max_len - 3] + '...'
    elif len(text) > wrap_at:
        text = text[:wrap_at] + '\n' + text[wrap_at:]
    return text

def auto_generate_scatter_plot(df, plot_dir, method='pearson', threshold=0.7, max_plots=10):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Initialize paths for plots
    maxtrix_path = None
    scatter_paths = []

    numeric_df = df[numeric_cols]

    corr_matrix = numeric_df.corr(method=method)

    # Vẽ heatmap
    maxtrix_path = os.path.join(plot_dir, f"{method}_matrix.svg")
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Ma trận tương quan ({method.title()})")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(maxtrix_path)
    plt.close()
    print("Matrix plot saved at:", maxtrix_path)
    
    # Lọc các cặp có tương quan mạnh
    corr_pairs = corr_matrix.unstack().dropna()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.map(abs).sort_values(ascending=False)

    printed = set()
    plot_count = 0

    print(f"Vẽ scatter plot cho các cặp có tương quan mạnh theo {method.title()}:")

    for (x_col, y_col), r in corr_pairs.items():
        key = tuple(sorted((x_col, y_col)))
        if key in printed or r < threshold:
            continue
        
        # Vẽ biểu đồ
        scatter_path = os.path.join(plot_dir, f"{method}_scatter_{plot_count}.svg")
        scatter_paths.append(scatter_path)

        plt.figure(figsize=(10, 4))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        unit_x = units.get(x_col, "")
        unit_y = units.get(y_col, "")
        title = f'Scatter plot giữa {x_col} ({unit_x}) và {y_col} ({unit_y})' if unit_x or unit_y else f'Scatter plot giữa {x_col} và {y_col}'
        plt.xlabel(f"{x_col} ({unit_x})" if unit_x else x_col)
        plt.ylabel(f"{y_col} ({unit_y})" if unit_y else y_col)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        print("Scatter plot saved at:", scatter_path)

        printed.add(key)
        plot_count += 1
        if plot_count >= max_plots:
            print(f"Đã đạt giới hạn {max_plots} biểu đồ.")
            break

    return maxtrix_path, scatter_paths, plot_count

def auto_generate_anova_plot(df, plot_dir, pvalue_threshold=0.05, max_plots=20):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Initialize paths for plots
    anova_paths = []

    plot_count = 0

    df_violin = df.copy()

    for cat_col in categorical_cols:
        if df_violin[cat_col].nunique() < 2:
            continue  # Bỏ qua nếu ít nhóm

        num_unique_groups = df_violin[cat_col].nunique()

        # Nếu số nhãn > 10 → gộp thành "Other"
        if num_unique_groups > 10:
            top_values = df[cat_col].value_counts().nlargest(10).index
            df_violin['_temp_cat'] = df_violin[cat_col].apply(lambda x: x if x in top_values else 'Others')
            current_cat_col = '_temp_cat'
        else:
            current_cat_col = cat_col

        for num_col in numeric_cols:
            groups = [group[num_col].dropna() for name, group in df_violin.groupby(current_cat_col)]
            if len(groups) < 2:
                continue

            try:
                stat, pval = f_oneway(*groups)
            except:
                continue

            if pval < pvalue_threshold:
                plot_path = os.path.join(plot_dir, f"anova_{plot_count}.svg")

                plt.figure(figsize=(10, 5))
                sns.violinplot(data=df_violin, x=current_cat_col, y=num_col)

                # Áp dụng custom nhãn trục X
                ax = plt.gca()
                new_labels = [wrap_or_trim(label.get_text()) for label in ax.get_xticklabels()]
                ax.set_xticklabels(new_labels, rotation=30)

                unit_val = units.get(num_col, "")
                title = (
                    f'Violin plot của {num_col} ({unit_val}) theo {cat_col}'
                    + (f' (Top 10 nhóm, nhóm nhỏ gộp "Others")' if num_unique_groups > 10 else '')
                    + f' (ANOVA = {pval:.3g})'
                    if unit_val else
                    f'Violin plot của {num_col} theo {cat_col}'
                    + (f' (Top 10 nhóm, nhóm nhỏ gộp "Others")' if num_unique_groups > 10 else '')
                    + f' (ANOVA = {pval:.3g})'
                )
                plt.xlabel(cat_col)
                plt.ylabel(f"{num_col} ({unit_val})" if unit_val else num_col)

                plt.title(title)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print("ANOVA plot saved at:", plot_path)

                anova_paths.append(plot_path)
                plot_count += 1

                if plot_count >= max_plots:
                    break

        if '_temp_cat' in df_violin.columns:
            df_violin.drop(columns=['_temp_cat'], inplace=True)

        if plot_count >= max_plots:
            break

    return anova_paths, plot_count