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
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def wrap_or_trim(label, wrap_at=15, max_len=20):
    text = str(label)
    if len(text) > max_len:
        text = text[:max_len - 3] + '...'
    elif len(text) > wrap_at:
        text = text[:wrap_at] + '\n' + text[wrap_at:]
    return text

def generate_correlation_plots(
    df, plot_dir,
    x_col=None, y_col=None, 
    density_x=None, density_y=None, 
    violin_cat=None, violin_val=None
):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    scatter_path = generate_scatter_plot(
        df, plot_dir,
        x_col=x_col, y_col=y_col
    )

    (density_2d_path, density_1d_path) = generate_density_plot(
        df, plot_dir,
        density_x=density_x, density_y=density_y
    )

    violin_path = generate_violin_plot(
        df, plot_dir,
        violin_cat=violin_cat, violin_val=violin_val
    )

    return scatter_path, density_2d_path, density_1d_path, violin_path

def generate_scatter_plot(
    df, plot_dir,
    x_col=None, y_col=None
):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Initialize paths for plots
    scatter_path = None

    # Scatter plot
    if x_col is None:
        x_col = df.select_dtypes('number').columns[0]
    if y_col is None:
        y_col = df.select_dtypes('number').columns[1]
    scatter_path = os.path.join(plot_dir, 'scatter.svg')
    plt.figure(figsize=(10,4))
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
    
    return scatter_path

def generate_density_plot(
    df, plot_dir,
    density_x=None, density_y=None
):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Initialize paths for plots
    density_2d_path = None
    density_1d_path = None

    # Density plot: 
    if density_x is None:
        density_x = df.select_dtypes('number').columns[0]
    if density_y is None:
        density_y = df.select_dtypes('number').columns[1]

    unit_x = units.get(density_x, "")
    unit_y = units.get(density_y, "")

    # Vẽ joint kdeplot giữa 2 cột

    density_2d_path = os.path.join(plot_dir, 'density_2d.svg')

    # Drop NaN đồng thời ở cả 2 cột
    df_density = df[[density_x, density_y]].dropna()

    plt.figure(figsize=(10, 4))
    try:
        sns.kdeplot(
            x=df_density[density_x],
            y=df_density[density_y],
            fill=True,
            cmap="Blues",
            thresh=0.05  # Ngưỡng vẽ contour
        )
        
        title = (
            f'Density plot giữa {density_x} ({unit_x}) và {density_y} ({unit_y})'
            if unit_x or unit_y else
            f'Density plot giữa {density_x} và {density_y}'
        )
        plt.xlabel(f"{density_x} ({unit_x})" if unit_x else density_x)
        plt.ylabel(f"{density_y} ({unit_y})" if unit_y else density_y)
        plt.title(title)

    except Exception as e:
        print(f"Density plot error: {e}")
        plt.text(0.5, 0.5, f"Không thể vẽ density plot:\n{e}", ha='center', va='center')
        plt.gca().set_axis_off()  # Ẩn trục khi lỗi

    plt.tight_layout()
    plt.savefig(density_2d_path)
    plt.close()

    print("Density plot saved at:", density_2d_path)

    # Kiểm tra đơn vị giống nhau không
    same_unit = (unit_x == unit_y)

    if same_unit:
        # --- Vẽ 2 Density Plot chồng nhau ---
        plt.figure(figsize=(10, 5))

        sns.kdeplot(data=df_density, x=density_x, fill=True, label=density_x, color='blue')
        sns.kdeplot(data=df_density, x=density_y, fill=True, label=density_y, color='green')

        unit_label = f" ({unit_x})" if unit_x else ""

        plt.xlabel(f"Giá trị{unit_label}")
        plt.title(f'Density Plot của {density_x} và {density_y}{unit_label}')
        plt.legend()

        density_1d_path = os.path.join(plot_dir, 'density_1d.svg')
        plt.tight_layout()
        plt.savefig(density_1d_path)
        plt.close()

        print("Density 1D chồng nhau plot saved at:", density_1d_path)

    return density_2d_path, density_1d_path

def generate_violin_plot(
    df, plot_dir,
    violin_cat=None, violin_val=None
):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Initialize paths for plots
    violin_path = None

    # Violin plot: chỉ lấy top 10 nhóm lớn nhất, các nhóm còn lại gộp thành 'Others' (nếu số nhóm > 10)
    if violin_cat is None:
        violin_cat = 'category' if 'category' in df.columns else df.select_dtypes('object').columns[0]
    if violin_val is None:
        violin_val = df.select_dtypes('number').columns[0]
    violin_path = os.path.join(plot_dir, 'violin.svg')

    # Kiểm tra số nhóm thực tế
    group_counts = df[violin_cat].value_counts()
    num_unique_groups = group_counts.shape[0]

    df_violin = df.copy()

    if num_unique_groups > 10:
        top_groups = group_counts.nlargest(10).index.tolist()
        df_violin[violin_cat] = df_violin[violin_cat].apply(lambda x: x if x in top_groups else 'Others')
        x_order = top_groups + ['Others']
    else:
        x_order = group_counts.index.tolist()  # Giữ nguyên thứ tự nhóm thực tế

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df_violin, x=violin_cat, y=violin_val, order=x_order)

    # Áp dụng custom nhãn trục X
    ax = plt.gca()
    new_labels = [wrap_or_trim(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=30)

    unit_val = units.get(violin_val, "")
    title = (
        f'Violin plot của {violin_val} ({unit_val}) theo {violin_cat}'
        + (f' (Top 10 nhóm, nhóm nhỏ gộp "Others")' if num_unique_groups > 10 else '')
        if unit_val else
        f'Violin plot của {violin_val} theo {violin_cat}'
        + (f' (Top 10 nhóm, nhóm nhỏ gộp "Others")' if num_unique_groups > 10 else '')
    )
    plt.ylabel(f"{violin_val} ({unit_val})" if unit_val else violin_val)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(violin_path)
    plt.close()

    print("Violin plot saved at:", violin_path)
    
    return violin_path
    