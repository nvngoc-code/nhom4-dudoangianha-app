import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cấu hình giao diện trang web Streamlit
st.set_page_config(
    page_title="Phân cụm K-means - Nhóm 4",
    page_icon="📊",
    layout="wide"
)

st.title("Ứng dụng Phân cụm Dữ liệu (K-means Clustering)")
st.markdown("---")

# --- 1. TẢI DỮ LIỆU ---
@st.cache_data
def load_data():
    file_path = 'nhom4_train_data.xlsx'
    df = pd.read_excel(file_path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Không tìm thấy file dữ liệu `nhom4_train_data.xlsx`. Lỗi chi tiết: {e}")
    st.stop()

# Hiển thị tổng quan dữ liệu ở thanh Sidebar hoặc màn hình chính
with st.expander("🔍 Xem thông tin tổng quan và dữ liệu gốc"):
    st.write(f"**Số lượng dòng/cột:** {df.shape}")
    st.dataframe(df.head(10))

# --- 2. THANH CÔNG CỤ (SIDEBAR) ---
st.sidebar.header("⚙️ Tùy chỉnh mô hình")

# Tự động chọn các cột có kiểu dữ liệu số làm features mặc định
default_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Cho phép người dùng tùy chọn thuộc tính phân cụm
features = st.sidebar.multiselect(
    "Chọn các thuộc tính (features) để phân cụm:",
    options=df.select_dtypes(include=[np.number]).columns.tolist(),
    default=default_features
)

if not features:
    st.warning("Vui lòng chọn ít nhất một thuộc tính ở thanh bên trái để tiếp tục!")
    st.stop()

# --- 3. TIỀN XỬ LÝ DỮ LIỆU & PHƯƠNG PHÁP ELBOW ---
# Lọc dữ liệu và chuẩn hóa
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tính toán Elbow (Khuỷu tay)
inertia_values = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Giao diện hiển thị biểu đồ Elbow
st.subheader("📈 Xác định số cụm tối ưu bằng phương pháp Elbow")
col1, col2 = st.columns([2, 1])

with col1:
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
    ax_elbow.plot(K_range, inertia_values, marker='o', linestyle='--', color='b')
    ax_elbow.set_title('Phương pháp Elbow', fontsize=12)
    ax_elbow.set_xlabel('Số lượng cụm (k)', fontsize=10)
    ax_elbow.set_ylabel('Inertia', fontsize=10)
    ax_elbow.set_xticks(K_range)
    ax_elbow.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig_elbow)

with col2:
    st.markdown("""
    **Hướng dẫn chọn số cụm ($k$):**
    * Quan sát biểu đồ bên cạnh tại vị trí đường cong bắt đầu bẻ cong lại (hình khuỷu tay).
    * Chọn giá trị $k$ phù hợp nhất ở thanh tùy chỉnh bên dưới.
    """)
    optimal_k = st.sidebar.slider("Chọn số lượng cụm (optimal_k):", min_value=1, max_value=10, value=3)

st.markdown("---")

# --- 4. HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC & GÁN NHÃN ---
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_model.fit(X_scaled)

# Tạo bản sao để hiển thị kết quả
X_result = X.copy()
X_result['Cluster_Label'] = kmeans_model.labels_

st.subheader(f"🎯 Kết quả phân cụm với $k = {optimal_k}$")

# Thống kê số lượng phần tử mỗi cụm
col_res1, col_res2 = st.columns([1, 2])

with col_res1:
    st.markdown("**Số lượng phần tử trong mỗi cụm:**")
    cluster_counts = X_result['Cluster_Label'].value_counts().reset_index()
    cluster_counts.columns = ['Cụm', 'Số lượng']
    st.dataframe(cluster_counts, hide_index=True)

with col_res2:
    st.markdown("**Bảng dữ liệu sau khi được gán nhãn cụm:**")
    st.dataframe(X_result.head(10), use_container_width=True)

# --- 5. TRỰC QUAN HÓA PHÂN CỤM ---
if len(features) >= 2:
    st.subheader("📊 Trực quan hóa không gian 2 thuộc tính đầu tiên")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(
        x=X_scaled[:, 0],
        y=X_scaled[:, 1],
        hue=X_result['Cluster_Label'],
        palette='viridis',
        legend='full',
        s=80,
        ax=ax_cluster
    )
    
    # Vẽ tâm cụm (Centroids)
    centroids = kmeans_model.cluster_centers_
    ax_cluster.scatter(
        centroids[:, 0], centroids[:, 1], 
        s=250, c='red', marker='X', 
        label='Tâm cụm (Centroids)'
    )
    
    ax_cluster.set_title(f'Phân cụm K-means (k={optimal_k})', fontsize=14)
    ax_cluster.set_xlabel(f'{features[0]} (đã chuẩn hóa)', fontsize=11)
    ax_cluster.set_ylabel(f'{features[1]} (đã chuẩn hóa)', fontsize=11)
    ax_cluster.legend()
    ax_cluster.grid(True, alpha=0.3)
    
    st.pyplot(fig_cluster)
else:
    st.info("Cần chọn ít nhất 2 thuộc tính số để vẽ biểu đồ trực quan hóa 2D.")