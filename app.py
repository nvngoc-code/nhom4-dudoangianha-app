import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cấu hình giao diện trang web Streamlit
st.set_page_config(
    page_title="Phân cụm & Hồi quy - Nhóm 4",
    page_icon="📊",
    layout="wide"
)

st.title("Ứng dụng Khai phá Dữ liệu: K-means Clustering & Linear Regression")
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

# Hiển thị tổng quan dữ liệu
with st.expander("🔍 Xem thông tin tổng quan và dữ liệu gốc"):
    st.write(f"**Số lượng dòng/cột:** {df.shape}")
    st.dataframe(df.head(10))

# Tạo các tab chức năng để giao diện gọn gàng và chuyên nghiệp hơn
tab1, tab2 = st.tabs(["📈 Phân cụm K-means", "🏠 Mô hình Hồi quy tuyến tính (Linear Regression)"])

# ==========================================
# TAB 1: PHÂN CỤM K-MEANS
# ==========================================
with tab1:
    st.header("Thuật toán Phân cụm K-means")
    
    st.sidebar.header("⚙️ Tùy chỉnh K-means")
    default_features_kmeans = df.select_dtypes(include=[np.number]).columns.tolist()

    features_kmeans = st.sidebar.multiselect(
        "Chọn thuộc tính phân cụm:",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=default_features_kmeans
    )

    if not features_kmeans:
        st.warning("Vui lòng chọn ít nhất một thuộc tính ở thanh bên trái!")
    else:
        X_kmeans = df[features_kmeans].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_kmeans)

        # Phương pháp Elbow
        inertia_values = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia_values.append(kmeans.inertia_)

        st.subheader("📈 Phương pháp Elbow (Xác định số cụm tối ưu)")
        col_el1, col_el2 = st.columns([2, 1])

        with col_el1:
            fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
            ax_elbow.plot(K_range, inertia_values, marker='o', linestyle='--', color='b')
            ax_elbow.set_xlabel('Số lượng cụm (k)')
            ax_elbow.set_ylabel('Inertia')
            ax_elbow.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_elbow)

        with col_el2:
            optimal_k = st.slider("Chọn số lượng cụm (k):", min_value=1, max_value=10, value=3)

        # Huấn luyện K-means chính thức
        kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_model.fit(X_scaled)
        
        X_result = X_kmeans.copy()
        X_result['Cluster_Label'] = kmeans_model.labels_

        st.subheader(f"🎯 Kết quả phân cụm ($k = {optimal_k}$)")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown("**Số lượng phần tử mỗi cụm:**")
            cluster_counts = X_result['Cluster_Label'].value_counts().reset_index()
            cluster_counts.columns = ['Cụm', 'Số lượng']
            st.dataframe(cluster_counts, hide_index=True)
        with col_res2:
            st.markdown("**Dữ liệu sau khi gán nhãn:**")
            st.dataframe(X_result.head(5), use_container_width=True)


# ==========================================
# TAB 2: HỒI QUY TUYẾN TÍNH
# ==========================================
with tab2:
    st.header("Dự đoán với Mô hình Hồi quy Tuyến tính (Linear Regression)")

    # Tự động chọn biến mục tiêu (cột cuối) và các biến đặc trưng
    target_col = df.columns[-1]
    default_features_lr = [col for col in df.columns if col != target_col and np.issubdtype(df[col].dtype, np.number)]

    st.markdown(f"**Biến mục tiêu mặc định (Target):** `{target_col}`")
    
    # Cho phép người dùng tùy chọn features cho Linear Regression nếu muốn
    features_lr = st.multiselect(
        "Chọn các biến đặc trưng (Features) huấn luyện hồi quy:",
        options=[col for col in df.select_dtypes(include=[np.number]).columns if col != target_col],
        default=default_features_lr
    )

    if not features_lr:
        st.warning("Vui lòng chọn ít nhất một biến đặc trưng!")
    else:
        # Tiền dữ liệu cho Linear Regression
        df_lr = df[features_lr + [target_col]].dropna()
        X_lr = df_lr[features_lr]
        y_lr = df_lr[target_col]

        # Chia tập dữ liệu 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

        # Huấn luyện mô hình
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Dự đoán
        y_pred = lr_model.predict(X_test)

        # Đánh giá chỉ số
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Hiển thị các chỉ số đánh giá
        st.subheader("📊 Các chỉ số đánh giá mô hình")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MSE", f"{mse:,.2f}")
        col_m2.metric("RMSE", f"{rmse:,.2f}")
        col_m3.metric("R² Score", f"{r2:.4f}")

        st.markdown("---")

        # Hiển thị bảng đối chiếu thực tế và dự đoán
        st.subheader("📋 Bảng đối chiếu Giá thực tế và Giá dự đoán (5 dòng đầu)")
        compare_df = pd.DataFrame({
            'Giá thực tế (Actual)': y_test.iloc[:5].values,
            'Giá dự đoán (Predicted)': y_pred[:5]
        })
        st.dataframe(compare_df, use_container_width=True)

        # Biểu đồ trực quan hóa Actual vs Predicted
        st.subheader("📈 Biểu đồ so sánh Giá Thực tế và Giá Dự đoán")
        fig_lr, ax_lr = plt.subplots(figsize=(8, 4))
        ax_lr.scatter(y_test, y_pred, color='purple', alpha=0.7, edgecolors='k')
        ax_lr.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Đường chuẩn lý tưởng (y = x)')
        ax_lr.set_xlabel('Giá thực tế (Actual)')
        ax_lr.set_ylabel('Giá dự đoán (Predicted)')
        ax_lr.set_title('So sánh Actual vs Predicted trên tập Test')
        ax_lr.legend()
        ax_lr.grid(True, alpha=0.3)
        st.pyplot(fig_lr)