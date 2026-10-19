import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 페이지 설정 (탭 제목 및 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 타이틀 및 설명
st.title("🎬 영화 유형 나누기 (K-Means 클러스터링)")
st.markdown("영화 데이터를 활용하여 비지도 학습 기반으로 영화의 유형을 묶어보고 시각화합니다.")

# 1. 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    total_count = len(df)
    
    # 필수 수치 데이터 컬럼 변환
    num_cols = ['first_scrn', 'total_audi', 'first_week_audi', 'days_in_top10']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 데이터 결측치 제거 및 첫 주 관객 0 이하 제거
    df_clean = df.dropna(subset=num_cols).copy()
    df_clean = df_clean[df_clean['first_week_audi'] > 0].copy()
    
    # 속성 생성 및 변환
    # 1) 스크린 수 상용로그 (np.log10)
    df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
    # 2) 누적 관객 상용로그
    df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])
    # 3) 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20 clipping)
    df_clean['long_run_index'] = (df_clean['total_audi'] / df_clean['first_week_audi']).clip(upper=20)
    # 4) 10위권 일수 (days_in_top10은 그대로 사용)
    
    filtered_count = len(df_clean)
    
    return df_clean, total_count, filtered_count

df, total_count, filtered_count = load_data()

# 전체 편수 및 사용 편수 안내
st.info(f"📊 **전체 영화 수:** {total_count:,}편 | **분석 대상 영화 수(결측치 및 조건 제외):** {filtered_count:,}편")

# 속성 매핑 정보
feature_dict = {
    "log_first_scrn": "스크린 수 (로그)",
    "log_total_audi": "누적 관객 (로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수"
}
inv_feature_dict = {v: k for k, v in feature_dict.items()}

# 2. 클러스터링 속성 선택
st.subheader("1. 묶는 데 사용할 속성 선택")
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 최소 2개 이상 선택하세요.",
    options=list(feature_dict.values()),
    default=list(feature_dict.values())
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

selected_features = [inv_feature_dict[name] for name in selected_feature_names]

# 3. 데이터 표준화 및 K-Means 군집화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

# random_state 고정
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['cluster_raw'] = clusters

# 누적 관객 수(total_audi)의 평균이 높은 순서대로 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
label_map = {raw_id: label for raw_id, label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}

df['cluster'] = df['cluster_raw'].map(label_map)

# 4. 시각화 섹션
st.subheader("2. 영화 유형 시각화")

# 2D 산점도
st.write("### 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis_name = st.selectbox("가로축 (X축)", options=selected_feature_names, index=0, key="2d_x")
with col2:
    y_axis_name = st.selectbox("세로축 (Y축)", options=selected_feature_names, index=min(1, len(selected_feature_names)-1), key="2d_y")

fig_2d = px.scatter(
    df,
    x=inv_feature_dict[x_axis_name],
    y=inv_feature_dict[y_axis_name],
    color="cluster",
    hover_name="movieNm",
    labels={
        inv_feature_dict[x_axis_name]: x_axis_name,
        inv_feature_dict[y_axis_name]: y_axis_name,
        "cluster": "유형"
    },
    title=f"2차원 산점도 ({x_axis_name} vs {y_axis_name})",
    category_orders={"cluster": ['㉮', '㉯', '㉰']}
)
st.plotly_chart(fig_2d, use_container_width=True)

# 3D 산점도
st.write("### 3차원 산점도")
if len(selected_feature_names) < 3:
    st.warning("⚠️ 3차원 산점도를 그리려면 선택된 속성이 3개 이상이어야 합니다.")
else:
    col3_1, col3_2, col3_3 = st.columns(3)
    with col3_1:
        x_3d_name = st.selectbox("X축", options=selected_feature_names, index=0, key="3d_x")
    with col3_2:
        y_3d_name = st.selectbox("Y축", options=selected_feature_names, index=1, key="3d_y")
    with col3_3:
        z_3d_name = st.selectbox("Z축", options=selected_feature_names, index=2, key="3d_z")
        
    fig_3d = px.scatter_3d(
        df,
        x=inv_feature_dict[x_3d_name],
        y=inv_feature_dict[y_3d_name],
        z=inv_feature_dict[z_3d_name],
        color="cluster",
        hover_name="movieNm",
        labels={
            inv_feature_dict[x_3d_name]: x_3d_name,
            inv_feature_dict[y_3d_name]: y_3d_name,
            inv_feature_dict[z_3d_name]: z_3d_name,
            "cluster": "유형"
        },
        title=f"3차원 산점도 ({x_3d_name} vs {y_3d_name} vs {z_3d_name})",
        category_orders={"cluster": ['㉮', '㉯', '㉰']}
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# 5. 묶음별 정보 표시
st.subheader("3. 묶음별 요약 정보 및 대표 영화")

# 원래 단위 속성 명칭
orig_features = {
    'first_scrn': '스크린 수 (평균)',
    'total_audi': '누적 관객 (평균)',
    'days_in_top10': '10위권 일수 (평균)',
    'long_run_index': '롱런 지수 (평균)'
}

# 묶음별 통계 표
summary_list = []
for c_label in ['㉮', '㉯', '㉰']:
    sub_df = df[df['cluster'] == c_label]
    count = len(sub_df)
    
    scrn_mean = sub_df['first_scrn'].mean()
    audi_mean = sub_df['total_audi'].mean()
    top10_mean = sub_df['days_in_top10'].mean()
    longrun_mean = sub_df['long_run_index'].mean()
    
    summary_list.append({
        '유형': c_label,
        '영화 수': f"{count:,}편",
        '스크린 수 (원래 단위 평균)': f"{scrn_mean:,.1f}개",
        '누적 관객 (원래 단위 평균)': f"{audi_mean:,.0f}명",
        '10위권 일수 (원래 단위 평균)': f"{top10_mean:,.1f}일",
        '롱런 지수 (원래 단위 평균)': f"{longrun_mean:,.2f}"
    })

summary_df = pd.DataFrame(summary_list).set_index('유형')
st.write("#### 📊 유형별 평균 속성 값")
st.dataframe(summary_df, use_container_width=True)

# 묶음별 누적 관객 탑 5 영화
st.write("#### 🏆 유형별 누적 관객 상위 5개 영화")
top5_cols = st.columns(3)

for idx, c_label in enumerate(['㉮', '㉯', '㉰']):
    with top5_cols[idx]:
        st.markdown(f"##### 유형 {c_label}")
        top5_movies = df[df['cluster'] == c_label].sort_values(by='total_audi', ascending=False).head(5)
        
        for rank, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"**{rank}. {row['movieNm']}** ({row['total_audi']:,}명)")