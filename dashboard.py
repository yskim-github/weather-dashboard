import os
import glob
import sys
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="270m High-Res Weather & Frost Dashboard",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 500;
        opacity: 0.85;
    }
    .frost-warning { color: #e74c3c; }
    .frost-watch { color: #f39c12; }
    .frost-safe { color: #2ecc71; }

    /* Sidebar Layout & Scroll Optimization */
    [data-testid="stSidebar"] {
        min-width: 310px;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 1.2rem !important;
        padding-bottom: 160px !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        overflow-y: auto !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 2px !important;
    }
    [data-testid="stSidebar"] .stSelectbox label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        z-index: 999999 !important;
        max-height: 400px !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 10px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 6px;
    }
    .sidebar-brand img {
        width: 38px;
        height: 38px;
        flex-shrink: 0;
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.2;
        color: #f1f5f9;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        line-height: 1.2;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 로컬(D:/weather/output) 및 클라우드 배포(./output) 자동 감지
_default_out = "D:/weather/output" if os.path.exists("D:/weather/output") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", _default_out)
FROST_DIR = os.getenv("FROST_DIR", os.path.join(OUTPUT_DIR, "xgb_frost_270m"))
FORECAST_DIR = os.getenv("FORECAST_DIR", os.path.join(OUTPUT_DIR, "forecast_270m"))

# ==========================================
# 2. Variable Definitions & Metadata
# ==========================================
var_names = [
    'Tmax', 'Tmin', 'Tavg', 
    'Solar_Rad', 'Sunshine', 
    'Precipitation', 'Humidity', 
    'Wind_3m', 'Wind_10m', 
    'Wind_Dir', 'Wind_Max_10m', 'Wind_Max_3m'
]

VAR_UNITS = {
    'Tmax': '°C',
    'Tmin': '°C',
    'Tavg': '°C',
    'Solar_Rad': 'MJ/m²',
    'Sunshine': 'hr',
    'Precipitation': 'mm',
    'Humidity': '%',
    'Wind_3m': 'm/s',
    'Wind_10m': 'm/s',
    'Wind_Dir': '° (도)',
    'Wind_Max_10m': 'm/s',
    'Wind_Max_3m': 'm/s'
}

VAR_LABELS = {
    'Tmax': '🌡️ 최고기온 (Tmax, °C)',
    'Tmin': '❄️ 최저기온 (Tmin, °C)',
    'Tavg': '⛅ 평균기온 (Tavg, °C)',
    'Solar_Rad': '☀️ 일사량 (Solar_Rad, MJ/m²)',
    'Sunshine': '🌤️ 일조시간 (Sunshine, hr)',
    'Precipitation': '🌧️ 강수량 (Precipitation, mm)',
    'Humidity': '💧 상대습도 (Humidity, %)',
    'Wind_3m': '🍃 3m 풍속 (Wind_3m, m/s)',
    'Wind_10m': '💨 10m 풍속 (Wind_10m, m/s)',
    'Wind_Dir': '🧭 풍향 (Wind_Dir, °)',
    'Wind_Max_10m': '🌪️ 10m 최대풍속 (Wind_Max_10m, m/s)',
    'Wind_Max_3m': '🌪️ 3m 최대풍속 (Wind_Max_3m, m/s)'
}

# ==========================================
# 3. High-Performance Data Loading Functions
# ==========================================
@st.cache_data(ttl=2)
def get_available_dates():
    """계산된 모든 산출물(검증결과, PNG, TIFF, 이상치리포트)로부터 즉시 날짜 추출"""
    dates = set()
    
    # 1. 검증 결과 CSV
    for f in glob.glob(os.path.join(OUTPUT_DIR, "validation_results_*.csv")):
        d = os.path.basename(f).replace("validation_results_", "").replace(".csv", "")
        if d.isdigit() and len(d) == 8:
            dates.add(d)
            
    # 2. PNG 격자 지도
    for f in glob.glob(os.path.join(OUTPUT_DIR, "png", "Tmax", "*", "Tmax_*.png")):
        d = os.path.basename(f).replace("Tmax_", "").replace(".png", "")
        if d.isdigit() and len(d) == 8:
            dates.add(d)
            
    # 3. TIFF 격자 원본
    for f in glob.glob(os.path.join(OUTPUT_DIR, "tiff", "Tmax", "*", "Tmax_*.tiff")):
        d = os.path.basename(f).replace("Tmax_", "").replace(".tiff", "")
        if d.isdigit() and len(d) == 8:
            dates.add(d)
            
    # 4. 이상치 QC 리포트
    for f in glob.glob(os.path.join(OUTPUT_DIR, "anomaly_report_*.csv")):
        d = os.path.basename(f).replace("anomaly_report_", "").replace(".csv", "")
        if d.isdigit() and len(d) == 8:
            dates.add(d)
            
    return sorted(list(dates), reverse=True)

@st.cache_data(ttl=2)
def load_daily_summary():
    summary_path = os.path.join(OUTPUT_DIR, "daily_rmse_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # Wind_Max_3m 컬럼 동기화 (U_max 기반)
        if 'U_max_RMSE' in df.columns:
            if 'Wind_Max_3m_RMSE' not in df.columns:
                df['Wind_Max_3m_RMSE'] = df['U_max_RMSE']
            else:
                df['Wind_Max_3m_RMSE'] = df['Wind_Max_3m_RMSE'].fillna(df['U_max_RMSE'])
        if 'U_max_MAE' in df.columns:
            if 'Wind_Max_3m_MAE' not in df.columns:
                df['Wind_Max_3m_MAE'] = df['U_max_MAE']
            else:
                df['Wind_Max_3m_MAE'] = df['Wind_Max_3m_MAE'].fillna(df['U_max_MAE'])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=2)
def load_validation_data(target_date):
    val_path = os.path.join(OUTPUT_DIR, f"validation_results_{target_date}.csv")
    if os.path.exists(val_path):
        df = pd.read_csv(val_path)
        # Wind_Max_3m 에일리어스 지원 (U_max 연동)
        if 'True_U_max' in df.columns and 'True_Wind_Max_3m' not in df.columns:
            df['Pred_Wind_Max_3m'] = df.get('Pred_U_max')
            df['True_Wind_Max_3m'] = df.get('True_U_max')
            df['Error_Wind_Max_3m'] = df.get('Error_U_max')
            df['AbsError_Wind_Max_3m'] = df.get('AbsError_U_max')
        return df
    return pd.DataFrame()

@st.cache_data(ttl=2)
def load_anomaly_report(target_date):
    path = os.path.join(OUTPUT_DIR, f"anomaly_report_{target_date}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=30)
def render_forecast_map(tiff_path, var_name):
    """270m 예보장 GeoTIFF를 다크 테마 고해상도 히트맵 이미지로 실시간 렌더링"""
    if not os.path.exists(tiff_path):
        return None, None
    import tifffile as tiff
    arr = tiff.imread(tiff_path)
    valid_mask = np.isfinite(arr) & (arr != -999.0) & (arr > -100)
    if not np.any(valid_mask):
        return None, None
    
    val_min = float(np.min(arr[valid_mask]))
    val_max = float(np.max(arr[valid_mask]))
    val_mean = float(np.mean(arr[valid_mask]))
    
    cmap_map = {
        'Tmin': 'coolwarm',
        'Tmax': 'coolwarm',
        'Tavg': 'coolwarm',
        'Precipitation': 'Blues',
        'Humidity': 'YlGnBu',
        'Wind_10m': 'viridis',
        'Solar_Rad': 'inferno',
        'Sunshine': 'YlOrRd'
    }
    cmap_name = cmap_map.get(var_name, 'viridis')
    arr_plot = np.where(valid_mask, arr, np.nan)
    
    fig, ax = plt.subplots(figsize=(6, 7), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    im = ax.imshow(arr_plot, cmap=cmap_name)
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    unit = VAR_UNITS.get(var_name, '')
    cbar.set_label(f"[{unit}]" if unit else "", color='white', fontsize=10)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0E1117')
    plt.close(fig)
    buf.seek(0)
    
    stats = {'min': val_min, 'max': val_max, 'mean': val_mean}
    return buf.getvalue(), stats

TARGET_POINTS = [
    {'name': '청주시 상당구 미원면', 'lat': 36.6575, 'lon': 127.7029, 'row': 962, 'col': 1206, 'valid_idx': 566002, 'alt': 312.0},
    {'name': '국립농업과학원', 'lat': 35.8335, 'lon': 127.0454, 'row': 1301, 'col': 992, 'valid_idx': 887557, 'alt': 17.0},
    {'name': '익산시 월성동', 'lat': 35.9818, 'lon': 126.9859, 'row': 1240, 'col': 973, 'valid_idx': 829518, 'alt': 15.0}
]

@st.cache_data(ttl=10)
def get_point_frost_weather_data(target_date: str):
    """지정된 3개 주요 농업 거점의 D+1~D+3 (3일간) 270m 기상 예보 및 서리 판별값 일괄 추출 (농가대응수칙 제외)"""
    import datetime
    year = target_date[:4]
    try:
        base_dt = datetime.datetime.strptime(target_date, '%Y%m%d')
    except Exception:
        return None
        
    lead_names = {1: 'D+1 (내일)', 2: 'D+2 (모레)', 3: 'D+3 (글피)'}
    frost_badge_map = {
        0: '✅ 안전 (Safe)',
        1: '⚠️ 주의 (Watch)',
        2: '🚨 경고 (Warning)',
        255: '결측'
    }
    
    try:
        from xgb_frost_predictor import get_5km_cloud_cover_0_10, load_xgb_models
        (model_a, scaler_a), (model_b, scaler_b) = load_xgb_models()
    except Exception:
        model_a = None
        
    records = []
    import tifffile as tiff
    
    for lead_day in [1, 2, 3]:
        target_dt = base_dt + datetime.timedelta(days=lead_day)
        target_date_str = target_dt.strftime('%Y-%m-%d')
        
        frost_tiff = os.path.join(FROST_DIR, year, f"XGB_Frost_Grade_{target_date}_D{lead_day}.tiff")
        tmin_tiff = os.path.join(FORECAST_DIR, 'Tmin', year, f'Tmin_forecast_{target_date}_D{lead_day}.tiff')
        pre_tiff = os.path.join(FORECAST_DIR, 'Precipitation', year, f'Precipitation_forecast_{target_date}_D{lead_day}.tiff')
        ws_tiff = os.path.join(FORECAST_DIR, 'Wind_10m', year, f'Wind_10m_forecast_{target_date}_D{lead_day}.tiff')
        hu_tiff = os.path.join(FORECAST_DIR, 'Humidity', year, f'Humidity_forecast_{target_date}_D{lead_day}.tiff')
        tmax_tiff = os.path.join(FORECAST_DIR, 'Tmax', year, f'Tmax_forecast_{target_date}_D{lead_day}.tiff')
        
        if not os.path.exists(frost_tiff):
            continue
            
        f_arr = tiff.imread(frost_tiff)
        tmin_arr = tiff.imread(tmin_tiff) if os.path.exists(tmin_tiff) else None
        pre_arr = tiff.imread(pre_tiff) if os.path.exists(pre_tiff) else None
        ws_arr = tiff.imread(ws_tiff) if os.path.exists(ws_tiff) else None
        hu_arr = tiff.imread(hu_tiff) if os.path.exists(hu_tiff) else None
        tmax_arr = tiff.imread(tmax_tiff) if os.path.exists(tmax_tiff) else None
        
        try:
            clo_all = get_5km_cloud_cover_0_10(target_date, lead_day=lead_day)
        except Exception:
            clo_all = None
            
        for pt in TARGET_POINTS:
            r, c = pt['row'], pt['col']
            v_i = pt['valid_idx']
            
            f_code = int(f_arr[r, c]) if f_arr is not None else 255
            tmin_val = float(tmin_arr[r, c]) if tmin_arr is not None else np.nan
            tmax_val = float(tmax_arr[r, c]) if tmax_arr is not None else np.nan
            pre_val = float(pre_arr[r, c]) if pre_arr is not None else np.nan
            ws_val = float(ws_arr[r, c]) if ws_arr is not None else np.nan
            hu_val = float(hu_arr[r, c]) if hu_arr is not None else np.nan
            clo_val = float(clo_all[v_i]) if clo_all is not None else np.nan
            
            pred_a_str = '-'
            pred_b_str = '-'
            if model_a is not None and not np.isnan(tmin_val) and not np.isnan(ws_val) and not np.isnan(hu_val):
                feat = np.array([[tmin_val, pre_val if not np.isnan(pre_val) else 0.0, ws_val, hu_val, clo_val if not np.isnan(clo_val) else 4.0]], dtype=np.float32)
                feat_a = scaler_a.transform(feat)
                feat_b = scaler_b.transform(feat)
                pa = int(model_a.predict(feat_a)[0])
                pb = int(model_b.predict(feat_b)[0])
                prob_a = float(model_a.predict_proba(feat_a)[0, 0])
                prob_b = float(model_b.predict_proba(feat_b)[0, 0])
                pred_a_str = f"서리 ({prob_a*100:.0f}%)" if pa==0 else f"안전 ({(1-prob_a)*100:.0f}%)"
                pred_b_str = f"서리 ({prob_b*100:.0f}%)" if pb==0 else f"안전 ({(1-prob_b)*100:.0f}%)"
                
            records.append({
                '지점명': pt['name'],
                '예보시점': lead_names[lead_day],
                '예보일자': target_date_str,
                '표고(고도)': f"{pt['alt']:.0f} m",
                '서리 판별': frost_badge_map.get(f_code, str(f_code)),
                '모형 A': pred_a_str,
                '모형 B': pred_b_str,
                '최저기온': f"{tmin_val:.2f} °C" if not np.isnan(tmin_val) else '-',
                '최고기온': f"{tmax_val:.2f} °C" if not np.isnan(tmax_val) else '-',
                '강수량': f"{pre_val:.2f} mm" if not np.isnan(pre_val) else '-',
                '풍속': f"{ws_val:.2f} m/s" if not np.isnan(ws_val) else '-',
                '상대습도': f"{hu_val:.1f} %" if not np.isnan(hu_val) else '-',
                '대기운량': f"{clo_val:.1f} / 10" if not np.isnan(clo_val) else '-'
            })
            
    df_res = pd.DataFrame(records)
    if not df_res.empty:
        df_res = df_res.sort_values(by=['지점명', '예보시점'])
    return df_res

# ==========================================
# 4. Sidebar UI (날짜 & 변수 & 실시간 새로고침)
# ==========================================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <img src="https://cdn-icons-png.flaticon.com/512/1163/1163661.png" alt="logo" />
    <div>
        <div class="sidebar-brand-title">초고해상도 270m 기상</div>
        <div class="sidebar-brand-sub">격자보간 · 품질검증 · 서리 · GNN예보</div>
    </div>
</div>
""", unsafe_allow_html=True)

dates = get_available_dates()
if not dates:
    st.error(f"{OUTPUT_DIR} 에 계산 완료된 데이터가 없습니다.")
    st.stop()

# 1열: 즉시 새로고침 버튼 & 실시간 자동 감지 토글 (가로 2단 배치)
col_btn, col_tog = st.sidebar.columns([1.1, 1.0])
with col_btn:
    if st.button("🔄 새로고침", use_container_width=True, help="백그라운드에서 방금 계산된 최신 일자 데이터를 즉시 불러옵니다."):
        st.cache_data.clear()
        st.rerun()
with col_tog:
    auto_refresh = st.toggle("⚡ 자동감지", value=False, help="켜두시면 10초마다 백그라운드 모형 연산 완료 여부를 자동 감지하여 화면을 최신화합니다.")

st.sidebar.caption(f"🟢 완료: **{len(dates):,}일**" + (" · ⚡ 자동 감지 중" if auto_refresh else ""))

# 2열: 연도 빠른 필터 & 일자 선택 (가로 2열 배치로 높이 대폭 절약)
years = ["전체"] + sorted(list(set(d[:4] for d in dates)), reverse=True)
col_y, col_d = st.sidebar.columns([0.85, 1.15])
with col_y:
    sel_year = st.selectbox("📅 연도", years, index=0)

filtered_dates = [d for d in dates if d.startswith(sel_year)] if sel_year != "전체" else dates
with col_d:
    selected_date = st.selectbox(
        f"📆 일자 ({len(filtered_dates):,}일)", 
        filtered_dates, 
        index=0,
        format_func=lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}"
    )

# 3열: 관측 분석 기상 변수 선택 (12종)
selected_var = st.sidebar.selectbox(
    "📊 기상 변수 선택 (12종)", 
    var_names, 
    format_func=lambda x: VAR_LABELS.get(x, x),
    index=0
)

# ==========================================
# 5. 상단 탭 구성 (관측보간 / 서리경보 / GNN수치예보)
# ==========================================
tab_weather, tab_frost, tab_forecast = st.tabs([
    "⛅ 270m 전국 기상 관측 보간 및 검증 (Observation & QC)",
    "❄️ 270m 농업기상 서리 조기경보 (XGBoost 앙상블)",
    "📡 270m 미세지형 수치예보 격자장 (GNN 다운스케일링)"
])

# -------------------------------------------------------------
# TAB 1: 기존 270m 기상 관측 보간 및 QC 대시보드
# -------------------------------------------------------------
with tab_weather:
    st.title(f"🌍 270m 전국 기상 관측 보간 분석 - {selected_date[:4]}-{selected_date[4:6]}-{selected_date[6:]}")
    st.markdown("---")
    
    col1, col2 = st.columns([1.2, 1])
    
    year = selected_date[:4]
    png_path = os.path.join(OUTPUT_DIR, "png", selected_var, year, f"{selected_var}_{selected_date}.png")
    tiff_path = os.path.join(OUTPUT_DIR, "tiff", selected_var, year, f"{selected_var}_{selected_date}.tiff")
    
    # 좌측: 270m 공간 격자도 & TIFF 다운로드
    with col1:
        st.subheader(f"🗺️ 270m 공간 격자도: {VAR_LABELS.get(selected_var, selected_var)}")
        if os.path.exists(png_path):
            img = Image.open(png_path)
            st.image(img, use_container_width=True)
        else:
            st.warning(f"{selected_var} PNG 이미지를 찾을 수 없습니다.")
            st.info(f"경로: `{png_path}`")
            
        if os.path.exists(tiff_path):
            file_size_mb = os.path.getsize(tiff_path) / (1024 * 1024)
            with open(tiff_path, "rb") as fp:
                st.download_button(
                    label=f"💾 {selected_var} 270m GeoTIFF 원본 다운로드 ({file_size_mb:.1f} MB)",
                    data=fp,
                    file_name=f"{selected_var}_{selected_date}.tiff",
                    mime="image/tiff",
                    use_container_width=True
                )
            
    # 우측: 검증 지표 및 산점도
    with col2:
        st.subheader("📊 검증 지표 (농진청 74개 Holdout 지점)")
        df_summary = load_daily_summary()
        df_val = load_validation_data(selected_date)
        
        rmse_col = f"{selected_var}_RMSE"
        mae_col = f"{selected_var}_MAE"
        rmse_val, mae_val = None, None
        if not df_summary.empty and rmse_col in df_summary.columns:
            day_row = df_summary[df_summary['Date'] == int(selected_date)]
            if not day_row.empty:
                r_v = day_row[rmse_col].values[0]
                m_v = day_row[mae_col].values[0] if mae_col in day_row.columns else None
                rmse_val = float(r_v) if pd.notnull(r_v) else None
                mae_val = float(m_v) if pd.notnull(m_v) else None
                
        # summary 파일에 없거나 NaN인 경우 개별 검증 CSV(df_val)로부터 동적 산출 (Fallback)
        if not df_val.empty:
            err_col = f"Error_{selected_var}"
            abs_err_col = f"AbsError_{selected_var}"
            if (mae_val is None or pd.isna(mae_val)) and abs_err_col in df_val.columns:
                valid_abs = df_val[abs_err_col].dropna()
                if not valid_abs.empty:
                    mae_val = float(valid_abs.mean())
            if (rmse_val is None or pd.isna(rmse_val)) and err_col in df_val.columns:
                valid_err = df_val[err_col].dropna()
                if not valid_err.empty:
                    rmse_val = float(np.sqrt(np.mean(valid_err**2)))
                
        unit = VAR_UNITS.get(selected_var, '')
        unit_str = f" {unit}" if unit else ""
        
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            val_str = f"{rmse_val:.3f}{unit_str}" if rmse_val is not None and not pd.isna(rmse_val) else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RMSE (오차 제곱근)</div>
                <div class="metric-value">{val_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            val_str = f"{mae_val:.3f}{unit_str}" if mae_val is not None and not pd.isna(mae_val) else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MAE (평균절대오차)</div>
                <div class="metric-value">{val_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            t_col_name = f"True_{selected_var}"
            if not df_val.empty and t_col_name in df_val.columns:
                v_cnt = df_val[t_col_name].notna().sum()
                tot_cnt = len(df_val)
                v_ratio = f"{v_cnt} / {tot_cnt}개"
            else:
                v_ratio = "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">유효 검증 관측소</div>
                <div class="metric-value">{v_ratio}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not df_val.empty:
            true_col = f"True_{selected_var}"
            pred_col = f"Pred_{selected_var}"
            if true_col in df_val.columns and pred_col in df_val.columns:
                df_plot = df_val[['Station_Name', true_col, pred_col]].dropna()
                if not df_plot.empty:
                    min_val = min(df_plot[true_col].min(), df_plot[pred_col].min())
                    max_val = max(df_plot[true_col].max(), df_plot[pred_col].max())
                    fig = px.scatter(
                        df_plot, x=true_col, y=pred_col, 
                        hover_name="Station_Name",
                        title=f"실측값 vs 모델 예측값 ({selected_var})",
                        color_discrete_sequence=['#4CAF50']
                    )
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val], 
                        mode='lines', name='Ideal (y=x)', 
                        line=dict(color='rgba(255,255,255,0.3)', dash='dash')
                    ))
                    fig.update_layout(
                        plot_bgcolor='#0E1117', 
                        paper_bgcolor='#0E1117',
                        font_color='#A0AAB5',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"💡 **{VAR_LABELS.get(selected_var, selected_var)}**은(는) 270m 공간 격자 보간 지도로 제공되며, 벡터 성분으로 품질 관리됩니다.")
                    
    st.markdown("---")
    
    # 중단: 오차 테이블 및 일별 RMSE 추이
    b1, b2 = st.columns(2)
    with b1:
        st.subheader("📋 관측소별 오차 테이블")
        if not df_val.empty:
            true_col = f'True_{selected_var}'
            pred_col = f'Pred_{selected_var}'
            err_col = f'Error_{selected_var}'
            abs_err_col = f'AbsError_{selected_var}'
            cols = ['Station_Name', true_col, pred_col, err_col, abs_err_col]
            cols = [c for c in cols if c in df_val.columns]
            if len(cols) > 1:
                df_table = df_val[cols].copy()
                total_stns = len(df_table)
                valid_stns = int(df_table[true_col].notna().sum()) if true_col in df_table.columns else 0
                missing_stns = total_stns - valid_stns

                col_chk, col_info = st.columns([1.6, 1.0])
                with col_chk:
                    filter_valid = st.checkbox("실제 관측값이 있는 지점만 표시 (결측 지점 제외)", value=True, key="chk_filter_valid_stns")
                with col_info:
                    st.caption(f"📍 유효: **{valid_stns}개** / 결측: **{missing_stns}개** (전체 {total_stns}개)")

                if filter_valid and true_col in df_table.columns:
                    df_table = df_table.dropna(subset=[true_col])

                if abs_err_col in df_table.columns:
                    df_table = df_table.sort_values(by=abs_err_col, ascending=False)

                unit_str = f" ({VAR_UNITS.get(selected_var, '')})" if VAR_UNITS.get(selected_var) else ""
                rename_map = {
                    'Station_Name': '관측소명',
                    true_col: f'실측값{unit_str}',
                    pred_col: f'예측값{unit_str}',
                    err_col: f'오차(예측-실측){unit_str}',
                    abs_err_col: f'절대오차{unit_str}'
                }
                df_table = df_table.rename(columns=rename_map)

                for num_c in [f'실측값{unit_str}', f'예측값{unit_str}', f'오차(예측-실측){unit_str}', f'절대오차{unit_str}']:
                    if num_c in df_table.columns:
                        df_table[num_c] = df_table[num_c].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")

                st.dataframe(df_table, use_container_width=True, height=350, hide_index=True)
            else:
                st.caption(f"{selected_var}의 개별 지점 실측값 테이블이 제공되지 않는 변수입니다.")
                
    with b2:
        st.subheader("📈 일별 모델 검증 오차 추이 (RMSE & MAE)")
        st.caption("📌 **가로축(X축)**: 실제 관측 날짜(연-월-일) | **세로축(Y축)**: 전국 74개 독립 검증지점(RDA) 오차 (파란색: RMSE, 녹색: MAE, 낮을수록 정밀)")
        if not df_summary.empty and rmse_col in df_summary.columns:
            df_s = df_summary.sort_values(by='Date').copy()
            df_s['Date_dt'] = pd.to_datetime(df_s['Date'].astype(str), format='%Y%m%d', errors='coerce')
            df_s = df_s.dropna(subset=['Date_dt', rmse_col])
            df_s['Date_display'] = df_s['Date_dt'].dt.strftime('%Y년 %m월 %d일')
            
            unit = VAR_UNITS.get(selected_var, '')
            unit_label = f" ({unit})" if unit else ""
            
            fig_t = go.Figure()
            
            # 1. RMSE 트레이스 (선명한 파란색 실선)
            fig_t.add_trace(go.Scatter(
                x=df_s['Date_dt'],
                y=df_s[rmse_col],
                mode='lines+markers',
                name='RMSE (오차 제곱근)',
                line=dict(color='#2196F3', width=2),
                marker=dict(size=4),
                customdata=df_s['Date_display'],
                hovertemplate="<b>날짜</b>: %{customdata}<br>" + f"<b>{selected_var} RMSE</b>: %{{y:.3f}}{unit_label}<extra></extra>"
            ))
            
            # 2. MAE 트레이스 (선명한 에메랄드 녹색 점선)
            has_mae = mae_col in df_s.columns and df_s[mae_col].notna().any()
            if has_mae:
                fig_t.add_trace(go.Scatter(
                    x=df_s['Date_dt'],
                    y=df_s[mae_col],
                    mode='lines+markers',
                    name='MAE (평균절대오차)',
                    line=dict(color='#00E676', width=2, dash='dot'),
                    marker=dict(size=4),
                    customdata=df_s['Date_display'],
                    hovertemplate="<b>날짜</b>: %{customdata}<br>" + f"<b>{selected_var} MAE</b>: %{{y:.3f}}{unit_label}<extra></extra>"
                ))
            
            # 레이아웃 및 범례(Legend) 설정
            fig_t.update_layout(
                title=f"📊 {selected_var} 일별 검증 오차 추이 (전체 {len(df_s)}일)",
                plot_bgcolor='#0E1117', 
                paper_bgcolor='#0E1117', 
                font_color='#A0AAB5',
                margin=dict(l=40, r=20, t=50, b=40),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#FFFFFF", size=11)
                )
            )
            
            # 가로축(X축: 날짜 타임라인) 명확화 및 기간 선택 버튼 추가
            fig_t.update_xaxes(
                title_text="📅 관측 일자 (Date: YYYY-MM-DD)",
                type="date",
                tickformat="%Y-%m-%d",
                tickangle=-25,
                showgrid=True,
                gridcolor="#262730",
                rangeselector=dict(
                    buttons=list([
                        dict(count=14, label="최근 14일", step="day", stepmode="backward"),
                        dict(count=1, label="최근 1개월", step="month", stepmode="backward"),
                        dict(count=3, label="최근 3개월", step="month", stepmode="backward"),
                        dict(count=1, label="최근 1년", step="year", stepmode="backward"),
                        dict(step="all", label="전체 기간")
                    ]),
                    bgcolor="#1E232F",
                    activecolor="#2196F3",
                    font=dict(color="#FFFFFF", size=10)
                )
            )
            
            # 세로축(Y축: 오차 단위 표시)
            fig_t.update_yaxes(
                title_text=f"검증 오차{unit_label}",
                showgrid=True,
                gridcolor="#262730"
            )
            
            # 현재 대시보드에서 선택된 날짜 강조 표시
            try:
                import datetime as dt_mod
                sel_dt = pd.to_datetime(selected_date, format='%Y%m%d')
                sel_row = df_s[df_s['Date_dt'] == sel_dt]
                if not sel_row.empty:
                    sel_rmse = float(sel_row[rmse_col].values[0]) if rmse_col in sel_row.columns and pd.notnull(sel_row[rmse_col].values[0]) else None
                    sel_mae = float(sel_row[mae_col].values[0]) if mae_col in sel_row.columns and pd.notnull(sel_row[mae_col].values[0]) else None
                    
                    # 기본 표시 범위를 선택일 전후 30일로 설정하여 가로축을 한눈에 읽을 수 있게 함
                    r_start = sel_dt - dt_mod.timedelta(days=25)
                    r_end = sel_dt + dt_mod.timedelta(days=7)
                    fig_t.update_xaxes(range=[r_start, r_end])
                    
                    # 선택일 텍스트 구성
                    ann_text = f"선택일({sel_dt.strftime('%m-%d')}): RMSE {sel_rmse:.3f}" if sel_rmse is not None else f"선택일: {sel_dt.strftime('%m-%d')}"
                    if sel_mae is not None:
                        ann_text += f", MAE {sel_mae:.3f}{unit}"
                    elif unit:
                        ann_text += f"{unit}"
                    
                    # 선택일 세로 기준선 표시
                    fig_t.add_vline(
                        x=sel_dt,
                        line_dash="dash",
                        line_color="#FFA726",
                        line_width=2,
                        annotation_text=ann_text,
                        annotation_position="top left",
                        annotation_font=dict(color="#FFA726", size=11),
                        annotation_bgcolor="#111827"
                    )
                    # 선택일 강조 마커 포인트 (RMSE)
                    if sel_rmse is not None:
                        fig_t.add_scatter(
                            x=[sel_dt],
                            y=[sel_rmse],
                            mode='markers',
                            marker=dict(color='#FFA726', size=11, symbol='star', line=dict(color='#FFFFFF', width=1.5)),
                            name=f"선택일 RMSE ({sel_dt.strftime('%Y-%m-%d')})",
                            hoverinfo='skip'
                        )
                    # 선택일 강조 마커 포인트 (MAE)
                    if sel_mae is not None:
                        fig_t.add_scatter(
                            x=[sel_dt],
                            y=[sel_mae],
                            mode='markers',
                            marker=dict(color='#00E676', size=9, symbol='diamond', line=dict(color='#FFFFFF', width=1)),
                            name=f"선택일 MAE ({sel_dt.strftime('%Y-%m-%d')})",
                            hoverinfo='skip'
                        )
            except Exception:
                pass
                
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("해당 변수는 시계열 RMSE/MAE 요약 컬럼이 없거나 단일 성분으로 집계됩니다.")
            
    # 하단: 센서 품질관리(QC) 및 이상치 탐지 리포트
    st.markdown("---")
    df_anomaly = load_anomaly_report(selected_date)
    n_anom = len(df_anomaly)
    exp_title = f"⚠️ 당일 관측 센서 품질관리(QC) 및 이상치 탐지 내역 ({n_anom}건 검출)" if n_anom > 0 else "✅ 당일 관측 센서 품질관리(QC) - 이상 없음 (0건)"
    
    with st.expander(exp_title, expanded=(n_anom > 0)):
        if n_anom > 0:
            aq1, aq2, aq3 = st.columns(3)
            with aq1:
                st.metric("총 이상 검출 센서 수", f"{n_anom}건")
            with aq2:
                rda_cnt = (df_anomaly['Source'] == 'RDA').sum() if 'Source' in df_anomaly.columns else 0
                st.metric("농진청(RDA) 지점 이상", f"{rda_cnt}건")
            with aq3:
                kma_cnt = (df_anomaly['Source'] == 'KMA').sum() if 'Source' in df_anomaly.columns else 0
                st.metric("기상청(KMA) 지점 이상", f"{kma_cnt}건")
                
            cols_show = ['Source', 'STN', 'Name', 'Variable', 'Value', 'Reason', 'Type']
            cols_exist = [c for c in cols_show if c in df_anomaly.columns]
            df_anom_disp = df_anomaly[cols_exist].copy().rename(columns={
                'Source': '기관',
                'STN': '지점번호',
                'Name': '관측소명',
                'Variable': '해당변수',
                'Value': '측정값(이상치)',
                'Reason': '검출 사유(QC)',
                'Type': '오류 유형'
            })
            st.dataframe(df_anom_disp, use_container_width=True, hide_index=True)
        else:
            st.success("✅ 해당 일자 관측 데이터에서 센서 단선, 전압 불량, 물리적 허용범위 초과 등 오류가 검출되지 않았습니다. (정상 관측 데이터 사용)")

# -------------------------------------------------------------
# TAB 2: 270m 서리 조기경보 (XGBoost 앙상블)
# -------------------------------------------------------------
with tab_frost:
    st.title("❄️ 전국 270m 농업기상 서리 조기경보 (XGBoost 앙상블)")
    st.markdown("5km 동네예보 + 지형 GNN 270m 미세지형 예보 격자를 기반으로 270m 필지 단위 서리 위험을 3단계로 조기 판별합니다.")
    st.markdown("---")
    
    fcol1, fcol2, fcol3 = st.columns([1.5, 1, 1.5])
    with fcol1:
        sel_frost_date = st.selectbox("📅 서리 예보 기준일 선택", dates, key="frost_date_sel", index=0)
    with fcol2:
        sel_lead_day = st.selectbox("🗺️ 전국 지도 표출 시점", [1, 2, 3], format_func=lambda x: f"D+{x} ({'내일' if x==1 else '모레' if x==2 else '글피'})", key="frost_lead_sel")
    with fcol3:
        st.write("")
        st.write("")
        btn_run = st.button("⚡ 3일간(D+1~D+3) 서리 예측 일괄 생성", use_container_width=True, key="btn_run_frost")
        
    if btn_run:
        with st.spinner(f"{sel_frost_date} 기준 D+1 ~ D+3 (3일간) 서리 예측 연산 실행 중..."):
            import subprocess
            _xgb_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_frost_predictor.py")
            for l_day in [1, 2, 3]:
                cmd = [sys.executable, _xgb_script, sel_frost_date, str(l_day)]
                subprocess.run(cmd, capture_output=True, text=True)
            st.success("3일간(D+1~D+3) 서리 예측 및 지도가 모두 생성되었습니다!")
            st.cache_data.clear()
                
    year_f = sel_frost_date[:4]
    png_f_path = os.path.join(FROST_DIR, year_f, f"XGB_Frost_Risk_{sel_frost_date}_D{sel_lead_day}.png")
    tiff_f_path = os.path.join(FROST_DIR, year_f, f"XGB_Frost_Grade_{sel_frost_date}_D{sel_lead_day}.tiff")
    
    col_map_f, col_info_f = st.columns([1.2, 1])
    
    with col_map_f:
        st.subheader("🗺️ 전국 270m 서리 조기경보 지도")
        if os.path.exists(png_f_path):
            img_f = Image.open(png_f_path)
            st.image(img_f, use_container_width=True)
            
            if os.path.exists(tiff_f_path):
                f_size_mb = os.path.getsize(tiff_f_path) / (1024 * 1024)
                with open(tiff_f_path, "rb") as fp:
                    st.download_button(
                        label=f"💾 270m 서리 판별 GeoTIFF 다운로드 ({f_size_mb:.1f} MB)",
                        data=fp,
                        file_name=f"XGB_Frost_Grade_{sel_frost_date}_D{sel_lead_day}.tiff",
                        mime="image/tiff",
                        use_container_width=True
                    )
        else:
            st.warning(f"{sel_frost_date} (D+{sel_lead_day})의 서리 예측 결과가 아직 생성되지 않았습니다.")
            st.info("우측 상단의 [⚡ 해당 날짜 서리 예측 생성/갱신] 버튼을 누르면 즉시 연산 및 지도가 생성됩니다.")
            
    with col_info_f:
        st.subheader("📊 270m 서리 위험 통계 및 농가 대응 요령")
        
        if os.path.exists(tiff_f_path):
            import tifffile as tiff
            arr = tiff.imread(tiff_f_path)
            valid_arr = arr[arr != 255]
            n_tot = len(valid_arr)
            n_safe = int(np.sum(valid_arr == 0))
            n_watch = int(np.sum(valid_arr == 1))
            n_warn = int(np.sum(valid_arr == 2))
            
            p_safe = (n_safe / n_tot) * 100 if n_tot else 0
            p_watch = (n_watch / n_tot) * 100 if n_tot else 0
            p_warn = (n_warn / n_tot) * 100 if n_tot else 0
        else:
            n_safe, n_watch, n_warn = 0, 0, 0
            p_safe, p_watch, p_warn = 0.0, 0.0, 0.0
            
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚨 경고 (Warning)</div>
                <div class="metric-value frost-warning">{p_warn:.1f}%</div>
                <small>{n_warn:,}개 격자</small>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⚠️ 주의 (Watch)</div>
                <div class="metric-value frost-watch">{p_watch:.1f}%</div>
                <small>{n_watch:,}개 격자</small>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✅ 안전 (Safe)</div>
                <div class="metric-value frost-safe">{p_safe:.1f}%</div>
                <small>{n_safe:,}개 격자</small>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        #### 📋 앙상블 판별 기준 및 대응 수칙
        * **🚨 경고 (둘 다 0)**:
          - **서리 발생 확실 (확률 90% 이상)**
          - **행동 요령**: 과수원 방상팬(송풍법) 및 미세살수기(살수빙결법) 전면 가동, 저온 취약 작물 보온 피복재 덮기
        * **⚠️ 주의 (둘 중 하나만 0)**:
          - **국지 서리 위험 (찬 공기 정체 구역)**
          - **행동 요령**: 산골짜기, 저지대, 하천 인근 과수원 예찰 강화 및 방제 장비 사전 점검
        * **✅ 안전 (둘 다 1)**:
          - **서리 미발생 (정상 영농 활동)**
        """)
        
        st.info("💡 **모델 구성**: 270m GNN 최저기온(tmin), 풍속(ws), 습도(hu), 강수량(pre) + 5km 대기 운량(clo)을 `xgboost_model_A.pkl`과 `xgboost_model_B.pkl`에 입력하여 도출합니다.")
        
    st.markdown("---")
    st.subheader("📍 주요 거점별 3일간(D+1~D+3) 서리 판별 & 270m 기상 예보 일괄 표출")
    st.markdown("청주시 상당구 미원면, 국립농업과학원, 익산시 월성동 3개 지점의 **D+1(내일), D+2(모레), D+3(글피)** 270m 고해상도 수치예보 기상장 및 서리 예측 결과입니다.")
    
    df_pts = get_point_frost_weather_data(sel_frost_date)
    if df_pts is not None and not df_pts.empty:
        st.dataframe(df_pts, use_container_width=True, hide_index=True)
        st.caption("💡 표 헤더(지점명, 예보시점, 예보일자, 최저기온 등)를 클릭하여 원하는 기준으로 오름차순/내림차순 정렬할 수 있습니다.")
    else:
        st.warning(f"선택하신 {sel_frost_date} 기준의 서리 및 기상 예보 격자가 아직 생성되지 않았습니다.")
        st.info("우측 상단의 [⚡ 3일간(D+1~D+3) 서리 예측 일괄 생성] 버튼을 누르면 10~15초 내에 3일치 연산 후 표가 즉시 채워집니다.")

# -------------------------------------------------------------
# TAB 3: 270m 미세지형 수치예보 기상장 (GNN 다운스케일링)
# -------------------------------------------------------------
with tab_forecast:
    st.title("📡 270m 전국 미세지형 수치예보 격자장 (GNN 다운스케일링)")
    st.markdown("기상청 5km 동네예보를 270m 수치표고모델(DEM), 경사도, 향, 동적 감률(Lapse rate)을 반영하여 134만개 격자로 공간 복원한 고해상도 예보장입니다.")
    st.markdown("---")
    
    fc_vars = ['Tmin', 'Tmax', 'Tavg', 'Precipitation', 'Humidity', 'Wind_10m', 'Solar_Rad', 'Sunshine']
    
    fc1, fc2, fc3, fc4 = st.columns([1.2, 1, 1.2, 1.2])
    with fc1:
        sel_fc_date = st.selectbox("📅 예보 기준일 (Base Date)", dates, key="fc_date_sel", index=0)
    with fc2:
        sel_fc_lead = st.selectbox("⏱️ 예보 시점", [1, 2, 3], format_func=lambda x: f"D+{x} ({'내일' if x==1 else '모레' if x==2 else '글피'})", key="fc_lead_sel")
    with fc3:
        sel_fc_var = st.selectbox("🌡️ 예보 기상 변수", fc_vars, format_func=lambda x: VAR_LABELS.get(x, x), key="fc_var_sel")
    with fc4:
        st.write("")
        st.write("")
        btn_run_gnn = st.button("⚡ 270m 예보장 즉시 생성 (GNN)", use_container_width=True, key="btn_run_gnn")
        
    if btn_run_gnn:
        with st.spinner(f"{sel_fc_date} (D+{sel_fc_lead}) 270m 전 기상변수 GNN 다운스케일링 실행 중..."):
            import subprocess
            cmd = [sys.executable, "-c", f"from downscale_pipeline import predict_and_generate_grid; predict_and_generate_grid('{sel_fc_date}', lead_day={sel_fc_lead})"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                st.success("270m 미세지형 수치예보 격자장 생성이 완료되었습니다!")
                st.cache_data.clear()
            else:
                st.error(f"예측 실패: {res.stderr}")
                
    fc_year = sel_fc_date[:4]
    fc_tiff_path = os.path.join(FORECAST_DIR, sel_fc_var, fc_year, f"{sel_fc_var}_forecast_{sel_fc_date}_D{sel_fc_lead}.tiff")
    
    col_fc_map, col_fc_info = st.columns([1.2, 1])
    
    with col_fc_map:
        st.subheader(f"🗺️ 270m 예보 격자도: {VAR_LABELS.get(sel_fc_var, sel_fc_var)}")
        if os.path.exists(fc_tiff_path):
            img_bytes, stats = render_forecast_map(fc_tiff_path, sel_fc_var)
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
                
            fc_size_mb = os.path.getsize(fc_tiff_path) / (1024 * 1024)
            with open(fc_tiff_path, "rb") as fp:
                st.download_button(
                    label=f"💾 {sel_fc_var} 270m 예보 GeoTIFF 다운로드 ({fc_size_mb:.1f} MB)",
                    data=fp,
                    file_name=f"{sel_fc_var}_forecast_{sel_fc_date}_D{sel_fc_lead}.tiff",
                    mime="image/tiff",
                    use_container_width=True
                )
        else:
            st.warning(f"{sel_fc_date} (D+{sel_fc_lead}) {sel_fc_var} 270m 예보 격자가 아직 생성되지 않았습니다.")
            st.info("우측 상단의 [⚡ 270m 예보장 즉시 생성 (GNN)] 버튼을 누르면 5~10초 내에 8개 전 기상변수 예보 격자가 일괄 생성됩니다.")
            
    with col_fc_info:
        st.subheader("📊 270m 예보장 전국 통계 지표")
        if os.path.exists(fc_tiff_path):
            _, stats = render_forecast_map(fc_tiff_path, sel_fc_var)
            unit_s = VAR_UNITS.get(sel_fc_var, '')
            if stats:
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">전국 최저값</div>
                        <div class="metric-value">{stats['min']:.2f}</div>
                        <small>{unit_s}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with f2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">전국 평균값</div>
                        <div class="metric-value">{stats['mean']:.2f}</div>
                        <small>{unit_s}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with f3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">전국 최고값</div>
                        <div class="metric-value">{stats['max']:.2f}</div>
                        <small>{unit_s}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            #### 📌 GNN 공간 다운스케일링 원리
            * **기상청 5km 동네예보 격자**: 전국 37,697개 격자 중 유효 4,292개 셀 입력
            * **270m 미세지형 복원**: 전국 1,341,592개 유효 필지(Land mask) 격자 예측
            * **물리 법칙 보호**: 동적 기온 감률 공식(`0.0098 - (RH/100)*0.0048`)을 적용하여 고도 및 상대습도에 따른 국지 냉기 침강(Cold air drainage) 및 야간 복사냉각 효과를 정밀하게 재현합니다.
            """)
        else:
            st.info("예보장 생성 후 통계 및 분포 지표가 이곳에 표출됩니다.")

# ==========================================
# 6. 실시간 자동 감지 주기적 갱신
# ==========================================
if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()
