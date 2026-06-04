"""
AI Pedestrian Accessibility Route Planner — Version 10.0 (Ultimate Production)
ตรงตามคำแนะนำบนแผ่นโน้ตข้อกำหนด และรองรับโครงสร้างไฟล์บน GitHub จริง 100%
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import warnings
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from folium.plugins import MiniMap

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Accessibility Route Planner V10", layout="wide", page_icon="♿")

# ─── HEADER DESIGN ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.custom-header {
    background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.7)),
        url("https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?semt=ais_hybrid&w=740&q=80");
    background-size:cover; background-position:center;
    padding:40px; border-radius:14px; color:white; text-align:center; margin-bottom:20px;
}
.custom-header h1 { color:#fff !important; font-size:2.4rem !important; text-shadow:2px 2px 8px rgba(0,0,0,0.8); }
.ai-badge { background:#e74c3c; color:white; padding:6px 16px; border-radius:20px; font-size:0.9rem; font-weight:bold; display:inline-block; margin-top:8px; }
</style>
<div class="custom-header">
  <h1>♿ AI Pedestrian Accessibility Route Planner (V10.0)</h1>
  <h3>ระบบวิเคราะห์เส้นทางฟุตบาทคนเดินด้วยปัญญาประดิษฐ์ชั้นสูงและการทดสอบสภาวะวิกฤต</h3>
  <span class="ai-badge">🔬 Pure Pedestrian Model & Hardcore Stress Test Activated</span>
</div>
""", unsafe_allow_html=True)

# ─── VECTORIZED PEDESTRIAN GEOMETRY ──────────────────────────────────────────
def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000.0  # รัศมีโลกเป็นเมตร (Dedicated Pedestrian Resolution)
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2 = np.radians(lat2_arr)
    lo2 = np.radians(lon2_arr)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ─── LOAD DATA EXACTLY MATCHING GITHUB ───────────────────────────────────────
@st.cache_data
def load_and_precompute_v10():
    base = "."
    
    # ดึงไฟล์ตรงตามชื่อจริงบน GitHub repository ของคุณเป๊ะๆ
    df_places    = pd.read_csv(f"{base}/bangkok_places_bus_spot.csv")
    df_stations  = pd.read_csv(f"{base}/bts_station.csv")
    df_acc       = pd.read_csv(f"{base}/BTS for wheelchair users spreadsheet - BTS green line.csv")
    df_bus_stops = pd.read_csv(f"{base}/bangkok_bus_stops_coordinates.csv")
    df_smalie    = pd.read_csv(f"{base}/ThaiSmalieBus - Sheet1.csv")
    df_rf        = pd.read_csv(f"{base}/wheelchair_random_forest_300rows.csv")

    # คลีนข้อมูลและเชื่อมโยงโครงข่ายทางเท้าและสถานี
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี','').str.strip()
    df_acc['clean_name']      = df_acc['สถานี'].str.replace('สถานี','').str.strip()

    df_bts = pd.merge(
        df_acc, 
        df_stations[['clean_name','lat','lng','btsline','location']], 
        on='clean_name', 
        how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)

    bts_lats, bts_lons = df_bts['lat'].values, df_bts['lng'].values
    bus_lats, bus_lons = df_bus_stops['latitude'].values, df_bus_stops['longitude'].values

    # คำนวณความเชื่อมโยงของโครงข่ายทางเท้าโดยไม่พึ่งพาถนนซุปเปอร์เวย์
    display_names, nearest_bts_idx, nearest_bus_dist = [], [], []
    for _, row in df_places.iterrows():
        lat, lon = row['latitude'], row['longitude']
        d_bts = haversine_vec(lat, lon, bts_lats, bts_lons)
        idx_b = int(np.argmin(d_bts))
        nearest_bts_idx.append(idx_b)
        
        d_bus = haversine_vec(lat, lon, bus_lats, bus_lons)
        nearest_bus_dist.append(float(np.min(d_bus)))
        display_names.append(row['place_name'])

    df_places['display_th'] = display_names
    df_places['_bts_idx'] = nearest_bts_idx
    df_places['_min_bus_dist'] = nearest_bus_dist

    # สร้างข้อมูลความหนาแน่นจำลองเชื่อมโยงกับโครงข่ายรถบัสไฟฟ้าขนาดเล็กเพื่อทางเดินเท้า
    crowd_map = {}
    for stn in df_bts['clean_name']:
        crowd_map[stn] = np.random.choice([1, 2, 3])

    return df_places, df_bts, df_bus_stops, df_smalie, df_rf, bts_lats, bts_lons, bus_lats, bus_lons, crowd_map

# รันฟังก์ชันโหลดข้อมูลที่มีความเสถียรสูง
df_places, df_bts, df_bus_stops, df_smalie, df_rf, bts_lats, bts_lons, bus_lats, bus_lons, crowd_map = load_and_precompute_v10()

# ─── AI MODELS TRAINING ──────────────────────────────────────────────────────
@st.cache_resource
def train_ai_cores_v10(df_rf):
    le = LabelEncoder()
    df = df_rf.copy()
    df['Transport_Type_enc'] = le.fit_transform(df['Transport_Type'])
    feats = ['Elevator','Ramp','Accessible_Exit','Cost','Travel_Time','BusSupport','Safety','Crowded_Level','Urgency','Prefer_Safe','Prefer_Cheap','Transport_Type_enc']
    
    clf1 = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
    clf1.fit(df[feats], df['Recommended'])

    # โมเดลประเมินความปลอดภัยสิ่งกีดขวางเฉพาะทางเดินเท้า (Footpath Viability Intelligence)
    X2 = np.array([
        [1.5,1,0,1],[0.7,0,3,0],[1.2,1,1,1],[1.8,1,0,1],[0.8,0,4,0],
        [2.0,1,0,1],[0.5,0,6,0],[1.4,1,1,1],[0.9,0,3,0],[2.5,1,0,1]
    ])
    y2 = np.array([1,0,1,1,0,1,0,1,0,1])
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf2.fit(X2, y2)
    return clf1, clf2, le, feats

route_rf, sidewalk_rf, le_transport, rf_features = train_ai_cores_v10(df_rf)

# ─── SIDEBAR INTERFACE ───────────────────────────────────────────────────────
st.sidebar.header("🕹️ AI Route Control Panel")
place_list = sorted(df_places['display_th'].tolist())
start_p = st.sidebar.selectbox("📍 Origin Pedestrian Node (จุดเริ่มต้นเท้า):", place_list, index=0)
end_p = st.sidebar.selectbox("🏁 Destination Pedestrian Node (จุดปลายทางเท้า):", place_list, index=min(1, len(place_list)-1))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🦮 Footpath Geometry Simulator")
sw_width = st.sidebar.slider("ความกว้างของหน้าฟุตบาท (เมตร)", 0.3, 3.0, 1.5, step=0.1)
sw_surface = st.sidebar.selectbox("สภาพพื้นผิวทางเดินเท้า:", ["Smooth Concrete (ทางเรียบสมบูรณ์)", "Broken/Uneven Pavement (อิฐบล็อกแตก/ขรุขระ)"])
sw_obstacles = st.sidebar.slider("จำนวนสิ่งกีดขวางทางกายภาพ (เสาไฟ/ป้าย/ร้านค้า)", 0, 15, 2)
sw_ramp = st.sidebar.checkbox("มีทางลาดเชื่อมต่อฝั่งตรงข้าม (Curb Ramp)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Harder Error Testing Suite")
st.sidebar.caption("ป้อนอุปสรรคขั้นรุนแรงเพื่อทดสอบความเสถียรของเทคนิค AI (The Harder, The Better)")
err_elevator_break = st.sidebar.checkbox("🚨 Force Complete Elevator Breakdown (ลิฟต์เสียระบบปิด)", value=False)
err_flash_flood = st.sidebar.checkbox("🌧️ Flash Flood on Footpath (น้ำท่วมขังมิดฟุตบาท)", value=False)
err_gridlock_surge = st.sidebar.checkbox("👨‍👩‍👧‍👦 Extreme Human Crowd Surge (+300%)", value=False)

# ─── CORE MATHEMATICAL INTERSECTION ─────────────────────────────────────────
s_row = df_places[df_places['display_th'] == start_p].iloc[0]
e_row = df_places[df_places['display_th'] == end_p].iloc[0]
slat, slon = s_row['latitude'], s_row['longitude']
elat, elon = e_row['latitude'], e_row['longitude']

d_s = haversine_vec(slat, slon, bts_lats, bts_lons)
bts_s = df_bts.iloc[int(np.argmin(d_s))]
d_e = haversine_vec(elat, elon, bts_lats, bts_lons)
bts_e = df_bts.iloc[int(np.argmin(d_e))]

# ประมวลผลลัพธ์ภายใต้ Error Testing Constraints
el_val = 0 if err_elevator_break else (1 if str(bts_s.get('มีลิฟต์','')) in ['1','มี'] else 0)
rmp_val = 0 if err_flash_flood else (1 if str(bts_s.get('ทางลาดสำหรับรถเข็น','')) in ['1','มี'] else 0)
acc_exit = 1 if (el_val and rmp_val) else 0
crowd_lvl = 4 if err_gridlock_surge else crowd_map.get(str(bts_s.get('clean_name','')), 2)

# รันโมเดล AI
