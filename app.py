"""
AI Pedestrian Accessibility Route Planner — Version 10.1 (Resilient Production)
แก้ไขปัญหาระบบหาไฟล์ ThaiSmalieBus ไม่เจอด้วย Smart Keyword Search Engine
ตรงตามข้อกำหนดเส้นทางเท้าล้วน (Pedestrian Only) และระบบทดสอบสภาวะวิกฤต (Stress Test)
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
  <h1>♿ AI Pedestrian Accessibility Route Planner (V10.1)</h1>
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

# ─── LOAD DATA WITH SMART KEYWORD SEARCH ─────────────────────────────────────
@st.cache_data
def load_and_precompute_v10():
    def smart_load_csv(keywords, default_name):
        """ค้นหาไฟล์แบบยืดหยุ่นสูงตามคำสำคัญ ป้องกันปัญหาพิมพ์ชื่อไฟล์ผิดหรือเว้นวรรคไม่ตรงกันบน GitHub"""
        search_dirs = [".", "data", "Data", "/mnt/user-data/uploads"]
        
        # 1. ลองค้นหาแบบชื่อตรงตัวเป๊ะๆ ก่อน
        for d in search_dirs:
            path = os.path.join(d, default_name)
            if os.path.exists(path):
                return pd.read_csv(path)
        
        # 2. ค้นหาแบบสแกนหา Keywords ทั่วโปรเจกต์ (Case-Insensitive)
        for d in search_dirs:
            if os.path.exists(d):
                for root, dirs, files in os.walk(d):
                    for f in files:
                        if f.endswith('.csv'):
                            for kw in keywords:
                                if kw.lower() in f.lower():
                                    return pd.read_csv(os.path.join(root, f))
                                    
        raise FileNotFoundError(f"Missing required file matching keywords {keywords} (Expected: {default_name})")

    try:
        df_places    = smart_load_csv(["places_bus_spot", "places"], "bangkok_places_bus_spot.csv")
        df_stations  = smart_load_csv(["bts_station", "station"], "bts_station.csv")
        df_acc       = smart_load_csv(["green", "wheelchair", "spreadsheet"], "BTS for wheelchair users spreadsheet - BTS green line.csv")
        df_bus_stops = smart_load_csv(["bus_stops", "coordinates"], "bangkok_bus_stops_coordinates.csv")
        df_smalie    = smart_load_csv(["smalie", "smile", "sheet1"], "ThaiSmalieBus - Sheet1.csv")
        df_rf        = smart_load_csv(["random_forest", "300rows"], "wheelchair_random_forest_300rows.csv")
        
    except Exception as e:
        # 🧪 ระบบวินิจฉัยความผิดพลาดอัตโนมัติ (Self-Diagnostic Fallback)
        st.error("❌ ไม่พบไฟล์ข้อมูลสำคัญบางไฟล์ในระบบ GitHub Repository ของคุณ")
        st.markdown("### 🔍 แผงวินิจฉัยโครงสร้างไฟล์อัตโนมัติ (Diagnostic Panel):")
        st.warning("กรุณาตรวจสอบว่าชื่อไฟล์บน GitHub ของคุณตรงกับรายการที่คาดหวังหรือไม่")
        
        csv_found = []
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith('.csv'):
                    csv_found.append(os.path.join(root, f))
                    
        if csv_found:
            st.write("📋 **ไฟล์ข้อมูลสัญกรณ์ `.csv` ที่พบบน Repository ของคุณในตอนนี้:**")
            for path in csv_found:
                st.code(path)
        else:
            st.error("⚠️ ไม่พบไฟล์นามสกุล `.csv` ใดๆ เลยในโปรเจกต์ของคุณ กรุณาอัปโหลดไฟล์ข้อมูลขึ้นสู่ GitHub")
            
        st.info(f"ข้อความแจ้งเตือนจากระบบส่วนลึก: `{str(e)}`")
        st.stop()

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

# รันฟังก์ชันโหลดข้อมูลที่มีความเสถียรสูงด้วย Smart Search
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
sw_surface = st
