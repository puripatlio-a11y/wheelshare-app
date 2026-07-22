from collections import deque
import os
import warnings
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ─── 0. STREAMLIT CONFIGURATION ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Wheelchair & BTS Google Maps Navigator",
    layout="wide",
    page_icon="🗺️",
)

# ─── 1. BTS STATIONS COORDINATES & GRAPH NETWORK ─────────────────────────
BTS_STATIONS = {
    # ===== Sukhumvit Line (สายสุขุมวิท) =====
    "คูคต": [13.9607, 100.6256],
    "แยก คปอ.": [13.9495, 100.6225],
    "พิพิธภัณฑ์กองทัพอากาศ": [13.9384, 100.6184],
    "โรงพยาบาลภูมิพลอดุลยเดช": [13.9257, 100.6095],
    "สะพานใหม่": [13.9132, 100.6034],
    "สายหยุด": [13.9022, 100.5964],
    "พหลโยธิน 59": [13.8907, 100.5920],
    "วัดพระศรีมหาธาตุ": [13.8758, 100.5965],
    "กรมทหารราบที่ 11": [13.8730, 100.5960],
    "บางบัว": [13.8632, 100.6043],
    "กรมป่าไม้": [13.8506, 100.6049],
    "มหาวิทยาลัยเกษตรศาสตร์": [13.8462, 100.5697],
    "เสนานิคม": [13.8420, 100.5717],
    "รัชโยธิน": [13.8305, 100.5685],
    "พหลโยธิน 24": [13.8245, 100.5664],
    "ห้าแยกลาดพร้าว": [13.8163, 100.5616],
    "หมอชิต": [13.8026, 100.5538],
    "สะพานควาย": [13.7937, 100.5495],
    "อารีย์": [13.7797, 100.5447],
    "สนามเป้า": [13.7726, 100.5420],
    "อนุสาวรีย์ชัยสมรภูมิ": [13.7628, 100.5371],
    "พญาไท": [13.7569, 100.5335],
    "ราชเทวี": [13.7518, 100.5316],
    "สยาม": [13.7466, 100.5346],
    "ชิดลม": [13.7440, 100.5440],
    "เพลินจิต": [13.7430, 100.5498],
    "นานา": [13.7406, 100.5550],
    "อโศก": [13.7370, 100.5605],
    "พร้อมพงษ์": [13.7305, 100.5696],
    "ทองหล่อ": [13.7243, 100.5788],
    "เอกมัย": [13.7192, 100.5853],
    "พระโขนง": [13.7152, 100.5918],
    "อ่อนนุช": [13.7057, 100.6010],
    "บางจาก": [13.6969, 100.6053],
    "ปุณณวิถี": [13.6892, 100.6094],
    "อุดมสุข": [13.6803, 100.6107],
    "บางนา": [13.6688, 100.6047],
    "แบริ่ง": [13.6614, 100.6012],
    "สำโรง": [13.6462, 100.5952],
    # ===== Silom Line (สายสีลม) =====
    "สนามกีฬาแห่งชาติ": [13.7460, 100.5290],
    "ราชดำริ": [13.7394, 100.5394],
    "ศาลาแดง": [13.7286, 100.5343],
    "ช่องนนทรี": [13.7235, 100.5293],
    "เซนต์หลุยส์": [13.7208, 100.5263],
    "สุรศักดิ์": [13.7187, 100.5217],
    "สะพานตากสิน": [13.7185, 100.5142],
    "กรุงธนบุรี": [13.7208, 100.5026],
    "วงเวียนใหญ่": [13.7210, 100.4956],
    "บางหว้า": [13.7203, 100.4570],
}

BTS_GRAPH = {
    # Sukhumvit Line
    "คูคต": ["แยก คปอ."],
    "แยก คปอ.": ["คูคต", "พิพิธภัณฑ์กองทัพอากาศ"],
    "พิพิธภัณฑ์กองทัพอากาศ": ["แยก คปอ.", "โรงพยาบาลภูมิพลอดุลยเดช"],
    "โรงพยาบาลภูมิพลอดุลยเดช": ["พิพิธภัณฑ์กองทัพอากาศ", "สะพานใหม่"],
    "สะพานใหม่": ["โรงพยาบาลภูมิพลอดุลยเดช", "สายหยุด"],
    "สายหยุด": ["สะพานใหม่", "พหลโยธิน 59"],
    "พหลโยธิน 59": ["สายหยุด", "วัดพระศรีมหาธาตุ"],
    "วัดพระศรีมหาธาตุ": ["พหลโยธิน 59", "กรมทหารราบที่ 11"],
    "กรมทหารราบที่ 11": ["วัดพระศรีมหาธาตุ", "บางบัว"],
    "บางบัว": ["กรมทหารราบที่ 11", "กรมป่าไม้"],
    "กรมป่าไม้": ["บางบัว", "มหาวิทยาลัยเกษตรศาสตร์"],
    "มหาวิทยาลัยเกษตรศาสตร์": ["กรมป่าไม้", "เสนานิคม"],
    "เสนานิคม": ["มหาวิทยาลัยเกษตรศาสตร์", "รัชโยธิน"],
    "รัชโยธิน": ["เสนานิคม", "พหลโยธิน 24"],
    "พหลโยธิน 24": ["รัชโยธิน", "ห้าแยกลาดพร้าว"],
    "ห้าแยกลาดพร้าว": ["พหลโยธิน 24", "หมอชิต"],
    "หมอชิต": ["ห้าแยกลาดพร้าว", "สะพานควาย"],
    "สะพานควาย": ["หมอชิต", "อารีย์"],
    "อารีย์": ["สะพานควาย", "สนามเป้า"],
    "สนามเป้า": ["อารีย์", "อนุสาวรีย์ชัยสมรภูมิ"],
    "อนุสาวรีย์ชัยสมรภูมิ": ["สนามเป้า", "พญาไท"],
    "พญาไท": ["อนุสาวรีย์ชัยสมรภูมิ", "ราชเทวี"],
    "ราชเทวี": ["พญาไท", "สยาม"],
    # Siam Station Node (4-Way Interchange Junction)
    "สยาม": ["ราชเทวี", "ชิดลม", "สนามกีฬาแห่งชาติ", "ราชดำริ"],
    "ชิดลม": ["สยาม", "เพลินจิต"],
    "เพลินจิต": ["ชิดลม", "นานา"],
    "นานา": ["เพลินจิต", "อโศก"],
    "อโศก": ["นานา", "พร้อมพงษ์"],
    "พร้อมพงษ์": ["อโศก", "ทองหล่อ"],
    "ทองหล่อ": ["พร้อมพงษ์", "เอกมัย"],
    "เอกมัย": ["ทองหล่อ", "พระโขนง"],
    "พระโขนง": ["เอกมัย", "อ่อนนุช"],
    "อ่อนนุช": ["พระโขนง", "บางจาก"],
    "บางจาก": ["อ่อนนุช", "ปุณณวิถี"],
    "ปุณณวิถี": ["บางจาก", "อุดมสุข"],
    "อุดมสุข": ["ปุณณวิถี", "บางนา"],
    "บางนา": ["อุดมสุข", "แบริ่ง"],
    "แบริ่ง": ["บางนา", "สำโรง"],
    "สำโรง": ["แบริ่ง"],
    # Silom Line
    "สนามกีฬาแห่งชาติ": ["สยาม"],
    "ราชดำริ": ["สยาม", "ศาลาแดง"],
    "ศาลาแดง": ["ราชดำริ", "ช่องนนทรี"],
    "ช่องนนทรี": ["ศาลาแดง", "เซนต์หลุยส์"],
    "เซนต์หลุยส์": ["ช่องนนทรี", "สุรศักดิ์"],
    "สุรศักดิ์": ["เซนต์หลุยส์", "สะพานตากสิน"],
    "สะพานตากสิน": ["สุรศักดิ์", "กรุงธนบุรี"],
    "กรุงธนบุรี": ["สะพานตากสิน", "วงเวียนใหญ่"],
    "วงเวียนใหญ่": ["กรุงธนบุรี", "บางหว้า"],
    "บางหว้า": ["วงเวียนใหญ่"],
}

# ─── 2. HELPER FUNCTIONS & GRAPH ALGORITHMS ──────────────────────────────
elif mode_code == "bts":
        # คำนวณหาสถานี BTS ต้นทางและปลายทางที่ใกล้จุด Origin/Destination ที่สุด
        df_bts_master["dist_s"] = [haversine_distance(start_info["latitude"], start_info["longitude"], r["lat"], r["lng"]) for i, r in df_bts_master.iterrows()]
        df_bts_master["dist_e"] = [haversine_distance(end_info["latitude"], end_info["longitude"], r["lat"], r["lng"]) for i, r in df_bts_master.iterrows()]
        
        bts_s = df_bts_master.sort_values("dist_s").iloc[0]
        bts_e = df_bts_master.sort_values("dist_e").iloc[0]

        bts_start_name = bts_s["clean_name"]
        bts_end_name = bts_e["clean_name"]

        # ค้นหาเส้นทางบน Network Graph ผ่าน BFS
        station_path = find_bts_path(bts_start_name, bts_end_name)

        # 2.1 Leg 1: เดินจากจุดเริ่มต้น -> สถานีขึ้น BTS (ใช้ถนนจริง OSRM)
        leg1_coords, leg1_dist, leg1_time = get_osrm_route(start_info["latitude"], start_info["longitude"], bts_s["lat"], bts_s["lng"], mode="foot")
        
        # 2.2 Leg 2: พิกัดสถานีบนราง
        bts_route_coords = [BTS_STATIONS[name] for name in station_path]
        num_stations = max(0, len(station_path) - 1)
        bts_time_sec = num_stations * 150  # เฉลี่ย 2.5 นาทีต่อสถานี
        
        bts_dist_m = 0
        for i in range(len(bts_route_coords) - 1):
            bts_dist_m += haversine_distance(bts_route_coords[i][0], bts_route_coords[i][1], bts_route_coords[i+1][0], bts_route_coords[i+1][1])

        # 2.3 Leg 3: เดินจากสถานีลง BTS -> จุดหมายปลายทาง (ใช้ถนนจริง OSRM)
        leg3_coords, leg3_dist, leg3_time = get_osrm_route(bts_e["lat"], bts_e["lng"], end_info["latitude"], end_info["longitude"], mode="foot")

        total_dist_m = leg1_dist + bts_dist_m + leg3_dist
        total_time_sec = leg1_time + bts_time_sec + leg3_time

        # --- วาดเส้นทางบนแผนที่ ---
        # 1. เส้นทางเดินเท้าจริง (Leg 1 & Leg 3)
        folium.PolyLine(leg1_coords, color="#34A853", weight=6, tooltip="เดินเท้าไปสถานี BTS").add_to(m)
        folium.PolyLine(leg3_coords, color="#EA4335", weight=6, tooltip="เดินเท้าไปยังจุดหมาย").add_to(m)

        # 2. เส้นทางราง BTS (ใช้เส้นประเพื่อสื่อว่าเป็น Transit Line ไม่ใช่ทางเดินเท้า)
        folium.PolyLine(
            bts_route_coords, 
            color="#0F9D58", 
            weight=4, 
            opacity=0.8, 
            dash_array="8, 8", 
            tooltip=f"แนวเส้นทาง BTS ({num_stations} สถานี)"
        ).add_to(m)

        # 3. แสดง Node สถานีรายทาง
        for st_name in station_path:
            pos = BTS_STATIONS[st_name]
            is_interchange = st_name == "สยาม"
            folium.CircleMarker(
                location=pos,
                radius=7 if is_interchange else 4,
                popup=f"สถานี {st_name} {'(จุดเปลี่ยนสาย Interchange)' if is_interchange else ''}",
                color="red" if is_interchange else "#0F9D58",
                fill=True,
                fill_opacity=1,
            ).add_to(m)

        folium.Marker([bts_s["lat"], bts_s["lng"]], popup=f"ขึ้นสถานี: {bts_s['clean_name']}", icon=folium.Icon(color="blue", icon="train", prefix="fa")).add_to(m)
        folium.Marker([bts_e["lat"], bts_e["lng"]], popup=f"ลงสถานี: {bts_e['clean_name']}", icon=folium.Icon(color="blue", icon="train", prefix="fa")).add_to(m)

        # Display Card Summary
        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int(total_time_sec//60)} นาที</div>
            <div class="dist-sub">🚇 BTS {bts_start_name} ➔ {bts_end_name} ({num_stations} สถานี)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ **ลำดับสถานีที่ผ่าน ({len(station_path)} สถานี):**\n\n" + " ➔ ".join(station_path))
        st.info(f"""
        * 🟢 **เดินเท้า/เข็นไปสถานีต้นทาง:** {bts_start_name} ({leg1_dist:.0f} เมตร) - *วาดเส้นตามถนนจริง*
        * 🛗 **สิ่งอำนวยความสะดวก:** มีลิฟต์และทางลาดรองรับรถเข็น
        * 🔴 **เดินเท้า/เข็นจากสถานีปลายทาง:** {bts_end_name} ไปจุดหมาย ({leg3_dist:.0f} เมตร) - *วาดเส้นตามถนนจริง*
        """)

# ─── 3. DATA LOADING & AI MODEL TRAINING ──────────────────────────────────
@st.cache_data
def load_data():
    try:
        df_places = pd.read_csv("bangkok_places_bus_spot.csv")
    except Exception:
        # Fallback dataset
        df_places = pd.DataFrame([
            {"place_name": "อนุสาวรีย์ชัยสมรภูมิ", "latitude": 13.7628, "longitude": 100.5371},
            {"place_name": "โรงพยาบาลจุฬาลงกรณ์", "latitude": 13.7314, "longitude": 100.5342},
            {"place_name": "สยามพารากอน", "latitude": 13.7460, "longitude": 100.5348},
            {"place_name": "โรงพยาบาลศิริราช", "latitude": 13.7588, "longitude": 100.4866},
            {"place_name": "เซ็นทรัลเวิลด์", "latitude": 13.7469, "longitude": 100.5395},
            {"place_name": "โรงพยาบาลรามาธิบดี", "latitude": 13.7651, "longitude": 100.5332},
        ])

    # รวมสถานที่ BTS เข้าไปใน Dataset เพื่อให้ผู้ใช้เลือกได้ง่ายขึ้น
    bts_list = []
    for k, v in BTS_STATIONS.items():
        bts_list.append({"clean_name": k, "lat": v[0], "lng": v[1]})
        # หากสถานที่ไม่อยู่ใน df_places ให้เพิ่มเป็นตัวเลือก
        if f"BTS {k}" not in df_places["place_name"].values:
            df_places = pd.concat([
                df_places, 
                pd.DataFrame([{"place_name": f"BTS {k}", "latitude": v[0], "longitude": v[1]}])
            ], ignore_index=True)

    df_bts = pd.DataFrame(bts_list)
    df_bts["มีลิฟต์"] = 1
    df_bts["ทางลาดสำหรับรถเข็น"] = 1
    return df_places, df_bts

df_places, df_bts_master = load_data()

@st.cache_resource
def train_ai_model():
    np.random.seed(42)
    n = 500
    lift = np.random.choice([0, 1], size=n, p=[0.1, 0.9])
    ramp = np.random.choice([0, 1], size=n, p=[0.1, 0.9])
    dist = np.random.uniform(50, 3000, size=n)
    mode = np.random.choice(["BTS", "BUS", "CAR", "AMBULANCE", "WALK"], size=n)

    labels = []
    for i in range(n):
        s = 1.0
        if mode[i] == "WALK" and dist[i] > 600: s -= 0.5
        if lift[i] == 0 or ramp[i] == 0: s -= 0.4
        labels.append(1 if s >= 0.5 else 0)

    df = pd.DataFrame({"Lift": lift, "Ramp": ramp, "Dist": dist, "Mode": mode, "Safety": labels})
    le = LabelEncoder()
    df["Mode_enc"] = le.fit_transform(df["Mode"])

    features = ["Lift", "Ramp", "Dist", "Mode_enc"]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[features], df["Safety"])
    return model, le, features

ai_model, ai_le, ai_features = train_ai_model()

# ─── 4. CSS STYLING & HEADER ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Maps Header Toolbar */
    .gmap-header {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin-bottom: 20px;
        border-left: 6px solid #4285F4;
    }
    .gmap-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #202124;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Route Metric Card */
    .time-card {
        background-color: #e8f0fe;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1a73e8;
        margin-bottom: 15px;
    }
    .time-main {
        font-size: 2.2rem;
        font-weight: 800;
        color: #188038;
        margin: 0;
    }
    .dist-sub {
        font-size: 1.1rem;
        color: #5f6368;
        font-weight: 500;
    }

    /* Mode Selection Highlight */
    .stRadio > div {
        flex-direction: row !important;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gmap-header">
    <div class="gmap-title">🗺️ AI Accessibility Google Maps & BTS Graph Navigation</div>
    <span style="color: #5f6368; font-size: 0.95rem;">ระบบคำนวณเส้นทางอัจฉริยะ รองรับ รถไฟฟ้า BTS (ค้นหาด้วย BFS Graph Network), รถเมล์ชานต่ำ, ทางเดินเท้า และรถพยาบาลฉุกเฉิน</span>
</div>
""", unsafe_allow_html=True)

# ─── 5. USER CONTROLS ─────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    start_place = st.selectbox("📍 เลือกจุดเริ่มต้น (Origin):", df_places["place_name"].tolist(), index=0)
with c2:
    end_place = st.selectbox("🏁 เลือกจุดหมายปลายทาง (Destination):", df_places["place_name"].tolist(), index=min(1, len(df_places)-1))

start_info = df_places[df_places["place_name"] == start_place].iloc[0]
end_info = df_places[df_places["place_name"] == end_place].iloc[0]

st.write("**🚗 เลือกโหมดการเดินทาง (Travel Mode Selector):**")
mode_choice = st.radio(
    label="เลือกรูปแบบการเดินทาง",
    options=[
        "🚗 ขับรถ/แท็กซี่",
        "🚇 รถไฟฟ้า BTS (Graph Navigation)",
        "🚌 รถเมล์ชานต่ำ (Low-Floor Bus)",
        "🚶 เดินเข็น (Wheelchair/Walk)",
        "🚑 รถพยาบาลฉุกเฉิน (Ambulance Fast-Track)"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if "🚗" in mode_choice: mode_code = "driving"
elif "🚇" in mode_choice: mode_code = "bts"
elif "🚌" in mode_choice: mode_code = "bus"
elif "🚶" in mode_choice: mode_code = "foot"
else: mode_code = "ambulance"

st.write("---")

# ─── 6. MAIN DASHBOARD & MAP DISPLAY ─────────────────────────────────────
col_left, col_right = st.columns([1.1, 1.9])

m = folium.Map(
    location=[(start_info["latitude"] + end_info["latitude"]) / 2, (start_info["longitude"] + end_info["longitude"]) / 2],
    zoom_start=13, tiles="CartoDB Positron"
)

# Marker ต้นทาง/ปลายทาง
folium.Marker([start_info["latitude"], start_info["longitude"]], popup=f"<b>ต้นทาง:</b> {start_place}", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
folium.Marker([end_info["latitude"], end_info["longitude"]], popup=f"<b>ปลายทาง:</b> {end_place}", icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)

total_dist_m = 0
total_time_sec = 0

with col_left:
    st.markdown("### 🧭 สรุปการเดินทาง (Trip Details)")

    # 1. โหมดขับรถ/แท็กซี่
    if mode_code == "driving":
        coords, total_dist_m, total_time_sec = get_osrm_route(start_info["latitude"], start_info["longitude"], end_info["latitude"], end_info["longitude"], mode="driving")
        folium.PolyLine(coords, color="#4285F4", weight=7, opacity=0.8, tooltip="เส้นทางขับรถ").add_to(m)
        
        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int(total_time_sec//60)} นาที</div>
            <div class="dist-sub">📏 ระยะทาง {total_dist_m/1000:.2f} กม. (ผ่านทางหลวงปกติ)</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 **คำแนะนำรถเข็น:** แนะนำเรียก GrabCar+ / SUV เพื่อพับเก็บรถเข็นไว้ท้ายรถได้อย่างสะดวกสบาย")

    # 2. โหมด BTS (ใช้ BFS Graph Network ในการหาเส้นทางบนราง)
    elif mode_code == "bts":
        # คำนวณหาสถานี BTS ต้นทางและปลายทางที่ใกล้จุด Origin/Destination ที่สุด
        df_bts_master["dist_s"] = [haversine_distance(start_info["latitude"], start_info["longitude"], r["lat"], r["lng"]) for i, r in df_bts_master.iterrows()]
        df_bts_master["dist_e"] = [haversine_distance(end_info["latitude"], end_info["longitude"], r["lat"], r["lng"]) for i, r in df_bts_master.iterrows()]
        
        bts_s = df_bts_master.sort_values("dist_s").iloc[0]
        bts_e = df_bts_master.sort_values("dist_e").iloc[0]

        bts_start_name = bts_s["clean_name"]
        bts_end_name = bts_e["clean_name"]

        # ค้นหาเส้นทางบน Network Graph ผ่าน BFS
        station_path = find_bts_path(bts_start_name, bts_end_name)

        # 2.1 Leg 1: เดินจากจุดเริ่มต้น -> สถานีขึ้น BTS
        leg1_coords, leg1_dist, leg1_time = get_osrm_route(start_info["latitude"], start_info["longitude"], bts_s["lat"], bts_s["lng"], mode="foot")
        
        # 2.2 Leg 2: วิ่งบนราง BTS ตาม Node Path ที่หาได้จาก Graph
        bts_route_coords = [BTS_STATIONS[name] for name in station_path]
        num_stations = max(0, len(station_path) - 1)
        bts_time_sec = num_stations * 150  # เฉลี่ย 2.5 นาทีต่อสถานี
        
        # คำนวณระยะทางรวมตามราง BTS
        bts_dist_m = 0
        for i in range(len(bts_route_coords) - 1):
            bts_dist_m += haversine_distance(bts_route_coords[i][0], bts_route_coords[i][1], bts_route_coords[i+1][0], bts_route_coords[i+1][1])

        # 2.3 Leg 3: เดินจากสถานีลง BTS -> จุดหมายปลายทาง
        leg3_coords, leg3_dist, leg3_time = get_osrm_route(bts_e["lat"], bts_e["lng"], end_info["latitude"], end_info["longitude"], mode="foot")

        total_dist_m = leg1_dist + bts_dist_m + leg3_dist
        total_time_sec = leg1_time + bts_time_sec + leg3_time

        # วาดเส้นทางบนแผนที่
        folium.PolyLine(leg1_coords, color="#34A853", weight=5, dash_array="4, 8", tooltip="เดินไปสถานี BTS").add_to(m)
        folium.PolyLine(bts_route_coords, color="#0F9D58", weight=8, opacity=0.9, tooltip=f"BTS Path ({num_stations} สถานี)").add_to(m)
        folium.PolyLine(leg3_coords, color="#EA4335", weight=5, dash_array="4, 8", tooltip="เดินไปยังจุดหมาย").add_to(m)

        # วาด Node สถานี BTS ที่นั่งผ่านตลอดเส้นทาง
        for st_name in station_path:
            pos = BTS_STATIONS[st_name]
            is_interchange = st_name == "สยาม"
            folium.CircleMarker(
                location=pos,
                radius=8 if is_interchange else 5,
                popup=f"สถานี {st_name} {'(จุดเปลี่ยนสาย Interchange)' if is_interchange else ''}",
                color="red" if is_interchange else "#0F9D58",
                fill=True,
                fill_opacity=1,
            ).add_to(m)

        folium.Marker([bts_s["lat"], bts_s["lng"]], popup=f"ขึ้นที่สถานี: {bts_s['clean_name']}", icon=folium.Icon(color="blue", icon="train", prefix="fa")).add_to(m)
        folium.Marker([bts_e["lat"], bts_e["lng"]], popup=f"ลงที่สถานี: {bts_e['clean_name']}", icon=folium.Icon(color="blue", icon="train", prefix="fa")).add_to(m)

        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int(total_time_sec//60)} นาที</div>
            <div class="dist-sub">🚇 BTS {bts_start_name} ➔ {bts_end_name} ({num_stations} สถานี)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ **ลำดับสถานีผ่าน ({len(station_path)} สถานี):**\n\n" + " ➔ ".join(station_path))
        st.info(f"""
        * 🟢 **เดินเข็นไปสถานีต้นทาง:** {bts_start_name} ({leg1_dist:.0f} เมตร)
        * 🛗 **สิ่งอำนวยความสะดวก:** มีลิฟต์และทางลาดรองรับรถเข็น
        * 🔴 **ออกจากสถานีปลายทาง:** {bts_end_name} ไปจุดหมาย ({leg3_dist:.0f} เมตร)
        """)

    # 3. โหมด รถเมล์ชานต่ำ
    elif mode_code == "bus":
        coords, total_dist_m, total_time_sec = get_osrm_route(start_info["latitude"], start_info["longitude"], end_info["latitude"], end_info["longitude"], mode="driving")
        folium.PolyLine(coords, color="#9C27B0", weight=7, tooltip="รถเมล์ชานต่ำ").add_to(m)

        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int((total_time_sec*1.25)//60)} นาที</div>
            <div class="dist-sub">🚌 รถเมล์ปรับอากาศชานต่ำ (Low-Floor Bus)</div>
        </div>
        """, unsafe_allow_html=True)
        st.warning("""
        * 🚌 **สายรถเมล์ชานต่ำแนะนำ:** สาย 8, 28, 515, 1-36 (Thai Smile Bus 100%)
        * ♿ **อารยสถาปัตย์:** มีทางลาดระบบไฮโดรลิกปรับลาดเอียงเทียบฟุตบาทได้สะดวก
        """)

    # 4. โหมด เดินเข็น
    elif mode_code == "foot":
        coords, total_dist_m, total_time_sec = get_osrm_route(start_info["latitude"], start_info["longitude"], end_info["latitude"], end_info["longitude"], mode="foot")
        folium.PolyLine(coords, color="#FBBC05", weight=6, tooltip="เส้นทางเข็นวีลแชร์").add_to(m)

        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int(total_time_sec//60)} นาที</div>
            <div class="dist-sub">🚶 ระยะทางเข็นรวม {total_dist_m:.0f} เมตร</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("♿ **คำแนะนำ:** เส้นทางเกาะแนวทางเท้า เลี่ยงจุดก่อสร้างและสี่แยกใหญ่")

    # 5. โหมด รถพยาบาลฉุกเฉิน
    elif mode_code == "ambulance":
        coords, total_dist_m, total_time_sec = get_osrm_route(start_info["latitude"], start_info["longitude"], end_info["latitude"], end_info["longitude"], mode="ambulance")
        folium.PolyLine(coords, color="#EA4335", weight=9, opacity=0.9, tooltip="เส้นทางฉุกเฉิน Fast-Track").add_to(m)

        st.markdown(f"""
        <div class="time-card" style="background-color:#fce8e6; border-left-color:#d93025;">
            <div class="time-main" style="color:#d93025;">⚡ {int(total_time_sec//60)} นาที (Fast-Track)</div>
            <div class="dist-sub">🚑 บริการรถพยาบาลฉุกเฉิน / สวัสดิการรับส่งผู้ป่วย</div>
        </div>
        """, unsafe_allow_html=True)
        st.error("""
        * 🚨 **สายด่วนรับแจ้งเหตุฉุกเฉิน:** โทร **1669** (สพฉ.)
        * 📞 **บริการรถตู้สวัสดิการผู้พิการ กทม.:** โทร **1555** / **1479**
        """)

    # ─── AI SAFETY MODEL ASSESSMENT ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 AI Safety Assessment (Random Forest)")
    
    mode_str_ai = "BTS" if mode_code == "bts" else ("BUS" if mode_code == "bus" else "CAR")
    encoded_mode = ai_le.transform([mode_str_ai])[0]
    input_vector = pd.DataFrame([[1, 1, total_dist_m, encoded_mode]], columns=ai_features)
    
    pred = ai_model.predict(input_vector)[0]
    prob = ai_model.predict_proba(input_vector)[0]

    if pred == 1:
        st.success(f"🟢 **AI Status: APPROVED**\n\nดัชนีความปลอดภัยสำหรับรถเข็น: **{prob[1]*100:.1f}%**")
    else:
        st.warning(f"⚠️ **AI Status: WARNING**\n\nพบความเสี่ยงในเส้นทาง (ระดับความปลอดภัย {prob[1]*100:.1f}%)")

with col_right:
    st.markdown("### 🗺️ Google Maps Interactive Canvas")
    st_folium(m, width="100%", height=650)
