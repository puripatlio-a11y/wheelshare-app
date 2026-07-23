from collections import deque
import math
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

# เส้นแบ่งสาย ใช้เพื่อตรวจจับจุดเปลี่ยนสาย (Interchange) ที่สถานีสยาม
SUKHUMVIT_LINE_STATIONS = {
    "คูคต", "แยก คปอ.", "พิพิธภัณฑ์กองทัพอากาศ", "โรงพยาบาลภูมิพลอดุลยเดช", "สะพานใหม่",
    "สายหยุด", "พหลโยธิน 59", "วัดพระศรีมหาธาตุ", "กรมทหารราบที่ 11", "บางบัว",
    "กรมป่าไม้", "มหาวิทยาลัยเกษตรศาสตร์", "เสนานิคม", "รัชโยธิน", "พหลโยธิน 24",
    "ห้าแยกลาดพร้าว", "หมอชิต", "สะพานควาย", "อารีย์", "สนามเป้า",
    "อนุสาวรีย์ชัยสมรภูมิ", "พญาไท", "ราชเทวี", "สยาม", "ชิดลม", "เพลินจิต",
    "นานา", "อโศก", "พร้อมพงษ์", "ทองหล่อ", "เอกมัย", "พระโขนง", "อ่อนนุช",
    "บางจาก", "ปุณณวิถี", "อุดมสุข", "บางนา", "แบริ่ง", "สำโรง",
}
SILOM_LINE_STATIONS = {
    "สยาม", "สนามกีฬาแห่งชาติ", "ราชดำริ", "ศาลาแดง", "ช่องนนทรี", "เซนต์หลุยส์",
    "สุรศักดิ์", "สะพานตากสิน", "กรุงธนบุรี", "วงเวียนใหญ่", "บางหว้า",
}

# ─── ค่าคงที่เวลาเดินทาง (BTS / รถเมล์) ────────────────────────────────────
BTS_AVERAGE_WAIT_SEC = 240        # เวลารอรถไฟฟ้าเฉลี่ย ~4 นาที (ช่วงเวลาให้บริการปกติ)
BTS_INTERCHANGE_TIME_SEC = 300    # เวลาเดิน+รอเปลี่ยนสายที่สถานีสยาม ~5 นาที
BTS_TIME_PER_STATION_SEC = 120    # เวลาวิ่งเฉลี่ยระหว่างสถานี ~2 นาที

BUS_STOP_INTERVAL_M = 700         # ระยะห่างป้ายรถเมล์จำลองโดยเฉลี่ย
BUS_DWELL_TIME_SEC = 25           # เวลาจอดรับ-ส่งผู้โดยสารต่อป้าย
BUS_AVERAGE_WAIT_SEC = 480        # เวลารอรถเมล์เฉลี่ย ~8 นาที (ความถี่รถเมล์ต่ำกว่า BTS)
BUS_TRAFFIC_FACTOR = 1.25         # ปัจจัยหน่วงเวลาจากสภาพจราจร/การจอดข้างทาง

# ─── 2. HELPER FUNCTIONS & GRAPH ALGORITHMS ──────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางตามเส้นโค้งโลก (เมตร)"""
    R = 6371000  # รัศมีโลก
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_bts_path(start_station, end_station):
    """ค้นหาเส้นทาง BTS ด้วย Breadth-First Search (BFS)"""
    if start_station == end_station:
        return [start_station]
    queue = deque([[start_station]])
    visited = {start_station}
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor in BTS_GRAPH.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path) + [neighbor]
                if neighbor == end_station:
                    return new_path
                queue.append(new_path)
    return [start_station, end_station]

def has_real_interchange(path):
    """
    ตรวจสอบว่าเส้นทาง BTS ต้องเปลี่ยนสายจริงที่สถานีสยามหรือไม่
    (ต้องเดินทางจากสายสุขุมวิทข้ามไปสายสีลม หรือกลับกัน)
    """
    if "สยาม" not in path:
        return False
    idx = path.index("สยาม")

    def line_of(station):
        in_suk = station in SUKHUMVIT_LINE_STATIONS
        in_sil = station in SILOM_LINE_STATIONS
        if in_suk and not in_sil:
            return "SUK"
        if in_sil and not in_suk:
            return "SIL"
        return None  # สถานีร่วม (สยามเอง) หรือไม่พบ

    before_line = line_of(path[idx - 1]) if idx > 0 else None
    after_line = line_of(path[idx + 1]) if idx < len(path) - 1 else None
    return bool(before_line and after_line and before_line != after_line)

def simulate_bus_stops(coords, interval_m=BUS_STOP_INTERVAL_M):
    """
    จำลองตำแหน่งป้ายรถเมล์ตลอดแนวเส้นทางถนน โดยวางป้ายทุกๆ ระยะ interval_m เมตร
    แทนการใช้เส้นทางรถยนต์ OSRM แบบตรงๆ โดยไม่มีจุดจอดกลางทาง
    """
    if len(coords) < 2:
        return list(coords)
    stops = [coords[0]]
    acc = 0.0
    for i in range(1, len(coords)):
        seg_dist = haversine_distance(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        acc += seg_dist
        if acc >= interval_m:
            stops.append(coords[i])
            acc = 0.0
    if stops[-1] != coords[-1]:
        stops.append(coords[-1])
    return stops

def get_osrm_route(lat1, lon1, lat2, lon2, mode="driving"):
    """ดึงข้อมูล Routing จาก OSRM API (รองรับ driving, foot, ambulance)"""
    profile = "foot" if mode == "foot" else "driving"
    url = f"http://router.project-osrm.org/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    try:
        res = requests.get(url, timeout=4).json()
        if res.get("code") == "Ok":
            coords = [[pt[1], pt[0]] for pt in res["routes"][0]["geometry"]["coordinates"]]
            dist = res["routes"][0]["distance"]
            dur = res["routes"][0]["duration"]
            if mode == "ambulance":
                dur *= 0.6  # รถพยาบาลวิ่งเร็วกว่าปกติ 40%
            return coords, dist, dur
    except Exception:
        pass
    # Fallback กรณี API มีปัญหา
    fallback_coords = [[lat1, lon1], [lat2, lon2]]
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    speed = 1.2 if mode == "foot" else 8.3  # m/s
    return fallback_coords, dist, dist / speed

# ─── 3. DATA LOADING & AI MODEL TRAINING ──────────────────────────────────
@st.cache_data
def load_data():
    try:
        df_places = pd.read_csv("bangkok_places_bus_spot.csv")
    except Exception:
        df_places = pd.DataFrame([
            {"place_name": "อนุสาวรีย์ชัยสมรภูมิ", "latitude": 13.7628, "longitude": 100.5371},
            {"place_name": "โรงพยาบาลจุฬาลงกรณ์", "latitude": 13.7314, "longitude": 100.5342},
            {"place_name": "สยามพารากอน", "latitude": 13.7460, "longitude": 100.5348},
            {"place_name": "โรงพยาบาลศิริราช", "latitude": 13.7588, "longitude": 100.4866},
            {"place_name": "เซ็นทรัลเวิลด์", "latitude": 13.7469, "longitude": 100.5395},
            {"place_name": "โรงพยาบาลรามาธิบดี", "latitude": 13.7651, "longitude": 100.5332},
        ])

    bts_list = []
    for k, v in BTS_STATIONS.items():
        bts_list.append({"clean_name": k, "lat": v[0], "lng": v[1]})
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
    # สภาพป้ายรถเมล์ / ความหนาแน่นบริเวณป้าย (1 = ดี/ไม่แออัด, 0 = แออัด/ไม่มีสิ่งอำนวยความสะดวก)
    bus_cond = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

    labels = []
    for i in range(n):
        s = 1.0
        if mode[i] == "WALK" and dist[i] > 600: s -= 0.5
        if lift[i] == 0 or ramp[i] == 0: s -= 0.4
        if mode[i] == "BUS" and bus_cond[i] == 0: s -= 0.3  # ป้ายรถเมล์แออัด/ไม่มีสิ่งอำนวยความสะดวก = เสี่ยงขึ้น
        labels.append(1 if s >= 0.5 else 0)

    df = pd.DataFrame({"Lift": lift, "Ramp": ramp, "Dist": dist, "Mode": mode, "BusCond": bus_cond, "Safety": labels})
    le = LabelEncoder()
    df["Mode_enc"] = le.fit_transform(df["Mode"])

    features = ["Lift", "Ramp", "Dist", "Mode_enc", "BusCond"]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[features], df["Safety"])
    return model, le, features

ai_model, ai_le, ai_features = train_ai_model()

# ─── 4. CSS STYLING & HEADER ──────────────────────────────────────────────
st.markdown("""
<style>
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
    .stRadio > div {
        flex-direction: row !important;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gmap-header">
    <div class="gmap-title">AI Accessibility Route Planner for Wheelchair Users</div>
    <span style="color: #5f6368; font-size: 0.95rem;">ระบบส่งเสริมการวางแผนการเดินทางด้วยปัญญาประดิษฐ์สำหรับผู้ใช้รถเข็น</span>
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

st.write("**เลือกโหมดการเดินทาง (Travel Mode Selector):**")
mode_choice = st.radio(
    label="เลือกรูปแบบการเดินทาง",
    options=[
        "🚗 ขับรถ/แท็กซี่",
        "🚇 รถไฟฟ้า BTS (Graph Navigation)",
        "🚌 รถเมล์ชานต่ำ (Low-Floor Bus)",
        "🚶 เดินเข็น (Wheelchair/Walk)",
        "รถพยาบาลฉุกเฉิน (Ambulance Fast-Track)"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if "🚗" in mode_choice: mode_code = "driving"
elif "🚇" in mode_choice: mode_code = "bts"
elif "🚌" in mode_choice: mode_code = "bus"
elif "🚶" in mode_choice: mode_code = "foot"
else: mode_code = "ambulance"

# ค่าเริ่มต้นของสภาพป้ายรถเมล์ (ใช้กับ AI Safety Model เฉพาะโหมดรถเมล์)
bus_cond_value = 1

st.write("---")

# ─── 6. MAIN DASHBOARD & MAP DISPLAY ─────────────────────────────────────
col_left, col_right = st.columns([1.1, 1.9])

m = folium.Map(
    location=[(start_info["latitude"] + end_info["latitude"]) / 2, (start_info["longitude"] + end_info["longitude"]) / 2],
    zoom_start=13, tiles="CartoDB Positron"
)

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

    # 2. โหมด BTS (เพิ่มเวลารอรถ + เวลาเปลี่ยนสายที่สถานีสยาม)
    elif mode_code == "bts":
        df_bts_master["dist_s"] = [haversine_distance(start_info["latitude"], start_info["longitude"], r["lat"], r["lng"]) for i, r in df_bts_master.iterrows()]
        df_bts_master["dist_e"] = [haversine_distance(end_info["latitude"], end_info["longitude"], r["lat"], r["lng"]) for i, r in df_bts_master.iterrows()]

        bts_s = df_bts_master.sort_values("dist_s").iloc[0]
        bts_e = df_bts_master.sort_values("dist_e").iloc[0]

        bts_start_name = bts_s["clean_name"]
        bts_end_name = bts_e["clean_name"]

        station_path = find_bts_path(bts_start_name, bts_end_name)
        is_transfer = has_real_interchange(station_path)

        # Leg 1: เดินเข็นไปสถานีขึ้น BTS (ใช้ถนนจริง OSRM)
        leg1_coords, leg1_dist, leg1_time = get_osrm_route(start_info["latitude"], start_info["longitude"], bts_s["lat"], bts_s["lng"], mode="foot")
        folium.PolyLine(leg1_coords, color="#34A853", weight=6, tooltip="เดินเท้าไปสถานี BTS").add_to(m)

        # Leg 2: วิ่งบนแนว BTS เชื่อมทีละสถานีด้วย OSRM Road Mapping
        bts_total_dist_m = 0
        all_bts_road_coords = []

        if len(station_path) > 1:
            for i in range(len(station_path) - 1):
                st_curr = station_path[i]
                st_next = station_path[i+1]
                pos_curr = BTS_STATIONS[st_curr]
                pos_next = BTS_STATIONS[st_next]

                seg_coords, seg_dist, _ = get_osrm_route(pos_curr[0], pos_curr[1], pos_next[0], pos_next[1], mode="driving")
                bts_total_dist_m += seg_dist
                all_bts_road_coords.extend(seg_coords)

            folium.PolyLine(
                all_bts_road_coords,
                color="#0F9D58",
                weight=7,
                opacity=0.85,
                tooltip=f"แนวเส้นทาง BTS ({len(station_path)-1} สถานี)"
            ).add_to(m)

        # Leg 3: เดินเข็นจากสถานีลง BTS -> จุดหมาย (ใช้ถนนจริง OSRM)
        leg3_coords, leg3_dist, leg3_time = get_osrm_route(bts_e["lat"], bts_e["lng"], end_info["latitude"], end_info["longitude"], mode="foot")
        folium.PolyLine(leg3_coords, color="#EA4335", weight=6, tooltip="เดินเท้าไปยังจุดหมาย").add_to(m)

        num_stations = max(0, len(station_path) - 1)
        bts_running_time_sec = num_stations * BTS_TIME_PER_STATION_SEC
        bts_interchange_sec = BTS_INTERCHANGE_TIME_SEC if is_transfer else 0

        total_dist_m = leg1_dist + bts_total_dist_m + leg3_dist
        total_time_sec = (
            leg1_time
            + BTS_AVERAGE_WAIT_SEC
            + bts_running_time_sec
            + bts_interchange_sec
            + leg3_time
        )

        # หมุดสถานี BTS
        for st_name in station_path:
            pos = BTS_STATIONS[st_name]
            is_interchange = (st_name == "สยาม") and is_transfer
            folium.CircleMarker(
                location=pos,
                radius=8 if is_interchange else 5,
                popup=f"สถานี {st_name} {'(จุดเปลี่ยนสาย Interchange)' if is_interchange else ''}",
                color="red" if is_interchange else "#0F9D58",
                fill=True,
                fill_opacity=1,
            ).add_to(m)

        folium.Marker([bts_s["lat"], bts_s["lng"]], popup=f"ขึ้นสถานี: {bts_s['clean_name']}", icon=folium.Icon(color="blue", icon="train", prefix="fa")).add_to(m)
        folium.Marker([bts_e["lat"], bts_e["lng"]], popup=f"ลงสถานี: {bts_e['clean_name']}", icon=folium.Icon(color="blue", icon="train", prefix="fa")).add_to(m)

        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int(total_time_sec//60)} นาที</div>
            <div class="dist-sub">🚇 BTS {bts_start_name} ➔ {bts_end_name} ({num_stations} สถานี){' • เปลี่ยนสายที่สยาม' if is_transfer else ''}</div>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"✅ **ลำดับสถานีที่ผ่าน ({len(station_path)} สถานี):**\n\n" + " ➔ ".join(station_path))

        breakdown_lines = [
            f"* **เดินเท้า/เข็นไปสถานีต้นทาง:** {bts_start_name} ({leg1_dist:.0f} เมตร, {int(leg1_time//60)} นาที) - *ตามทางเท้าถนนจริง*",
            f"* **เวลารอรถไฟฟ้าเฉลี่ย:** ~{BTS_AVERAGE_WAIT_SEC//60} นาที",
            f"* **เวลาวิ่งบนขบวน:** ~{bts_running_time_sec//60} นาที ({num_stations} สถานี)",
        ]
        if is_transfer:
            breakdown_lines.append(f"* 🔁 **เวลาเปลี่ยนสายที่สถานีสยาม:** +{bts_interchange_sec//60} นาที (เดินข้ามชานชาลา/รอขบวนใหม่)")
        breakdown_lines.append(f"* 🛗 **สิ่งอำนวยความสะดวก:** มีลิฟต์และทางลาดรองรับรถเข็น")
        breakdown_lines.append(f"* 🔴 **เดินเท้า/เข็นจากสถานีปลายทาง:** {bts_end_name} ไปจุดหมาย ({leg3_dist:.0f} เมตร, {int(leg3_time//60)} นาที) - *ตามทางเท้าถนนจริง*")
        st.info("\n".join(breakdown_lines))

    # 3. โหมด รถเมล์ชานต่ำ (จำลองป้ายจอดตลอดเส้นทาง + เวลารอรถเมล์)
    elif mode_code == "bus":
        coords, base_dist_m, base_drive_time = get_osrm_route(start_info["latitude"], start_info["longitude"], end_info["latitude"], end_info["longitude"], mode="driving")
        folium.PolyLine(coords, color="#9C27B0", weight=7, tooltip="รถเมล์ชานต่ำ").add_to(m)

        bus_stop_coords = simulate_bus_stops(coords)
        # ป้ายกลางทาง = ไม่นับจุดต้นทาง/ปลายทางในลิสต์
        intermediate_stops = bus_stop_coords[1:-1] if len(bus_stop_coords) > 2 else []
        num_bus_stops = len(intermediate_stops)

        for i, pt in enumerate(intermediate_stops, start=1):
            folium.CircleMarker(
                location=pt,
                radius=5,
                popup=f"ป้ายรถเมล์จำลอง #{i}",
                color="#6A1B9A",
                fill=True,
                fill_opacity=0.9,
            ).add_to(m)

        # เลือกสภาพป้ายรถเมล์ปัจจุบัน (ใช้เป็น Feature ใน AI Safety Model)
        bus_cond_choice = st.selectbox(
            "🚏 สภาพป้ายรถเมล์บริเวณเส้นทางนี้ (Bus Stop Condition):",
            ["ดี (มีหลังคา/ไม่แออัด)", "แออัด/ไม่มีสิ่งอำนวยความสะดวก"],
        )
        bus_cond_value = 1 if "ดี" in bus_cond_choice else 0

        dwell_total_sec = num_bus_stops * BUS_DWELL_TIME_SEC
        congestion_time = base_drive_time * BUS_TRAFFIC_FACTOR
        total_time_sec = congestion_time + dwell_total_sec + BUS_AVERAGE_WAIT_SEC
        total_dist_m = base_dist_m

        st.markdown(f"""
        <div class="time-card">
            <div class="time-main">⏱️ {int(total_time_sec//60)} นาที</div>
            <div class="dist-sub">🚌 รถเมล์ปรับอากาศชานต่ำ (Low-Floor Bus) • ผ่านป้ายจำลอง {num_bus_stops} ป้าย</div>
        </div>
        """, unsafe_allow_html=True)

        st.warning(f"""
        * **สายรถเมล์ชานต่ำแนะนำ:** สาย 8, 28, 515, 1-36 (Thai Smile Bus 100%)
        * **เวลารอรถเมล์เฉลี่ย:** ~{BUS_AVERAGE_WAIT_SEC//60} นาที (ความถี่ต่ำกว่ารถไฟฟ้า)
        * **เวลาจอดรับ-ส่งที่ป้ายระหว่างทาง:** ~{dwell_total_sec//60} นาที ({num_bus_stops} ป้าย x {BUS_DWELL_TIME_SEC} วินาที)
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

    # Mapping Mode ให้ตรงกับที่ Train ไว้ใน LabelEncoder
    mode_map = {
        "bts": "BTS",
        "bus": "BUS",
        "driving": "CAR",
        "ambulance": "AMBULANCE",
        "foot": "WALK"
    }
    mode_str_ai = mode_map.get(mode_code, "CAR")
    encoded_mode = ai_le.transform([mode_str_ai])[0]

    input_vector = pd.DataFrame([[1, 1, total_dist_m, encoded_mode, bus_cond_value]], columns=ai_features)

    pred = ai_model.predict(input_vector)[0]
    prob = ai_model.predict_proba(input_vector)[0]

    if pred == 1:
        st.success(f"🟢 **AI Status: APPROVED**\n\nดัชนีความปลอดภัยสำหรับรถเข็น: **{prob[1]*100:.1f}%**")
    else:
        st.warning(f"⚠️ **AI Status: WARNING**\n\nพบความเสี่ยงในเส้นทาง (ระดับความปลอดภัย {prob[1]*100:.1f}%)")

with col_right:
    st.markdown("### 🗺️ Google Maps Interactive Canvas")
    st_folium(m, width="100%", height=650)
