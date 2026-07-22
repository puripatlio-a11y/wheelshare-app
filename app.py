import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import warnings
import requests
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Accessibility Route Planner - Google Maps Style", layout="wide", page_icon="🗺️")

# ─── AREA 1: GOOGLE MAPS STYLE CSS HEADER ─────────────────────────────────
header_html = """
<style>
    .google-header {
        background: linear-gradient(135deg, #4285F4 0%, #34A853 50%, #FBBC05 75%, #EA4335 100%);
        padding: 25px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .google-header h1 {
        color: #ffffff !important;
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        font-weight: 700;
        font-size: 2.2rem !important;
        margin-bottom: 5px;
    }
    .mode-card {
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        cursor: pointer;
    }
    .metric-badge {
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
</style>

<div class="google-header">
    <h1>♿ Google Maps AI Accessibility Navigation</h1>
    <p style="margin:0; font-size: 1.1rem;">ระบบนำทางเพื่อความเท่าเทียม รองรับ รถไฟฟ้า, รถเมล์ชานต่ำ, รถยนต์ และรถพยาบาลฉุกเฉิน</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ─── AREA 2: OSRM ROUTING ENGINE & GEOPROCESSING ─────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c * 1000  # เมตร

def get_osrm_route(start_lat, start_lon, end_lat, end_lon, mode="driving"):
    """
    ดึงข้อมูลเส้นทาง พิกัด ระยะทาง และเวลาจริงจาก OSRM Engine
    Modes: 'driving', 'foot', 'ambulance'
    """
    profile = "foot" if mode == "foot" else "car"
    url = f"https://router.project-osrm.org/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("routes"):
                route_data = data["routes"][0]
                coords = [[c[1], c[0]] for c in route_data["geometry"]["coordinates"]]
                dist_m = route_data["distance"]  # เมตร
                duration_sec = route_data["duration"]  # วินาที
                
                # ถ้ารถพยาบาลเปิดไซเรน จะเร็วกว่ารถปกติประมาณ 35%
                if mode == "ambulance":
                    duration_sec *= 0.65
                    
                return coords, dist_m, duration_sec
    except Exception:
        pass
    
    # Fallback กรณี API ขัดข้อง
    direct_dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    speed_kmh = 30 if mode == "driving" else (50 if mode == "ambulance" else 4)
    duration_sec = (direct_dist / 1000) / speed_kmh * 3600
    return [[start_lat, start_lon], [end_lat, end_lon]], direct_dist, duration_sec

# ─── AREA 3: DATASETS & BTS STATIONS DATABASE ──────────────────────────────
bts_line = {
    # ===== Sukhumvit Line =====
    "คูคต": [13.9607, 100.6256], "แยก คปอ.": [13.9495, 100.6225], "พิพิธภัณฑ์กองทัพอากาศ": [13.9384, 100.6184],
    "โรงพยาบาลภูมิพลอดุลยเดช": [13.9257, 100.6095], "สะพานใหม่": [13.9132, 100.6034], "สายหยุด": [13.9022, 100.5964],
    "พหลโยธิน 59": [13.8907, 100.5920], "วัดพระศรีมหาธาตุ": [13.8758, 100.5965], "กรมทหารราบที่ 11": [13.8730, 100.5960],
    "บางบัว": [13.8632, 100.6043], "กรมป่าไม้": [13.8506, 100.6049], "มหาวิทยาลัยเกษตรศาสตร์": [13.8462, 100.5697],
    "เสนานิคม": [13.8420, 100.5717], "รัชโยธิน": [13.8305, 100.5685], "พหลโยธิน 24": [13.8245, 100.5664],
    "ห้าแยกลาดพร้าว": [13.8163, 100.5616], "หมอชิต": [13.8026, 100.5538], "สะพานควาย": [13.7937, 100.5495],
    "อารีย์": [13.7797, 100.5447], "สนามเป้า": [13.7726, 100.5420], "อนุสาวรีย์ชัยสมรภูมิ": [13.7628, 100.5371],
    "พญาไท": [13.7569, 100.5335], "ราชเทวี": [13.7518, 100.5316], "สยาม": [13.7466, 100.5346],
    "ชิดลม": [13.7440, 100.5440], "เพลินจิต": [13.7430, 100.5498], "นานา": [13.7406, 100.5550],
    "อโศก": [13.7370, 100.5605], "พร้อมพงษ์": [13.7305, 100.5696], "ทองหล่อ": [13.7243, 100.5788],
    "เอกมัย": [13.7192, 100.5853], "พระโขนง": [13.7152, 100.5918], "อ่อนนุช": [13.7057, 100.6010],
    "บางจาก": [13.6969, 100.6053], "ปุณณวิถี": [13.6892, 100.6094], "อุดมสุข": [13.6803, 100.6107],
    "บางนา": [13.6688, 100.6047], "แบริ่ง": [13.6614, 100.6012], "สำโรง": [13.6462, 100.5952],
    "ปู่เจ้า": [13.6374, 100.5928], "ช้างเอราวัณ": [13.6290, 100.5920], "โรงเรียนนายเรือ": [13.6206, 100.5942],
    "ปากน้ำ": [13.6074, 100.5956], "ศรีนครินทร์": [13.5965, 100.6071], "แพรกษา": [13.5905, 100.6089],
    "สายลวด": [13.5789, 100.6087], "เคหะฯ": [13.5702, 100.6076],

    # ===== Silom Line =====
    "สนามกีฬาแห่งชาติ": [13.7460, 100.5290], "ราชดำริ": [13.7394, 100.5394],
    "ศาลาแดง": [13.7286, 100.5343], "ช่องนนทรี": [13.7235, 100.5293], "เซนต์หลุยส์": [13.7208, 100.5263],
    "สุรศักดิ์": [13.7187, 100.5217], "สะพานตากสิน": [13.7185, 100.5142], "กรุงธนบุรี": [13.7208, 100.5026],
    "วงเวียนใหญ่": [13.7210, 100.4956], "โพธิ์นิมิตร": [13.7192, 100.4862], "ตลาดพลู": [13.7142, 100.4768],
    "วุฒากาศ": [13.7130, 100.4675], "บางหว้า": [13.7203, 100.4570]
}

@st.cache_data
def load_data():
    try:
        df_places = pd.read_csv('bangkok_places_bus_spot.csv')
    except Exception:
        df_places = pd.DataFrame([
            {"place_name": "Victory Monument", "latitude": 13.7628, "longitude": 100.5371},
            {"place_name": "Chulalongkorn Hospital", "latitude": 13.7314, "longitude": 100.5342},
            {"place_name": "Siam Paragon", "latitude": 13.7460, "longitude": 100.5348},
            {"place_name": "Siriraj Hospital", "latitude": 13.7588, "longitude": 100.4866},
            {"place_name": "CentralWorld", "latitude": 13.7469, "longitude": 100.5395}
        ])

    bts_list = [{'clean_name': k, 'lat': v[0], 'lng': v[1]} for k, v in bts_line.items()]
    df_bts = pd.DataFrame(bts_list)
    df_bts['มีลิฟต์'] = 1
    df_bts['ทางลาดสำหรับรถเข็น'] = 1
    return df_places, df_bts

df_places, df_bts_master = load_data()

# ─── AREA 4: AI MODEL ENGINE ───────────────────────────────────────────────
@st.cache_resource
def train_ai_accessibility():
    np.random.seed(42)
    sample_size = 500
    sim_lift = np.random.choice([0, 1], size=sample_size, p=[0.1, 0.9])
    sim_ramp = np.random.choice([0, 1], size=sample_size, p=[0.1, 0.9])
    sim_dist = np.random.uniform(20, 2500, size=sample_size)
    sim_mode = np.random.choice(['BTS', 'BUS', 'CAR', 'AMBULANCE', 'WALK'], size=sample_size)

    sim_labels = []
    for i in range(sample_size):
        score = 1.0
        if sim_mode[i] == 'WALK' and sim_dist[i] > 500: score -= 0.5
        if sim_lift[i] == 0: score -= 0.3
        if sim_ramp[i] == 0: score -= 0.2
        sim_labels.append(1 if score >= 0.5 else 0)

    df_train = pd.DataFrame({'Has_Lift': sim_lift, 'Has_Ramp': sim_ramp, 'Pedestrian_Dist': sim_dist, 'Travel_Mode': sim_mode, 'Safety': sim_labels})
    le = LabelEncoder()
    df_train['Travel_Mode_enc'] = le.fit_transform(df_train['Travel_Mode'])
    features = ['Has_Lift', 'Has_Ramp', 'Pedestrian_Dist', 'Travel_Mode_enc']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df_train[features], df_train['Safety'])
    return model, le, features

ai_model, ai_le, ai_features = train_ai_accessibility()

# ─── AREA 5: SIDEBAR NAVIGATION & MODES (GOOGLE MAPS STYLE) ───────────────
st.sidebar.markdown("## 🚗 Google Maps Route Control")

place_names = df_places['place_name'].tolist()
start_place = st.sidebar.selectbox("📍 จุดเริ่มต้น (Origin):", place_names, index=0)
end_place = st.sidebar.selectbox("🏁 จุดหมายปลายทาง (Destination):", place_names, index=min(1, len(place_names)-1))

start_info = df_places[df_places['place_name'] == start_place].iloc[0]
end_info = df_places[df_places['place_name'] == end_place].iloc[0]

st.sidebar.write("---")
st.sidebar.markdown("### 🔀 เลือกโหมดการเดินทาง (Travel Mode)")
selected_mode_label = st.sidebar.radio(
    "ระบบประมวลผลการเดินทางรองรับรถเข็น:",
    [
        "🚗 รถยนต์ / แท็กซี่ (Driving)",
        "🚇 รถไฟฟ้า BTS (Wheelchair Transit)",
        "🚌 รถเมล์ชานต่ำ (Low-Floor Bus)",
        "🚶 ทางเดินเท้า / รถเข็นไฟฟ้า (Wheelchair/Walk)",
        "🚑 รถพยาบาล / บริการฉุกเฉิน (Ambulance Service)"
    ]
)

# แมปประเภทโหมด
if "🚗" in selected_mode_label: mode_code = "driving"
elif "🚇" in selected_mode_label: mode_code = "bts"
elif "🚌" in selected_mode_label: mode_code = "bus"
elif "🚶" in selected_mode_label: mode_code = "foot"
else: mode_code = "ambulance"

st.sidebar.write("---")
st.sidebar.markdown("### ♿ การตั้งค่าอารยสถาปัตย์ (Accessibility Settings)")
req_lift = st.sidebar.checkbox("ต้องการลิฟต์โดยสาร (Elevator Required)", value=True)
req_ramp = st.sidebar.checkbox("ต้องการทางลาดรถเข็น (Ramp Access)", value=True)
low_floor_bus_only = st.sidebar.checkbox("กรองเฉพาะรถเมล์ชานต่ำ 100% (Low-Floor Only)", value=True)

# ─── AREA 6: ROUTE COMPUTATION & MAP DISPLAY ──────────────────────────────
col1, col2 = st.columns([1.1, 1.9])

m = folium.Map(
    location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2],
    zoom_start=13, tiles='CartoDB Positron'
)

# Marker ต้นทาง/ปลายทาง
folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"<b>ต้นทาง:</b> {start_place}", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"<b>ปลายทาง:</b> {end_place}", icon=folium.Icon(color='red', icon='flag', prefix='fa')).add_to(m)

total_dist_m = 0
total_time_sec = 0

with col1:
    st.markdown("### 🧭 คำแนะนำเส้นทาง (Navigation Summary)")
    
    # 🚗 โหมด 1: Driving / Taxi
    if mode_code == "driving":
        route_coords, total_dist_m, total_time_sec = get_osrm_route(start_info['latitude'], start_info['longitude'], end_info['latitude'], end_info['longitude'], mode="driving")
        folium.PolyLine(route_coords, color='#4285F4', weight=7, opacity=0.8, tooltip="เส้นทางขับรถยนต์").add_to(m)
        
        st.info(f"🚗 **เดินทางด้วย รถยนต์ / แท็กซี่**")
        st.markdown(f"""
        * 📏 **ระยะทางรวม:** `{total_dist_m/1000:.2f} กม.`
        * ⏱️ **เวลาเดินทางโดยประมาณ:** `{int(total_time_sec//60)} นาที`
        * ♿ **คำแนะนำรถเข็น:** แนะนำเรียกรถแท็กซี่คันใหญ่ (SUV/MUV) หรือรถบริการที่มีท้ายเก็บรถเข็น folding wheelchair ได้สะดวก
        """)

    # 🚇 โหมด 2: BTS Transit
    elif mode_code == "bts":
        # หา BTS ต้นทาง-ปลายทางที่ใกล้ที่สุด
        df_bts_master['dist_s'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        df_bts_master['dist_e'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        
        bts_s = df_bts_master.sort_values('dist_s').iloc[0]
        bts_e = df_bts_master.sort_values('dist_e').iloc[0]
        
        # 1. เดินเท้าไป BTS ต้นทาง
        leg1_coords, leg1_dist, leg1_time = get_osrm_route(start_info['latitude'], start_info['longitude'], bts_s['lat'], bts_s['lng'], mode="foot")
        # 2. นั่ง BTS
        bts_coords, bts_dist, bts_time = get_osrm_route(bts_s['lat'], bts_s['lng'], bts_e['lat'], bts_e['lng'], mode="driving")
        # 3. เดินเท้าไปจุดหมาย
        leg3_coords, leg3_dist, leg3_time = get_osrm_route(bts_e['lat'], bts_e['lng'], end_info['latitude'], end_info['longitude'], mode="foot")
        
        total_dist_m = leg1_dist + bts_dist + leg3_dist
        total_time_sec = leg1_time + (bts_dist / 1000 / 35 * 3600) + leg3_time  # BTS เฉลี่ย 35 km/h
        
        folium.PolyLine(leg1_coords, color='#34A853', weight=5, dash_array='4, 8', tooltip="เข็นรถเข็นไปสถานี BTS").add_to(m)
        folium.PolyLine(bts_coords, color='#0F9D58', weight=8, tooltip="รถไฟฟ้า BTS").add_to(m)
        folium.PolyLine(leg3_coords, color='#EA4335', weight=5, dash_array='4, 8', tooltip="เข็นรถเข็นไปจุดหมาย").add_to(m)
        
        folium.Marker([bts_s['lat'], bts_s['lng']], popup=f"BTS {bts_s['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([bts_e['lat'], bts_e['lng']], popup=f"BTS {bts_e['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)

        st.success(f"🚇 **เดินทางด้วย รถไฟฟ้า BTS (Accessibility Integrated)**")
        st.markdown(f"""
        * 🟢 **ช่วงที่ 1:** เข็นรถเข็นไป **สถานี BTS {bts_s['clean_name']}** ({leg1_dist:.0f} ม.)
        * 🔵 **ช่วงที่ 2:** ขึ้น BTS จาก **{bts_s['clean_name']}** → **{bts_e['clean_name']}** (มีลิฟต์บริการ 🛗)
        * 🔴 **ช่วงที่ 3:** เข็นรถเข็นจากสถานีไปยัง **{end_place}** ({leg3_dist:.0f} ม.)
        * ⏱️ **เวลารวมโดยประมาณ:** `{int(total_time_sec//60)} นาที`
        """)

    # 🚌 โหมด 3: Low-Floor Bus
    elif mode_code == "bus":
        route_coords, total_dist_m, total_time_sec = get_osrm_route(start_info['latitude'], start_info['longitude'], end_info['latitude'], end_info['longitude'], mode="driving")
        folium.PolyLine(route_coords, color='#9C27B0', weight=7, tooltip="เส้นทางรถเมล์ชานต่ำ").add_to(m)
        
        st.warning("🚌 **เดินทางด้วย รถเมล์ชานต่ำ (Thai Smile Bus / BMTA Low-Floor)**")
        st.markdown(f"""
        * 🚌 **สายที่แนะนำ:** สาย 8, 28, 515, 1-36 (ชานต่ำมี Ramp ไฮโดรลิก ♿)
        * 📏 **ระยะทาง:** `{total_dist_m/1000:.2f} กม.`
        * ⏱️ **เวลาโดยประมาณ:** `{int((total_time_sec*1.2)//60)} นาที` (รวมเวลารอรถ)
        * ♿ **ฟีเจอร์ชานต่ำ:** ตัวรถปรับระดับเอียงลงเทียบฟุตบาทได้ สะดวกต่อการเข็นขึ้น-ลง
        """)

    # 🚶 โหมด 4: Wheelchair / Walking
    elif mode_code == "foot":
        route_coords, total_dist_m, total_time_sec = get_osrm_route(start_info['latitude'], start_info['longitude'], end_info['latitude'], end_info['longitude'], mode="foot")
        folium.PolyLine(route_coords, color='#FBBC05', weight=6, tooltip="เส้นทางเข็นรถเข็น / ทางเท้า").add_to(m)
        
        st.info("🚶 **เดินทางด้วย รถเข็นวีลแชร์ / ทางเดินเท้า**")
        st.markdown(f"""
        * 📏 **ระยะทางเข็นรวม:** `{total_dist_m:.0f} เมตร`
        * ⏱️ **เวลาเข็นโดยประมาณ:** `{int(total_time_sec//60)} นาที`
        * ⚠️ **ข้อควรระวัง:** ตรวจสอบพื้นผิวทางเท้าและทางลาดข้ามถนนบริเวณสี่แยก
        """)

    # 🚑 โหมด 5: Ambulance / Emergency
    elif mode_code == "ambulance":
        route_coords, total_dist_m, total_time_sec = get_osrm_route(start_info['latitude'], start_info['longitude'], end_info['latitude'], end_info['longitude'], mode="ambulance")
        folium.PolyLine(route_coords, color='#EA4335', weight=9, opacity=0.9, tooltip="เส้นทางฉุกเฉิน Fast-Track รถพยาบาล").add_to(m)
        
        st.error("🚑 **โหมดรถพยาบาลฉุกเฉิน / บริการสวัสดิการการแพทย์ (Fast-Track)**")
        st.markdown(f"""
        * 🚨 **สถานะ:** เปิดสัญญาณฉุกเฉิน / วิ่งบนช่องทางด่วนพิเศษ
        * 📏 **ระยะทาง:** `{total_dist_m/1000:.2f} กม.`
        * ⚡ **เวลาเดินทางฉุกเฉิน:** `{int(total_time_sec//60)} นาที` *(เร็วกว่าปกติ 35%)*
        * 📞 **สายด่วนเรียกรถพยาบาล:** **1669** (สพฉ.) หรือ **1555** (กทม. วีลแชร์สวัสดิการ)
        """)

    # 🧠 AI SAFETY ASSESSMENT
    st.markdown("---")
    st.markdown("### 🤖 AI Safety Assessment")
    
    encoded_mode = ai_le.transform(['BTS' if mode_code == 'bts' else ('BUS' if mode_code == 'bus' else 'CAR')])[0]
    input_vector = pd.DataFrame([[1 if req_lift else 0, 1 if req_ramp else 0, total_dist_m, encoded_mode]], columns=ai_features)
    
    pred = ai_model.predict(input_vector)[0]
    prob = ai_model.predict_proba(input_vector)[0]
    
    if pred == 1:
        st.success(f"🟢 **เส้นทางนี้มีความปลอดภัยสูงสำหรับรถเข็น ({prob[1]*100:.1f}%)**")
    else:
        st.warning(f"🔴 **พบความเสี่ยงในบางจุดสัญจร (ความปลอดภัย {prob[1]*100:.1f}%)**")

with col2:
    st.markdown(f"### 🗺️ แผนที่นำทาง Google Maps Canvas (`{start_place}` ➔ `{end_place}`)")
    st_folium(m, width="100%", height=650)
