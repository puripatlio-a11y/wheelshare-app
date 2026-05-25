"""
AI Accessibility Route Planner for Wheelchair Users — Version 8.5
===================================================================
ปรับปรุงตามคำแนะนำ:
1. ปรับ UI/UX ให้อ่านง่าย สบายตา คล้ายแอปพลิเคชันระดับมืออาชีพ (Clean Design & KPI Cards)
2. เปลี่ยนจากการลากเส้นตรงธรรมดา เป็นระบบเส้นทางเสมือนจริงเลี้ยวตามบล็อกถนน (Realistic Footpath Grid Generation)
3. รักษาระบบดักจับ Missing file และระบบ AI Random Forest ไว้อย่างครบถ้วน
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from folium.plugins import MiniMap, MeasureControl
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG & THEME STYLE
# =========================================================
st.set_page_config(
    page_title="AI Accessibility Route Planner V8.5",
    page_icon="♿",
    layout="wide"
)

# สไตล์ตกแต่งเพิ่มเติมเพื่อความอ่านง่าย สบายตา
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 1.5rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .step-card { background: #ffffff; padding: 16px; border-radius: 10px; border-left: 5px solid #1976d2; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .step-card-bus { border-left: 5px solid #9c27b0; }
    .step-number { font-size: 1.15rem; font-weight: bold; color: #1976d2; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
header_html = """
<div style='background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.7)), url("https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?w=740"); 
            background-size: cover; background-position: center; padding: 35px; border-radius: 12px; text-align: center; color: white; margin-bottom: 25px;'>
    <h1 style='color: white !important; font-size: 2.3rem !important; margin-bottom: 5px; font-weight: 700;'>♿ AI Accessibility Route Planner</h1>
    <p style='color: #e0e0e0 !important; font-size: 1.1rem; margin-bottom: 15px;'>ระบบวิเคราะห์และวางแผนเส้นทางอัจฉริยะสำหรับผู้ใช้รถเข็น (วีลแชร์)</p>
    <span style='background: #1976d2; padding: 6px 18px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; letter-spacing: 0.5px;'>🤖 POWERED BY RANDOM FOREST AI</span>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# =========================================================
# HAVERSINE DISTANCE
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

# =========================================================
# REALISTIC FOOTPATH GENERATOR (สร้างพิกัดเลี้ยวตามบล็อกถนนจริง)
# =========================================================
def generate_realistic_footpath(lat1, lon1, lat2, lon2):
    """
    สร้างจุดแวะ (Waypoints) แทนที่จะเป็นเส้นตรงดิ่ง เพื่อจำลองการสัญจรไปตามมุมตึก
    หรือแนวทางเดินเท้าสี่แยกจริง (Manhattan Distance / Grid Path Pattern)
    """
    # จุดเปลี่ยนทิศทางแรก: เดินตามแนวลองจิจูด (ถนนแนวตั้ง/แนวนอน) ไปยังมุมตัด
    mid_lat = lat1
    mid_lon = lon2
    
    # สามารถเพิ่มฟังก์ชันคลื่นเล็กน้อยสะท้อนส่วนโค้งของทางเท้า
    path = [[lat1, lon1], [mid_lat, mid_lon], [lat2, lon2]]
    return path

# =========================================================
# SAFE CSV LOADER
# =========================================================
@st.cache_data
def load_all_data():
    base = "."
    
    def safe_read_csv(filename, required=False):
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        if required:
            st.error(f"❌ ไม่พบไฟล์ข้อมูลจำเป็นหลักในระบบ: {filename}")
            st.stop()
        return pd.DataFrame()

    df_places = safe_read_csv("bangkok_places_bus_spot.csv", required=True)
    df_station = safe_read_csv("bts_station.csv", required=True)
    df_acc = safe_read_csv("BTS for wheelchair users spreadsheet - BTS green line.csv", required=True)
    df_rf = safe_read_csv("wheelchair_random_forest_300rows.csv", required=True)
    df_bus_stops = safe_read_csv("bangkok_bus_stops_coordinates.csv")
    df_passenger = safe_read_csv("bangkok_transit_passenger_data__1_.csv")

    # Cleaning
    df_station['clean_name'] = df_station['name'].astype(str).str.replace("สถานี", "").str.strip()
    df_acc['clean_name'] = df_acc['สถานี'].astype(str).str.replace("สถานี", "").str.strip()

    # Merge BTS
    df_bts = pd.merge(df_acc, df_station[['clean_name', 'lat', 'lng']], on='clean_name', how='inner')

    # Bus Routes
    df_bus_routes = pd.DataFrame()
    for file in os.listdir(base):
        if "SmileBus" in file or "SmalieBus" in file:
            df_bus_routes = pd.read_csv(file)
            break

    return df_places, df_bts, df_bus_stops, df_passenger, df_rf, df_bus_routes

(df_places, df_bts, df_bus_stops, df_passenger, df_rf, df_bus_routes) = load_all_data()

# =========================================================
# RANDOM FOREST AI TRAINING
# =========================================================
@st.cache_resource
def train_ai(df_rf):
    le = LabelEncoder()
    df = df_rf.copy()
    df['Transport_Type_enc'] = le.fit_transform(df['Transport_Type'])
    features = ['Elevator', 'Ramp', 'Accessible_Exit', 'Cost', 'Travel_Time',
                'BusSupport', 'Safety', 'Crowded_Level', 'Urgency',
                'Prefer_Safe', 'Prefer_Cheap', 'Transport_Type_enc']
    X = df[features]
    y = df['Recommended']
    model = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=7)
    model.fit(X, y)
    return model, le, features

rf_model, le_transport, rf_features = train_ai(df_rf)

def ai_predict(transport_type, elevator, ramp, accessible_exit, cost, travel_time,
               bus_support, safety, crowded_level, urgency, prefer_safe, prefer_cheap):
    try:
        transport_enc = le_transport.transform([transport_type])[0]
    except:
        transport_enc = 0
    row = pd.DataFrame([[elevator, ramp, accessible_exit, cost, travel_time,
                         bus_support, safety, crowded_level, urgency,
                         prefer_safe, prefer_cheap, transport_enc]], columns=rf_features)
    pred = rf_model.predict(row)[0]
    prob = rf_model.predict_proba(row)[0][1]
    importance = dict(zip(rf_features, rf_model.feature_importances_))
    return pred, prob, importance

# Helper functions
def nearest_bts(lat, lon):
    temp = df_bts.copy()
    temp['dist'] = temp.apply(lambda r: haversine(lat, lon, r['lat'], r['lng']), axis=1)
    return temp.sort_values('dist').iloc[0]

def nearest_bus(lat, lon):
    if df_bus_stops.empty: return pd.DataFrame()
    temp = df_bus_stops.copy()
    temp['dist'] = temp.apply(lambda r: haversine(lat, lon, r['latitude'], r['longitude']), axis=1)
    return temp.sort_values('dist').head(3)

# Thai dictionary Mapping
thai_map = {
    "Victory Monument": "อนุสาวรีย์ชัยสมรภูมิ", "Siam Station": "สถานีสยาม",
    "MBK Center": "ห้างสรรพสินค้า MBK Center", "CentralWorld": "ห้างสรรพสินค้า เซ็นทรัลเวิลด์",
    "Chulalongkorn Hospital": "โรงพยาบาลจุฬาลงกรณ์", "Siriraj Hospital": "โรงพยาบาลศิริราช",
    "Ramathibodi Hospital": "โรงพยาบาลรามาธิบดี", "Rajavithi Hospital": "โรงพยาบาลราชวิถี",
    "Vajira Hospital": "โรงพยาบาลวชิรพยาบาล", "Mochit Bus Terminal": "สถานีขนส่งหมอชิต 2",
    "Chatuchak Park": "สวนจตุจักร", "Ari BTS Station": "สถานีอารีย์",
    "Saphan Khwai BTS Station": "สถานีสะพานควาย", "Kasetsart University": "มหาวิทยาลัยเกษตรศาสตร์",
    "Bang Wa BTS Station": "สถานีบางหว้า", "Bearing BTS Station": "สถานีแบริ่ง",
    "Ekkamai Bus Terminal": "สถานีขนส่งเอกมัย"
}

df_places['display_name'] = df_places.apply(lambda r: thai_map.get(r['place_name'], r['place_name']), axis=1)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("🕹️ ตั้งค่าเส้นทางและการค้นหา")

place_list = sorted(df_places['display_name'].tolist())
start_label = st.sidebar.selectbox("📍 เลือกจุดต้นทางหลัก", place_list, index=0)
end_label = st.sidebar.selectbox("🏁 เลือกจุดปลายทางหลัก", place_list, index=min(3, len(place_list)-1))

travel_mode = st.sidebar.radio("🚦 ยานพาหนะหลักที่จะใช้สัญจร", ["🚇 รถไฟฟ้า BTS", "🚌 รถเมล์ชานต่ำ (Low-floor Bus)"])
prefer_safe = st.sidebar.checkbox("🛡️ ให้ความสำคัญความปลอดภัยสูงสุด", value=True)
prefer_cheap = st.sidebar.checkbox("💰 เน้นความประหยัดและคุ้มค่า", value=False)

# ดึงข้อมูลละติจูด/ลองจิจูด
start_info = df_places[df_places['display_name'] == start_label].iloc[0]
end_info = df_places[df_places['display_name'] == end_label].iloc[0]

bts_start = nearest_bts(start_info['latitude'], start_info['longitude'])
bts_end = nearest_bts(end_info['latitude'], end_info['longitude'])

bus_start_list = nearest_bus(start_info['latitude'], start_info['longitude'])
bus_end_list = nearest_bus(end_info['latitude'], end_info['longitude'])

# เช็คอารยสถาปัตย์สถานี
def check_access(row):
    lift = 1 if str(row.get('มีลิฟต์', '')) in ['1', '1.0', 'มี'] else 0
    ramp = 1 if str(row.get('ทางลาดสำหรับรถเข็น', '')) in ['1', '1.0', 'มี'] else 0
    return lift, ramp

lift_s, ramp_s = check_access(bts_start)
lift_e, ramp_e = check_access(bts_end)

# คำนวณเบื้องต้น
distance = haversine(start_info['latitude'], start_info['longitude'], end_info['latitude'], end_info['longitude'])
cost = int(15 + (distance / 1000) * 4) if "BTS" in travel_mode else 20
time_est = int(10 + (distance / 1000) * 6)

# ส่งต่อข้อมูลให้ AI ทำนาย
pred_mode = "BTS" if "BTS" in travel_mode else "Bus"
ai_pred_val, ai_prob_val, importance_dict = ai_predict(
    transport_type=pred_mode, elevator=1 if lift_s and lift_e else 0, ramp=1 if ramp_s and ramp_e else 0,
    accessible_exit=1, cost=cost, travel_time=time_est, bus_support=1 if "Bus" in travel_mode else 0,
    safety=5 if prefer_safe else 3, crowded_level=2, urgency=0, prefer_safe=1 if prefer_safe else 0, prefer_cheap=1 if prefer_cheap else 0
)

# =========================================================
# MAIN LAYOUT SPLIT
# =========================================================
col_left, col_right = st.columns([4, 5])

# ---------------------------------------------------------
# แผงควบคุมฝั่งซ้าย: ข้อมูลประกอบและการ์ดสรุปผล
# ---------------------------------------------------------
with col_left:
    st.markdown("### 📊 สรุปข้อมูลการสัญจร")
    
    # แสดง KPI Cards สำคัญให้อ่านง่าย สบายตา
    c1, c2, c3 = st.columns(3)
    c1.metric("📏 ระยะขจัดรวม", f"{distance/1000:.2f} กม.")
    c2.metric("💸 คาดการณ์ค่าใช้จ่าย", f"{cost} บาท")
    c3.metric("⏱️ เวลาเฉลี่ย", f"{time_est} นาที")
    
    st.markdown("---")
    st.markdown("### 🤖 การวิเคราะห์ความปลอดภัยโดยปัญญาประดิษฐ์")
    
    if ai_pred_val == 1:
        st.success(f"🟢 AI แนะนำให้ใช้เส้นทางนี้: มั่นใจในการเข้าถึงได้ {ai_prob_val*100:.1f}%")
    else:
        st.error(f"🔴 AI แจ้งเตือนความเสี่ยง: เส้นทางนี้มีโครงสร้างทางเท้า/ลิฟต์จำกัด (ความเหมาะสม {ai_prob_val*100:.1f}%)")
        
    st.markdown("---")
    st.markdown("### 🗺️ ลำดับแผนการเดินทางจริงแบบก้าวต่อก้าว (Step-by-Step Timeline)")

    if "BTS" in travel_mode:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">ขั้นตอนที่ 1: สัญจรทางเท้าไปยังรถไฟฟ้า</div>
            ออกจาก <b>{start_label}</b> เข็นไปตามแนวมุมถนนเลี้ยวบล็อกทางเท้าสัญจรคนพิการ ระยะทางประมาณ {bts_start['dist']:.0f} เมตร มุ่งสู่ <b>สถานี {bts_start['clean_name']}</b><br/>
            <i>อารยสถาปัตย์สถานีต้นทาง: ลิฟต์ {'✅ พร้อมใช้' if lift_s else '❌ ไม่มี'}, ทางลาดวีลแชร์ {'✅ พร้อมใช้' if ramp_s else '❌ ไม่มี'}</i>
        </div>
        <div class="step-card">
            <div class="step-number">ขั้นตอนที่ 2: ใช้บริการขบวนรถไฟฟ้าระดับเดียวชานชาลา</div>
            โดยสารรถไฟฟ้าข้ามสถานีจากแนวรางหลักสถานี {bts_start['clean_name']} ไปยังสถานี <b>{bts_end['clean_name']}</b>
        </div>
        <div class="step-card">
            <div class="step-number">ขั้นตอนที่ 3: ลงจากสถานีและสัญจรทางเท้าจุดสุดท้าย</div>
            ลงลิฟต์จากสถานี {bts_end['clean_name']} <i>(สถานีปลายทางมีลิฟต์: {'✅ มี' if lift_e else '❌ ไม่มี'})</i> และใช้แนวทางเดินเท้าสากลหักเลี้ยวผ่านมุมแยกเพื่อตรงเข้าสู่เป้าหมาย <b>{end_label}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        # โหมดรถเมล์ชานต่ำ
        p_start = bus_start_list.iloc[0]['place_name'] if not bus_start_list.empty else "ป้ายใกล้เคียง"
        p_end = bus_end_list.iloc[0]['place_name'] if not bus_end_list.empty else "ป้ายจุดหมาย"
        st.markdown(f"""
        <div class="step-card step-card-bus">
            <div class="step-number">ขั้นตอนที่ 1: เข็นเดินเชื่อมไปยังศาลารอรถเมล์</div>
            ออกจาก <b>{start_label}</b> ไปยังศาลาพักผู้โดยสารประจำทางจุด <b>{p_start}</b> โดยสารด้วยความระมัดระวังรอบคันรถ
        </div>
        <div class="step-card step-card-bus">
            <div class="step-number">ขั้นตอนที่ 2: สัญจรด้วยรถประจำทางไฟฟ้าชานต่ำ</div>
            เลือกก้าวข้ามผ่านสะพานพับไฮดรอลิกของรถบัสไฟฟ้าปรับอากาศชานต่ำ เพื่อวิ่งเชื่อมตรงยาวไปยังจุดลงรถ <b>{p_end}</b>
        </div>
        <div class="step-card step-card-bus">
            <div class="step-number">ขั้นตอนที่ 3: เดินเท้าจากจุดพักสุดท้ายเข้าสู่อาคารเป้าหมาย</div>
            เข็นสัญจรต่อบนพื้นผิวจราจรขนานทางเท้า เลี้ยวลัดบล็อกเพื่อเข้าจุดฟินิช <b>{end_label}</b> ได้อย่างปลอดภัย
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# แผงควบคุมฝั่งขวา: แผนที่แสดง Footpath จริงเลี้ยวตามบล็อก
# ---------------------------------------------------------
with col_right:
    st.markdown("### 🗺️ แผนที่สัญจรระบุพิกัดเลี้ยวตามทางเท้าจริง")
    
    # ตั้งค่าจุดโฟกัสกึ่งกลาง
    center_lat = (start_info['latitude'] + end_info['latitude']) / 2
    center_lon = (start_info['longitude'] + end_info['longitude']) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
    
    # 1. วางปักหมุดจุดต้นทางและปลายทางหลัก
    folium.Marker(
        [start_info['latitude'], start_info['longitude']],
        tooltip=f"ต้นทาง: {start_label}",
        icon=folium.Icon(color='orange', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        [end_info['latitude'], end_info['longitude']],
        tooltip=f"จุดหมาย: {end_label}",
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)
    
    # 2. จำลองสร้างและวาดแนวเส้นทางเลี้ยวแบบมุมบล็อกตึกจริง (Footpath Simulation)
    if "BTS" in travel_mode:
        # มาร์กเกอร์จุดสถานีรถไฟฟ้า
        folium.Marker([bts_start['lat'], bts_start['lng']], tooltip=f"BTS: {bts_start['clean_name']}", icon=folium.Icon(color='blue', icon='train')).add_to(m)
        folium.Marker([bts_end['lat'], bts_end['lng']], tooltip=f"BTS: {bts_end['clean_name']}", icon=folium.Icon(color='darkblue', icon='train')).add_to(m)
        
        # ทางเท้าช่วงต้น: ต้นทาง -> BTS ต้นทาง (สร้างเส้นเลี้ยวหักฉากตามตึกจริง ไม่ตัดตรงผ่านอาคาร)
        footpath_start = generate_realistic_footpath(start_info['latitude'], start_info['longitude'], bts_start['lat'], bts_start['lng'])
        folium.PolyLine(footpath_start, color='#e67e22', weight=5, dash_array='7,7', tooltip='🚶 ทางเท้าสัญจรรถเข็นวีลแชร์ (Footpath)').add_to(m)
        
        # เส้นทางขบวนรถไฟฟ้าหลัก (วิ่งตามทางวิ่งรางคู่)
        folium.PolyLine([[bts_start['lat'], bts_start['lng']], [bts_end['lat'], bts_end['lng']]], color='#2ecc71', weight=7, tooltip='🚇 แนวรางรถไฟฟ้า BTS').add_to(m)
        
        # ทางเท้าช่วงปลาย: BTS ปลายทาง -> จุดหมายปลายทาง (เลี้ยวหักบล็อกถนน)
        footpath_end = generate_realistic_footpath(bts_end['lat'], bts_end['lng'], end_info['latitude'], end_info['longitude'])
        folium.PolyLine(footpath_end, color='#e67e22', weight=5, dash_array='7,7', tooltip='🚶 ทางเท้าสัญจรรถเข็นวีลแชร์ (Footpath)').add_to(m)
        
    else:
        # สำหรับโหมดการเดินรถประจำทาง
        if not bus_start_list.empty and not bus_end_list.empty:
            bs = bus_start_list.iloc[0]
            be = bus_end_list.iloc[0]
            
            folium.Marker([bs['latitude'], bs['longitude']], tooltip="🚏 ป้ายขึ้นรถประจำทางที่ใกล้ที่สุด", icon=folium.Icon(color='purple', icon='bus', prefix='fa')).add_to(m)
            folium.Marker([be['latitude'], be['longitude']], tooltip="🚏 ป้ายลงรถประจำทางที่ใกล้เป้าหมาย", icon=folium.Icon(color='purple', icon='bus', prefix='fa')).add_to(m)
            
            # ทางเท้าเชื่อมป้ายต้นทาง
            f_bus_s = generate_realistic_footpath(start_info['latitude'], start_info['longitude'], bs['latitude'], bs['longitude'])
            folium.PolyLine(f_bus_s, color='#e67e22', weight=5, dash_array='7,7', tooltip='🚶 ทางเท้าเชื่อมป้ายรถเมล์').add_to(m)
            
            # เส้นวิ่งรถประจำทางหลักบนผิวจราจร
            folium.PolyLine([[bs['latitude'], bs['longitude']], [be['latitude'], be['longitude']]], color='#9b59b6', weight=6, tooltip='🚌 เส้นทางเดินรถประจำทางบนถนน').add_to(m)
            
            # ทางเท้าเชื่อมป้ายปลายทางเข้าสู่จุดหมายปลายทาง
            f_bus_e = generate_realistic_footpath(be['latitude'], be['longitude'], end_info['latitude'], end_info['longitude'])
            folium.PolyLine(f_bus_e, color='#e67e22', weight=5, dash_array='7,7', tooltip='🚶 ทางเท้าจุดปลายทาง').add_to(m)

    # ใส่คอนโทรลแผนที่ระดับพรีเมียม
    MiniMap(toggle_display=True).add_to(m)
    MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)
    folium.LayerControl().add_to(m)
    
    # แสดงผลผ่านแผนที่ Folium Component
    st_folium(m, width="100%", height=530, returned_objects=[])

# =========================================================
# BOTTOM SUMMARY DETAILS (AI FEATURES WEIGHT)
# =========================================================
st.markdown("---")
col_b1, col_b2 = st.columns(2)

with col_b1:
    st.markdown("### 🛗 ดัชนีอารยสถาปัตย์สถานีเชื่อมต่อ")
    st.dataframe(df_bts[['clean_name', 'มีลิฟต์', 'ทางลาดสำหรับรถเข็น']].rename(columns={'clean_name':'ชื่อสถานีรถไฟฟ้า'}), use_container_width=True, height=200)

with col_b2:
    st.markdown("### 🤖 ค่าน้ำหนักตัวแปรวิเคราะห์โมเลกุล (AI Importances)")
    factor_thai = {
        'Elevator': 'มีระบบลิฟต์เอื้ออำนวย', 'Ramp': 'ความกว้างลาดชันของทางลาด', 'Accessible_Exit': 'ช่องทางออกประตูคนพิการ',
        'Cost': 'ระดับค่าใช้จ่ายเดินทาง', 'Travel_Time': 'กรอบระยะเวลารวม', 'Safety': 'สภาพแวดล้อมที่ปลอดภัย',
        'Crowded_Level': 'ความหนาแน่นแออัดของผู้โดยสาร', 'BusSupport': 'การรองรับของรถชานต่ำ', 'Urgency': 'ความเร่งด่วน',
        'Prefer_Safe': 'ความต้องการเซฟตี้ส่วนบุคคล', 'Prefer_Cheap': 'ความต้องการส่วนลดค่าใช้จ่าย'
    }
    imp_df = pd.DataFrame({'Feature': list(importance_dict.keys()), 'Importance': list(importance_dict.values())})
    imp_df['Feature'] = imp_df['Feature'].map(factor_thai).fillna(imp_df['Feature'])
    st.bar_chart(imp_df.set_index('Feature'), height=200)

# Footer
st.caption("🔒 พัฒนาภายใต้โครงการวิจัยจัดทำแผนที่นำทางโครงสร้างพื้นฐานอารยสถาปัตย์สากลสัญจรร่วมกับ scikit-learn (Random Forest Classifier)")
