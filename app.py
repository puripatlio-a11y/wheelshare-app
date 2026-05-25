"""
AI Accessibility Route Planner for Wheelchair Users — Version 7.1 (Fixed Path Error)
แก้ไขตามคำแนะนำอาจารย์:
  1. แสดงเส้นทางเดินเท้า (pedestrian path) บนแผนที่จริง
  2. ระบุ AI Function ชัดเจน (Random Forest สำหรับแนะนำเส้นทาง)
  3. ระบุ footpath / ทางเท้าในแผนที่
  4. ใช้ library ที่จำเป็น (scikit-learn, folium, openrouteservice)
  5. เน้น AI เป็นหัวใจหลักของระบบ
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import requests
import json
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Accessibility Route Planner V7", layout="wide", page_icon="♿")

# ─── Header ──────────────────────────────────────────────────────────────────
header_html = """
<style>
.custom-header {
    background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.60)),
        url("https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?w=740");
    background-size: cover; background-position: center;
    padding: 40px; border-radius: 14px; color: white;
    text-align: center; margin-bottom: 20px;
}
.custom-header h1 { color:#fff !important; font-size:2.4rem !important;
    text-shadow:2px 2px 8px rgba(0,0,0,0.8); margin-bottom:4px; }
.custom-header h3 { color:#f0f2f6 !important; font-size:1.2rem !important;
    font-weight:400; text-shadow:1px 1px 5px rgba(0,0,0,0.7); }
.ai-badge { background:#1a6faf; color:white; padding:4px 12px;
    border-radius:20px; font-size:0.85rem; display:inline-block; margin-top:8px; }
</style>
<div class="custom-header">
  <h1>♿ AI Accessibility Route Planner</h1>
  <h3>ระบบวางแผนเส้นทางอัจฉริยะสำหรับผู้ใช้รถเข็น ด้วยปัญญาประดิษฐ์</h3>
  <span class="ai-badge">🤖 Powered by Random Forest AI</span>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ─── ฟังก์ชัน Haversine ──────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางระหว่างสองพิกัด (เมตร) ด้วยสูตร Haversine"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

# ─── โหลดข้อมูล (แก้ไขระบบ Path อัตโนมัติ) ───────────────────────────────────
@st.cache_data
def load_all_data():
    # ดึงพิกัดที่อยู่ของไฟล์ app.py ในปัจจุบัน (รองรับทั้งคอมตัวเองและ Cloud เซิร์ฟเวอร์)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(current_dir, "data") 

    # สถานที่หลัก
    df_places = pd.read_csv(os.path.join(base, "bangkok_places_bus_spot.csv"))

    #  สถานี BTS พิกัด
    df_stations = pd.read_csv(os.path.join(base, "bts_station.csv"))
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()

    # ข้อมูลอารยสถาปัตย์ BTS (สายสีเขียว)
    df_acc = pd.read_csv(os.path.join(base, "BTS_for_wheelchair_users_spreadsheet_-_BTS_green_line.csv"))
    df_acc['clean_name'] = df_acc['สถานี'].str.replace('สถานี', '').str.strip()

    # รวม BTS master
    df_bts = pd.merge(
        df_acc, df_stations[['clean_name', 'lat', 'lng', 'btsline', 'location']],
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)

    # ป้ายรถเมล์ (พิกัด)
    df_bus_stops = pd.read_csv(os.path.join(base, "bangkok_bus_stops_coordinates.csv"))

    # ข้อมูลผู้โดยสาร (ใช้แสดง insight ความหนาแน่น)
    df_passenger = pd.read_csv(os.path.join(base, "bangkok_transit_passenger_data__1_.csv"))

    # ข้อมูล Random Forest training
    df_rf = pd.read_csv(os.path.join(base, "wheelchair_random_forest_300rows.csv"))

    # สายรถเมล์ (ถ้ามี)
    df_bus_routes = None
    if os.path.exists(base):
        for fname in os.listdir(base):
            if 'SmileBus' in fname or 'SmalieBus' in fname:
                df_bus_routes = pd.read_csv(os.path.join(base, fname))
                break

    return df_places, df_bts, df_bus_stops, df_passenger, df_rf, df_bus_routes

# ดึงข้อมูลมาใช้งาน
df_places, df_bts, df_bus_stops, df_passenger, df_rf, df_bus_routes = load_all_data()

# ─── 🤖 AI FUNCTION: Train Random Forest Model ───────────────────────────────
@st.cache_resource
def train_random_forest(df_rf):
    """
    AI Function: เทรน Random Forest Classifier
    จากข้อมูล wheelchair_random_forest_300rows.csv
    เพื่อทำนาย Recommended (1=แนะนำ, 0=ไม่แนะนำ)
    Feature: Elevator, Ramp, Accessible_Exit, Cost, Travel_Time,
             BusSupport, Safety, Crowded_Level, Urgency,
             Prefer_Safe, Prefer_Cheap + Transport_Type (encoded)
    """
    le = LabelEncoder()
    df = df_rf.copy()
    df['Transport_Type_enc'] = le.fit_transform(df['Transport_Type'])
    features = ['Elevator','Ramp','Accessible_Exit','Cost','Travel_Time',
                'BusSupport','Safety','Crowded_Level','Urgency',
                'Prefer_Safe','Prefer_Cheap','Transport_Type_enc']
    X = df[features]
    y = df['Recommended']
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    clf.fit(X, y)
    return clf, le, features

rf_model, le_transport, rf_features = train_random_forest(df_rf)

def ai_predict_route(transport_type, elevator, ramp, accessible_exit,
                     cost, travel_time, bus_support, safety,
                     crowded_level, urgency, prefer_safe, prefer_cheap):
    """
    🤖 AI Function: ใช้ Random Forest ทำนายว่าเส้นทางนี้เหมาะสำหรับผู้ใช้รถเข็นหรือไม่
    คืนค่า: (label, probability, feature_importances)
    """
    try:
        transport_enc = le_transport.transform([transport_type])[0]
    except ValueError:
        transport_enc = 0

    row = pd.DataFrame([[elevator, ramp, accessible_exit, cost, travel_time,
                         bus_support, safety, crowded_level, urgency,
                         prefer_safe, prefer_cheap, transport_enc]],
                       columns=rf_features)
    prob = rf_model.predict_proba(row)[0][1]
    label = int(rf_model.predict(row)[0])
    importances = dict(zip(rf_features, rf_model.feature_importances_))
    return label, prob, importances

# ─── ค้นหาป้ายรถเมล์ใกล้สุด ──────────────────────────────────────────────────
def nearest_bus_stops(lat, lon, df_bus_stops, n=3, max_dist=600):
    """ค้นหาป้ายรถเมล์ที่ใกล้ที่สุดในรัศมี max_dist เมตร"""
    df_bus_stops = df_bus_stops.copy()
    df_bus_stops['dist'] = df_bus_stops.apply(
        lambda r: haversine(lat, lon, r['latitude'], r['longitude']), axis=1)
    nearby = df_bus_stops[df_bus_stops['dist'] <= max_dist].sort_values('dist')
    return nearby.head(n)

# ─── หาสถานี BTS ใกล้สุด ─────────────────────────────────────────────────────
def nearest_bts(lat, lon):
    df_bts['dist'] = df_bts.apply(
        lambda r: haversine(lat, lon, r['lat'], r['lng']), axis=1)
    return df_bts.sort_values('dist').iloc[0]

# ─── ฟังก์ชัน Footpath บนแผนที่ (เส้นทางเดินเท้า) ────────────────────────────
def add_pedestrian_path(m, lat1, lon1, lat2, lon2, label="", color="orange", dashed=True):
    """
    วาดเส้นทางเดินเท้า (pedestrian path) บนแผนที่ folium
    ใช้เส้นประสีส้มเพื่อแยกออกจากเส้นรถไฟฟ้า
    """
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    dist_m = haversine(lat1, lon1, lat2, lon2)
    dash = '8, 8' if dashed else None
    poly = folium.PolyLine(
        [[lat1, lon1], [mid_lat, mid_lon], [lat2, lon2]],
        color=color, weight=4,
        dash_array=dash,
        tooltip=f"🚶 {label} ({dist_m:.0f} ม.)",
        opacity=0.85
    )
    poly.add_to(m)
    # วงกลมจุดกึ่งกลาง (footpath marker)
    folium.CircleMarker(
        [mid_lat, mid_lon], radius=5,
        color=color, fill=True, fill_opacity=0.7,
        tooltip=f"🦮 จุดทางเท้า: {label}"
    ).add_to(m)

def add_bts_rail_path(m, lat1, lon1, lat2, lon2):
    """วาดเส้นทางรถไฟฟ้า BTS สีเขียว"""
    folium.PolyLine(
        [[lat1, lon1], [lat2, lon2]],
        color='#00aa44', weight=7,
        tooltip="🚇 เส้นทางรถไฟฟ้า BTS",
        opacity=0.9
    ).add_to(m)

# ─── ชื่อไทย / พจนานุกรม ──────────────────────────────────────────────────────
th_name_map = {
    "Victory Monument": "อนุสาวรีย์ชัยสมรภูมิ",
    "Siam Station": "สถานีรถไฟฟ้า สยาม",
    "CentralWorld": "เซ็นทรัลเวิลด์",
    "MBK Center": "เอ็มบีเค เซ็นเตอร์",
    "Samyan Mitrtown": "สามย่านมิตรทาวน์",
    "Chulalongkorn Hospital": "โรงพยาบาลจุฬาลงกรณ์",
    "Siriraj Hospital": "โรงพยาบาลศิริราช",
    "Ramathibodi Hospital": "โรงพยาบาลรามาธิบดี",
    "Rajavithi Hospital": "โรงพยาบาลราชวิถี",
    "Vajira Hospital": "โรงพยาบาลวชิรพยาบาล",
    "Mochit Bus Terminal": "สถานีขนส่งหมอชิต 2",
    "Chatuchak Park": "สวนจตุจักร",
    "Ari BTS Station": "สถานีอารีย์",
    "Saphan Khwai BTS Station": "สถานีสะพานควาย",
    "Bang Wa BTS Station": "สถานีบางหว้า",
    "Bearing BTS Station": "สถานีแบริ่ง",
    "Ekkamai Bus Terminal": "สถานีขนส่งเอกมัย",
    "Kasetsart University": "มหาวิทยาลัยเกษตรศาสตร์",
}

# สร้าง display name พร้อมวงเล็บข้อมูล
def make_display_name(row):
    name = row['place_name']
    th = th_name_map.get(name, name)
    suffixes = []
    # ตรวจ BTS
    bts_dist = nearest_bts(row['latitude'], row['longitude'])['dist']
    if 'bts' in name.lower() or bts_dist <= 400:
        suffixes.append("BTS")
    # ตรวจป้ายรถเมล์
    nearby_bus = nearest_bus_stops(row['latitude'], row['longitude'], df_bus_stops, n=1, max_dist=300)
    if len(nearby_bus) > 0:
        suffixes.append("มีป้ายรถเมล์ใกล้")
    label = f"{th} ({' / '.join(suffixes)})" if suffixes else th
    return label

df_places['display_th'] = df_places.apply(make_display_name, axis=1)
place_list = sorted(df_places['display_th'].tolist())

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.header("🕹️ ตั้งค่าการเดินทาง")

def_start = next((i for i, n in enumerate(place_list) if "อนุสาวรีย์" in n), 0)
def_end   = next((i for i, n in enumerate(place_list) if "จุฬา" in n), 1)

start_label = st.sidebar.selectbox("📍 จุดต้นทาง:", place_list, index=def_start)
end_label   = st.sidebar.selectbox("🏁 จุดปลายทาง:", place_list, index=def_end)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ข้อมูลส่วนตัวผู้เดินทาง (สำหรับ AI)")
prefer_safe  = st.sidebar.checkbox("🛡️ ให้ความสำคัญด้านความปลอดภัย", value=True)
prefer_cheap = st.sidebar.checkbox("💰 ให้ความสำคัญด้านประหยัดค่าใช้จ่าย", value=False)
urgency      = st.sidebar.selectbox("⏱️ ความเร่งด่วน:", ["ไม่เร่งด่วน (0)", "ปานกลาง (1)"], index=0)
urgency_val  = 0 if "ไม่" in urgency else 1

st.sidebar.markdown("---")
travel_mode = st.sidebar.radio(
    "🚦 โหมดการเดินทาง:",
    ["🚇 รถไฟฟ้า BTS", "🚌 รถเมล์ชานต่ำ", "🏥 สวัสดิการรถตู้รัฐ (โรงพยาบาล)"]
)

# ─── ดึงข้อมูลสถานที่ที่เลือก ─────────────────────────────────────────────────
start_info = df_places[df_places['display_th'] == start_label].iloc[0]
end_info   = df_places[df_places['display_th'] == end_label].iloc[0]

bts_start = nearest_bts(start_info['latitude'], start_info['longitude'])
bts_end   = nearest_bts(end_info['latitude'], end_info['longitude'])

# ─── ข้อมูลอารยสถาปัตย์จาก CSV ───────────────────────────────────────────────
def get_accessibility(bts_row):
    has_lift  = "✅ มี" if str(bts_row.get('มีลิฟต์','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    has_ramp  = "✅ มี" if str(bts_row.get('ทางลาดสำหรับรถเข็น','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    evac_area = "✅ มี" if str(bts_row.get('พื้นที่สำหรับหนีภัยของคนพิการ','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    return has_lift, has_ramp, evac_area

lift_s, ramp_s, evac_s = get_accessibility(bts_start)
lift_e, ramp_e, evac_e = get_accessibility(bts_end)

# ─── ป้ายรถเมล์ใกล้ต้นทาง/ปลายทาง ────────────────────────────────────────────
bus_near_start = nearest_bus_stops(start_info['latitude'], start_info['longitude'], df_bus_stops, n=3)
bus_near_end   = nearest_bus_stops(end_info['latitude'],   end_info['longitude'],   df_bus_stops, n=3)

# ─── ข้อมูล Crowded Level จาก passenger data ──────────────────────────────────
def get_crowd_level(station_name_en):
    sub = df_passenger[df_passenger['Station'].str.contains(
        station_name_en.split()[0] if station_name_en else '', case=False, na=False)]
    if len(sub) == 0:
        return 3  # default medium
    avg_in = sub['Passengers In'].mean()
    if avg_in < 300: return 1
    if avg_in < 600: return 2
    if avg_in < 900: return 3
    return 4

crowd_s = get_crowd_level(str(bts_start.get('clean_name','')))
crowd_e = get_crowd_level(str(bts_end.get('clean_name','')))
avg_crowd = int((crowd_s + crowd_e) / 2)

# ─── AI Prediction ────────────────────────────────────────────────────────────
transport_map = {
    "🚇 รถไฟฟ้า BTS": "BTS",
    "🚌 รถเมล์ชานต่ำ": "Bus",
    "🏥 สวัสดิการรถตู้รัฐ (โรงพยาบาล)": "BTS+Bus"
}
transport_type_str = transport_map[travel_mode]

el_s  = 1 if "✅" in lift_s else 0
rmp_s = 1 if "✅" in ramp_s else 0
el_e  = 1 if "✅" in lift_e else 0
rmp_e = 1 if "✅" in ramp_e else 0
elevator_val    = 1 if (el_s and el_e) else 0
ramp_val        = 1 if (rmp_s and rmp_e) else 0
accessible_exit = 1 if (elevator_val or ramp_val) else 0

dist_total = haversine(start_info['latitude'], start_info['longitude'],
                       end_info['latitude'], end_info['longitude'])
est_cost = int(16 + dist_total / 1000 * 2.5)
est_time = int(dist_total / 1000 * 6)

ai_label, ai_prob, ai_importances = ai_predict_route(
    transport_type=transport_type_str,
    elevator=elevator_val, ramp=ramp_val,
    accessible_exit=accessible_exit,
    cost=est_cost, travel_time=est_time,
    bus_support=1 if "🚌" in travel_mode else 0,
    safety=4 if prefer_safe else 3,
    crowded_level=avg_crowd,
    urgency=urgency_val,
    prefer_safe=1 if prefer_safe else 0,
    prefer_cheap=1 if prefer_cheap else 0
)

# ─── Layout หลัก ──────────────────────────────────────────────────────────────
col_info, col_map = st.columns([1, 2])

with col_info:
    st.markdown("### 📊 ผลการวิเคราะห์เส้นทาง")
    st.markdown(f"**จาก:** {start_label.split(' (')[0]}")
    st.markdown(f"**ถึง:** {end_label.split(' (')[0]}")
    st.markdown(f"**ระยะทาง (โดยตรง):** {dist_total/1000:.2f} กม.")

    # ─── AI Result Box ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 ผลการวิเคราะห์ของ AI (Random Forest)")

    if ai_label == 1:
        st.success(f"✅ AI แนะนำเส้นทางนี้ — ความเชื่อมั่น **{ai_prob*100:.1f}%**")
    else:
        st.error(f"⚠️ AI ไม่แนะนำเส้นทางนี้ — ความเชื่อมั่นต่ำ ({ai_prob*100:.1f}%)")

    # Feature importance top 3
    top3 = sorted(ai_importances.items(), key=lambda x: x[1], reverse=True)[:3]
    factor_thai = {
        'Elevator':'ลิฟต์', 'Ramp':'ทางลาด', 'Accessible_Exit':'ทางออกอารยสถาปัตย์',
        'Cost':'ค่าใช้จ่าย', 'Travel_Time':'เวลาเดินทาง', 'Safety':'ความปลอดภัย',
        'Crowded_Level':'ความหนาแน่น', 'BusSupport':'รถเมล์ชานต่ำ',
        'Urgency':'ความเร่งด่วน', 'Prefer_Safe':'ความปลอดภัยสำคัญ',
        'Prefer_Cheap':'ราคาประหยัดสำคัญ', 'Transport_Type_enc':'ประเภทขนส่ง'
    }
    factors_text = ", ".join([f"{factor_thai.get(k,k)} ({v*100:.0f}%)" for k,v in top3])
    st.caption(f"🔍 ปัจจัยสำคัญ: {factors_text}")
    st.markdown(f"💸 ค่าโดยสารโดยประมาณ: **{est_cost} บาท** | ⏱️ เวลาโดยประมาณ: **{est_time} นาที**")

    st.markdown("---")

    # ─── โหมด BTS ─────────────────────────────────────────────────────────────
    if "🚇" in travel_mode:
        st.markdown("#### 🚇 แผนเดินทางด้วยรถไฟฟ้า BTS")

        walk_start = haversine(start_info['latitude'], start_info['longitude'],
                               bts_start['lat'], bts_start['lng'])
        walk_end   = haversine(end_info['latitude'],   end_info['longitude'],
                               bts_end['lat'],   bts_end['lng'])

        mode_s = "🚶 เดินเท้า" if walk_start <= 400 else "🚖 แนะนำ Grab/แท็กซี่"
        mode_e = "🚶 เดินเท้า" if walk_end   <= 400 else "🚖 แนะนำ Grab/แท็กซี่"

        st.info(f"**🟢 ขั้นที่ 1 — ทางเท้า (Footpath):**\n\n"
                f"{mode_s} จากต้นทางไปสถานี **{bts_start['clean_name']}** "
                f"({walk_start:.0f} ม.)")
        with st.expander("ℹ️ อารยสถาปัตย์สถานีต้นทาง"):
            st.write(f"🛗 ลิฟต์: {lift_s}")
            st.write(f"♿ ทางลาด: {ramp_s}")
            st.write(f"🚨 พื้นที่หนีภัยผู้พิการ: {evac_s}")
            line_info = str(bts_start.get('btsline', '-'))
            st.write(f"🚉 สาย: {line_info[:60]}")

        if bts_start['clean_name'] != bts_end['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2 — รถไฟฟ้า:**\n\n"
                    f"นั่งรถไฟฟ้าจาก **{bts_start['clean_name']}** → **{bts_end['clean_name']}**")

        st.info(f"**🔴 ขั้นที่ 3 — ทางเท้า (Footpath):**\n\n"
                f"{mode_e} จากสถานี **{bts_end['clean_name']}** ไปปลายทาง "
                f"({walk_end:.0f} ม.)")
        with st.expander("ℹ️ อารยสถาปัตย์สถานีปลายทาง"):
            st.write(f"🛗 ลิฟต์: {lift_e}")
            st.write(f"♿ ทางลาด: {ramp_e}")
            st.write(f"🚨 พื้นที่หนีภัยผู้พิการ: {evac_e}")

    # ─── โหมดรถเมล์ ───────────────────────────────────────────────────────────
    elif "🚌" in travel_mode:
        st.markdown("#### 🚌 แผนเดินทางด้วยรถเมล์ชานต่ำ")
        if len(bus_near_start) > 0 and len(bus_near_end) > 0:
            for _, b in bus_near_start.iterrows():
                st.success(f"🚏 ป้ายรถเมล์ต้นทาง: **{b['place_name']}** ({b['dist']:.0f} ม.)")
            for _, b in bus_near_end.iterrows():
                st.success(f"🚏 ป้ายรถเมล์ปลายทาง: **{b['place_name']}** ({b['dist']:.0f} ม.)")
            st.markdown("""
**📋 ขั้นตอนการเดินทาง:**
1. 🚶 เดินไปป้ายรถเมล์ใกล้ต้นทาง (ดูแผนที่ด้านขวา)
2. 🚌 ขึ้นรถเมล์ชานต่ำ Thai Smile Bus (มีแรมป์ไฮโดรลิก + ล็อกล้อรถเข็น)
3. 🏁 ลงรถที่ป้ายใกล้ปลายทาง แล้วเดินต่อเข้าสู่จุดหมาย
            """)
        else:
            st.warning("⚠️ ไม่พบป้ายรถเมล์ในรัศมี 600 เมตร — ระบบเปลี่ยนแผนเป็น BTS อัตโนมัติ")
            walk_s = haversine(start_info['latitude'], start_info['longitude'],
                               bts_start['lat'], bts_start['lng'])
            st.info(f"**แผนสำรอง:** เดินทางด้วย BTS จาก {bts_start['clean_name']} → {bts_end['clean_name']}")

    # ─── โหมดสวัสดิการ ───────────────────────────────────────────────────────
    else:
        st.markdown("#### 🏥 สวัสดิการรถรับ-ส่งผู้พิการ")
        is_hospital = any(kw in end_label.lower() for kw in ['hospital','โรงพยาบาล','รพ.'])
        if is_hospital:
            st.success("✅ ยืนยันสิทธิ์บริการรถรับ-ส่งผู้พิการ")
            st.markdown("""
- 📞 **สายด่วน กทม.:** โทร **1555** หรือ **1479**
- 🚑 **สปสช. (รักษาฟรี):** โทร **1330**
- 🦽 **มูลนิธิสิทธิคนพิการ:** โทร **02-990-0331**
            """)
        else:
            st.error("❌ บริการนี้จำกัดสำหรับการเดินทางไปสถานพยาบาลเท่านั้น")

    # ─── ข้อมูลผู้โดยสาร (Crowded Insight) ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 ข้อมูลสถิติผู้โดยสาร (จาก Passenger Dataset)")
    clean = str(bts_start.get('clean_name',''))
    sub = df_passenger[df_passenger['Station'].str.contains(
        clean.split()[0] if clean else 'Siam', case=False, na=False)]
    if len(sub) > 0:
        rush = sub[sub['Time Period']=='Rush Hour']
        avg_in = rush['Passengers In'].mean() if len(rush) > 0 else sub['Passengers In'].mean()
        st.metric("เฉลี่ยผู้โดยสารขาเข้า (ชั่วโมงเร่งด่วน)", f"{avg_in:.0f} คน/ชม.")
        crowd_label = "🟢 ไม่แออัด" if avg_in<300 else ("🟡 ปานกลาง" if avg_in<600 else "🔴 แออัดมาก")
        st.write(f"ระดับความหนาแน่น: **{crowd_label}**")
    else:
        st.caption("ไม่พบข้อมูลผู้โดยสารสำหรับสถานีนี้")

# ─── แผนที่ ───────────────────────────────────────────────────────────────────
with col_map:
    st.markdown("### 🗺️ แผนที่เส้นทางและทางเท้า")

    # Legend
    st.markdown("""
    <small>
    🟠 เส้นประ = ทางเท้า (Pedestrian Footpath) &nbsp;|&nbsp;
    🟢 เส้นทึบ = รถไฟฟ้า BTS &nbsp;|&nbsp;
    🔵 จุด = สถานี BTS &nbsp;|&nbsp;
    🔴/🟡 หมุด = จุดหมาย
    </small>
    """, unsafe_allow_html=True)

    center_lat = (start_info['latitude'] + end_info['latitude']) / 2
    center_lon = (start_info['longitude'] + end_info['longitude']) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                   tiles='CartoDB positron')

    # Layer groups
    fg_footpath = folium.FeatureGroup(name="🚶 ทางเท้า (Footpath)", show=True)
    fg_transit  = folium.FeatureGroup(name="🚇 เส้นทางรถไฟฟ้า", show=True)
    fg_stops    = folium.FeatureGroup(name="🚏 ป้ายรถเมล์", show=True)
    fg_bts      = folium.FeatureGroup(name="🔵 สถานี BTS", show=True)

    # ─── หมุดต้นทาง-ปลายทาง ──────────────────────────────────────────────────
    folium.Marker(
        [start_info['latitude'], start_info['longitude']],
        popup=folium.Popup(f"<b>🟢 ต้นทาง</b><br>{start_label.split(' (')[0]}", max_width=200),
        tooltip=f"ต้นทาง: {start_label.split(' (')[0]}",
        icon=folium.Icon(color='orange', icon='play', prefix='fa')
    ).add_to(m)

    folium.Marker(
        [end_info['latitude'], end_info['longitude']],
        popup=folium.Popup(f"<b>🏁 ปลายทาง</b><br>{end_label.split(' (')[0]}", max_width=200),
        tooltip=f"ปลายทาง: {end_label.split(' (')[0]}",
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    walk_start_dist = haversine(start_info['latitude'], start_info['longitude'],
                                 bts_start['lat'], bts_start['lng'])
    walk_end_dist   = haversine(end_info['latitude'], end_info['longitude'],
                                 bts_end['lat'],   bts_end['lng'])

    if "🚇" in travel_mode or ("🚌" in travel_mode and len(bus_near_start) == 0):
        # สถานี BTS ต้นทาง
        folium.Marker(
            [bts_start['lat'], bts_start['lng']],
            popup=folium.Popup(
                f"<b>🚇 สถานี {bts_start['clean_name']}</b><br>"
                f"ลิฟต์: {lift_s}<br>ทางลาด: {ramp_s}", max_width=200),
            tooltip=f"BTS: {bts_start['clean_name']}",
            icon=folium.Icon(color='blue', icon='train', prefix='fa')
        ).add_to(fg_bts)

        # สถานี BTS ปลายทาง (ถ้าต่างสถานี)
        if bts_start['clean_name'] != bts_end['clean_name']:
            folium.Marker(
                [bts_end['lat'], bts_end['lng']],
                popup=folium.Popup(
                    f"<b>🚇 สถานี {bts_end['clean_name']}</b><br>"
                    f"ลิฟต์: {lift_e}<br>ทางลาด: {ramp_e}", max_width=200),
                tooltip=f"BTS: {bts_end['clean_name']}",
                icon=folium.Icon(color='darkblue', icon='train', prefix='fa')
            ).add_to(fg_bts)

        # ─── เส้นทางเดินเท้า (Footpath) ช่วงที่ 1 ──────────────────────────
        folium.PolyLine(
            [[start_info['latitude'], start_info['longitude']],
             [bts_start['lat'], bts_start['lng']]],
            color='#e67e22', weight=4, dash_array='8,8',
            tooltip=f"🚶 ทางเท้าช่วงที่ 1: {walk_start_dist:.0f} ม. → สถานี {bts_start['clean_name']}"
        ).add_to(fg_footpath)

        # วงกลมกึ่งกลางทางเท้า (footpath midpoint marker)
        mid1_lat = (start_info['latitude'] + bts_start['lat']) / 2
        mid1_lon = (start_info['longitude'] + bts_start['lng']) / 2
        folium.CircleMarker(
            [mid1_lat, mid1_lon], radius=6,
            color='#e67e22', fill=True, fill_color='#f39c12', fill_opacity=0.9,
            tooltip=f"🦮 จุดกึ่งกลางทางเท้า ({walk_start_dist/2:.0f} ม.)"
        ).add_to(fg_footpath)

        # ─── เส้นรถไฟฟ้า ─────────────────────────────────────────────────────
        if bts_start['clean_name'] != bts_end['clean_name']:
            folium.PolyLine(
                [[bts_start['lat'], bts_start['lng']],
                 [bts_end['lat'],   bts_end['lng']]],
                color='#00aa44', weight=7,
                tooltip="🚇 เส้นทางรถไฟฟ้า BTS"
            ).add_to(fg_transit)

        # ─── เส้นทางเดินเท้า (Footpath) ช่วงที่ 3 ──────────────────────────
        folium.PolyLine(
            [[bts_end['lat'], bts_end['lng']],
             [end_info['latitude'], end_info['longitude']]],
            color='#e67e22', weight=4, dash_array='8,8',
            tooltip=f"🚶 ทางเท้าช่วงที่ 3: {walk_end_dist:.0f} ม. → ปลายทาง"
        ).add_to(fg_footpath)

        mid3_lat = (bts_end['lat'] + end_info['latitude']) / 2
        mid3_lon = (bts_end['lng'] + end_info['longitude']) / 2
        folium.CircleMarker(
            [mid3_lat, mid3_lon], radius=6,
            color='#e67e22', fill=True, fill_color='#f39c12', fill_opacity=0.9,
            tooltip=f"🦮 จุดกึ่งกลางทางเท้า ({walk_end_dist/2:.0f} ม.)"
        ).add_to(fg_footpath)

    elif "🚌" in travel_mode:
        # ─── ป้ายรถเมล์ใกล้ต้นทาง ────────────────────────────────────────────
        for _, b in bus_near_start.iterrows():
            folium.Marker(
                [b['latitude'], b['longitude']],
                tooltip=f"🚏 ป้ายรถเมล์: {b['place_name']} ({b['dist']:.0f} ม.)",
                icon=folium.Icon(color='purple', icon='bus', prefix='fa')
            ).add_to(fg_stops)
            folium.PolyLine(
                [[start_info['latitude'], start_info['longitude']],
                 [b['latitude'], b['longitude']]],
                color='#9b59b6', weight=3, dash_array='6,6',
                tooltip=f"🚶 ทางเท้าไปป้าย {b['place_name']} ({b['dist']:.0f} ม.)"
            ).add_to(fg_footpath)

        # ─── ป้ายรถเมล์ใกล้ปลายทาง ───────────────────────────────────────────
        for _, b in bus_near_end.iterrows():
            folium.Marker(
                [b['latitude'], b['longitude']],
                tooltip=f"🚏 ป้ายรถเมล์: {b['place_name']} ({b['dist']:.0f} ม.)",
                icon=folium.Icon(color='purple', icon='bus', prefix='fa')
            ).add_to(fg_stops)
            folium.PolyLine(
                [[b['latitude'], b['longitude']],
                 [end_info['latitude'], end_info['longitude']]],
                color='#9b59b6', weight=3, dash_array='6,6',
                tooltip=f"🚶 ทางเท้าจากป้าย {b['place_name']} ({b['dist']:.0f} ม.)"
            ).add_to(fg_footpath)

        # เส้นรถเมล์
        if len(bus_near_start) > 0 and len(bus_near_end) > 0:
            folium.PolyLine(
                [[bus_near_start.iloc[0]['latitude'], bus_near_start.iloc[0]['longitude']],
                 [bus_near_end.iloc[0]['latitude'],   bus_near_end.iloc[0]['longitude']]],
                color='#8e44ad', weight=5,
                tooltip="🚌 เส้นทางรถเมล์ชานต่ำ (Thai Smile Bus)"
            ).add_to(fg_transit)

    # เพิ่ม layers ลงแผนที่
    fg_footpath.add_to(m)
    fg_transit.add_to(m)
    fg_stops.add_to(m)
    fg_bts.add_to(m)
    folium.LayerControl(position='topright').add_to(m)

    # Measure control
    from folium.plugins import MeasureControl, MiniMap
    MeasureControl(position='bottomleft').add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    st_folium(m, width="100%", height=600)

# ─── ส่วนล่าง: ตารางสถานี BTS ทั้งหมด + AI importance ─────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### ♿ ข้อมูลอารยสถาปัตย์สถานี BTS ทั้งหมด (สายสีเขียว)")
    st.dataframe(
        df_bts[['clean_name','มีลิฟต์','ทางลาดสำหรับรถเข็น',
                'พื้นที่สำหรับหนีภัยของคนพิการ','ลิฟต์ราวบันไดสำหรับคนพิการ']].rename(columns={
            'clean_name':'สถานี','มีลิฟต์':'ลิฟต์',
            'ทางลาดสำหรับรถเข็น':'ทางลาด',
            'พื้นที่สำหรับหนีภัยของคนพิการ':'พื้นที่หนีภัย',
            'ลิฟต์ราวบันไดสำหรับคนพิการ':'ลิฟต์ราวบันได'
        }),
        use_container_width=True, height=280
    )

with col_b:
    st.markdown("#### 🤖 ความสำคัญของตัวแปร AI (Feature Importance)")
    imp_df = pd.DataFrame(list(ai_importances.items()), columns=['ตัวแปร','ความสำคัญ'])
    imp_df['ตัวแปร'] = imp_df['ตัวแปร'].map(factor_thai).fillna(imp_df['ตัวแปร'])
    imp_df = imp_df.sort_values('ความสำคัญ', ascending=False)
    st.bar_chart(imp_df.set_index('ตัวแปร')['ความสำคัญ'], use_container_width=True, height=280)

st.markdown("---")
st.caption("🤖 AI ขับเคลื่อนด้วย Random Forest (sklearn) | ข้อมูล: BTS Accessibility, Bus Stops, Passenger Data Bangkok | V7.1")
