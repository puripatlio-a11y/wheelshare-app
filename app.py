"""
AI Accessibility Route Planner for Wheelchair Users — Version 8.0 (MERGED)
รวมคุณสมบัติ:
  V7: Random Forest จาก wheelchair_300rows, ข้อมูล BTS accessibility, bus stops, passenger data
  New: OpenRouteService real road routing, Turn-by-Turn, sidewalk AI sliders, obstacle markers
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from folium.plugins import MeasureControl, MiniMap
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config (ต้องอยู่บรรทัดแรกสุด) ──────────────────────────────────────
st.set_page_config(
    page_title="AI Accessibility Route Planner V8",
    layout="wide",
    page_icon="♿"
)

# ─── Header ──────────────────────────────────────────────────────────────────
header_html = """
<style>
.custom-header {
    background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.60)),
        url("https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?semt=ais_hybrid&w=740&q=80");
    background-size: cover; background-position: center;
    padding: 40px; border-radius: 14px; color: white;
    text-align: center; margin-bottom: 20px;
}
.custom-header h1 { color:#fff !important; font-size:2.4rem !important;
    text-shadow:2px 2px 8px rgba(0,0,0,0.8); margin-bottom:4px; }
.custom-header h3 { color:#f0f2f6 !important; font-size:1.2rem !important;
    font-weight:400; text-shadow:1px 1px 5px rgba(0,0,0,0.7); }
.ai-badge { background:#1a6faf; color:white; padding:4px 14px;
    border-radius:20px; font-size:0.85rem; display:inline-block; margin-top:8px; }
</style>
<div class="custom-header">
  <h1>♿ AI Accessibility Route Planner for Wheelchair Users</h1>
  <h3>ระบบวางแผนเส้นทางอัจฉริยะสำหรับผู้ใช้รถเข็น ด้วยปัญญาประดิษฐ์ + Real Road Routing</h3>
  <span class="ai-badge">🤖 Powered by Random Forest AI + OpenRouteService</span>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("---")

# ─── ORS Client (Real Road Routing) ──────────────────────────────────────────
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjM4ZTc2MTM5NzUyZjQ1ZjJhOTJjYTlhOTEwMzg5NmZlIiwiaCI6Im11cm11cjY0In0="
ors_available = False
try:
    import openrouteservice
    if ORS_API_KEY != "PUT_YOUR_OPENROUTESERVICE_API_KEY_HERE" and len(ORS_API_KEY) > 10:
        ors_client = openrouteservice.Client(key=ORS_API_KEY)
        ors_available = True
except ImportError:
    pass

# ─── ฟังก์ชัน Haversine ──────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางระหว่างสองพิกัด (เมตร) ด้วยสูตร Haversine"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

# ─── โหลดข้อมูลทุกไฟล์ CSV ───────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    base = "/mnt/user-data/uploads"

    df_places   = pd.read_csv(f"{base}/bangkok_places_bus_spot.csv")
    df_stations = pd.read_csv(f"{base}/bts_station.csv")
    df_acc      = pd.read_csv(f"{base}/BTS_for_wheelchair_users_spreadsheet_-_BTS_green_line.csv")
    df_bus_stops= pd.read_csv(f"{base}/bangkok_bus_stops_coordinates.csv")
    df_passenger= pd.read_csv(f"{base}/bangkok_transit_passenger_data__1_.csv")
    df_rf       = pd.read_csv(f"{base}/wheelchair_random_forest_300rows.csv")

    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี','').str.strip()
    df_acc['clean_name']      = df_acc['สถานี'].str.replace('สถานี','').str.strip()

    df_bts = pd.merge(
        df_acc,
        df_stations[['clean_name','lat','lng','btsline','location']],
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)

    return df_places, df_bts, df_bus_stops, df_passenger, df_rf

df_places, df_bts, df_bus_stops, df_passenger, df_rf = load_all_data()

# ─── 🤖 AI MODEL 1: Random Forest จาก wheelchair_random_forest_300rows ────────
@st.cache_resource
def train_route_rf(df_rf):
    """
    AI Function 1 — Route Recommendation RF
    Input features: Elevator, Ramp, Accessible_Exit, Cost, Travel_Time,
                    BusSupport, Safety, Crowded_Level, Urgency,
                    Prefer_Safe, Prefer_Cheap, Transport_Type (encoded)
    Output: Recommended (1=แนะนำ, 0=ไม่แนะนำ)
    """
    le = LabelEncoder()
    df = df_rf.copy()
    df['Transport_Type_enc'] = le.fit_transform(df['Transport_Type'])
    features = ['Elevator','Ramp','Accessible_Exit','Cost','Travel_Time',
                'BusSupport','Safety','Crowded_Level','Urgency',
                'Prefer_Safe','Prefer_Cheap','Transport_Type_enc']
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    clf.fit(df[features], df['Recommended'])
    return clf, le, features

route_rf, le_transport, rf_features = train_route_rf(df_rf)

# ─── 🤖 AI MODEL 2: Sidewalk Accessibility RF (จาก New Code) ──────────────────
@st.cache_resource
def train_sidewalk_rf():
    """
    AI Function 2 — Sidewalk/Footpath Accessibility Classifier
    Input: sidewalk_width, surface (0/1), obstacles, ramp (0/1)
    Output: ปลอดภัย (1) หรือ ไม่ปลอดภัย (0) สำหรับรถเข็น
    ใช้ข้อมูลจำลองเพื่อเทรน (ปรับได้ด้วยข้อมูลจริงในภายหลัง)
    """
    X = np.array([
        [1.5,1,0,1],[0.7,0,3,0],[1.2,1,1,1],[1.8,1,0,1],[0.8,0,2,0],
        [2.0,1,0,1],[0.6,0,5,0],[1.4,1,1,1],[1.0,0,2,0],[2.5,1,0,1],
        [0.9,0,4,0],[1.6,1,1,1],[1.1,1,0,1],[0.5,0,6,0],[2.0,1,2,1],
    ])
    y = np.array([1,0,1,1,0, 1,0,1,0,1, 0,1,1,0,1])
    mdl = RandomForestClassifier(n_estimators=50, random_state=42)
    mdl.fit(X, y)
    return mdl

sidewalk_rf = train_sidewalk_rf()

def ai_route_predict(transport_type, elevator, ramp, accessible_exit,
                     cost, travel_time, bus_support, safety,
                     crowded_level, urgency, prefer_safe, prefer_cheap):
    """🤖 AI Function 1: ทำนายความเหมาะสมของเส้นทางด้วย Random Forest"""
    try:    t_enc = le_transport.transform([transport_type])[0]
    except: t_enc = 0
    row = pd.DataFrame([[elevator,ramp,accessible_exit,cost,travel_time,
                         bus_support,safety,crowded_level,urgency,
                         prefer_safe,prefer_cheap,t_enc]], columns=rf_features)
    prob  = route_rf.predict_proba(row)[0][1]
    label = int(route_rf.predict(row)[0])
    imp   = dict(zip(rf_features, route_rf.feature_importances_))
    return label, prob, imp

def ai_sidewalk_predict(width, surface_val, obstacles, ramp_val):
    """🤖 AI Function 2: ประเมินความปลอดภัยของทางเท้าจากสภาพแวดล้อม"""
    feats = np.array([[width, surface_val, obstacles, ramp_val]])
    prob  = sidewalk_rf.predict_proba(feats)[0][1]
    label = int(sidewalk_rf.predict(feats)[0])
    return label, prob

# ─── Helper Functions ─────────────────────────────────────────────────────────
def nearest_bts(lat, lon):
    df_bts['_d'] = df_bts.apply(lambda r: haversine(lat,lon,r['lat'],r['lng']), axis=1)
    return df_bts.sort_values('_d').iloc[0]

def nearest_bus_stops(lat, lon, n=3, max_dist=600):
    df_bus_stops['_d'] = df_bus_stops.apply(
        lambda r: haversine(lat,lon,r['latitude'],r['longitude']), axis=1)
    return df_bus_stops[df_bus_stops['_d']<=max_dist].sort_values('_d').head(n)

def get_accessibility(row):
    lift  = "✅ มี" if str(row.get('มีลิฟต์','')).strip()              in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    ramp  = "✅ มี" if str(row.get('ทางลาดสำหรับรถเข็น','')).strip()   in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    evac  = "✅ มี" if str(row.get('พื้นที่สำหรับหนีภัยของคนพิการ','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    return lift, ramp, evac

def get_crowd(station_clean):
    word  = station_clean.split()[0] if station_clean else ''
    sub   = df_passenger[df_passenger['Station'].str.contains(word, case=False, na=False)]
    if len(sub) == 0: return 3
    avg   = sub[sub['Time Period']=='Rush Hour']['Passengers In'].mean()
    if   avg < 300: return 1
    elif avg < 600: return 2
    elif avg < 900: return 3
    return 4

def get_ors_route(start_lon, start_lat, end_lon, end_lat):
    """ดึงเส้นทางเดินเท้าจริงจาก OpenRouteService API"""
    try:
        route = ors_client.directions(
            coordinates=[(start_lon,start_lat),(end_lon,end_lat)],
            profile='foot-walking', format='geojson'
        )
        coords   = [[p[1],p[0]] for p in route['features'][0]['geometry']['coordinates']]
        summary  = route['features'][0]['properties']['summary']
        steps    = route['features'][0]['properties']['segments'][0]['steps']
        dist_km  = summary['distance'] / 1000
        dur_min  = summary['duration'] / 60
        return coords, dist_km, dur_min, steps
    except Exception as e:
        return None, None, None, None

# ─── พจนานุกรมชื่อไทย ─────────────────────────────────────────────────────────
th_name_map = {
    "Victory Monument":"อนุสาวรีย์ชัยสมรภูมิ","Siam Station":"สถานีรถไฟฟ้า สยาม",
    "CentralWorld":"เซ็นทรัลเวิลด์","MBK Center":"เอ็มบีเค เซ็นเตอร์",
    "Samyan Mitrtown":"สามย่านมิตรทาวน์","Chulalongkorn Hospital":"โรงพยาบาลจุฬาลงกรณ์",
    "Siriraj Hospital":"โรงพยาบาลศิริราช","Ramathibodi Hospital":"โรงพยาบาลรามาธิบดี",
    "Rajavithi Hospital":"โรงพยาบาลราชวิถี","Vajira Hospital":"โรงพยาบาลวชิรพยาบาล",
    "Mochit Bus Terminal":"สถานีขนส่งหมอชิต 2","Chatuchak Park":"สวนจตุจักร",
    "Ari BTS Station":"สถานีอารีย์","Saphan Khwai BTS Station":"สถานีสะพานควาย",
    "Bang Wa BTS Station":"สถานีบางหว้า","Bearing BTS Station":"สถานีแบริ่ง",
    "Ekkamai Bus Terminal":"สถานีขนส่งเอกมัย","Kasetsart University":"มหาวิทยาลัยเกษตรศาสตร์",
}

factor_thai = {
    'Elevator':'ลิฟต์','Ramp':'ทางลาด','Accessible_Exit':'ทางออกอารยสถาปัตย์',
    'Cost':'ค่าใช้จ่าย','Travel_Time':'เวลาเดินทาง','Safety':'ความปลอดภัย',
    'Crowded_Level':'ความหนาแน่น','BusSupport':'รถเมล์ชานต่ำ',
    'Urgency':'ความเร่งด่วน','Prefer_Safe':'ความปลอดภัยสำคัญ',
    'Prefer_Cheap':'ราคาประหยัดสำคัญ','Transport_Type_enc':'ประเภทขนส่ง'
}

def make_display_name(row):
    th   = th_name_map.get(row['place_name'], row['place_name'])
    bts  = nearest_bts(row['latitude'], row['longitude'])
    suf  = []
    if 'bts' in row['place_name'].lower() or bts['_d'] <= 400: suf.append("BTS")
    if len(nearest_bus_stops(row['latitude'], row['longitude'], n=1, max_dist=300)) > 0:
        suf.append("มีป้ายรถเมล์")
    return f"{th} ({' / '.join(suf)})" if suf else th

df_places['display_th'] = df_places.apply(make_display_name, axis=1)
place_list = sorted(df_places['display_th'].tolist())

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.header("🕹️ ตั้งค่าการเดินทาง")

def_start = next((i for i,n in enumerate(place_list) if "อนุสาวรีย์" in n), 0)
def_end   = next((i for i,n in enumerate(place_list) if "จุฬา" in n), 1)

start_label = st.sidebar.selectbox("📍 จุดต้นทาง:", place_list, index=def_start)
end_label   = st.sidebar.selectbox("🏁 จุดปลายทาง:", place_list, index=def_end)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ข้อมูลส่วนตัว (สำหรับ AI)")
prefer_safe  = st.sidebar.checkbox("🛡️ ให้ความสำคัญด้านความปลอดภัย", value=True)
prefer_cheap = st.sidebar.checkbox("💰 ให้ความสำคัญด้านประหยัด", value=False)
urgency_opt  = st.sidebar.selectbox("⏱️ ความเร่งด่วน:", ["ไม่เร่งด่วน (0)","ปานกลาง (1)"])
urgency_val  = 0 if "ไม่" in urgency_opt else 1

st.sidebar.markdown("---")
travel_mode = st.sidebar.radio("🚦 โหมดการเดินทาง:", [
    "🚇 รถไฟฟ้า BTS","🚌 รถเมล์ชานต่ำ","🏥 สวัสดิการรถตู้รัฐ (โรงพยาบาล)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🦮 ประเมินสภาพทางเท้า (AI Function 2)")
sidewalk_width = st.sidebar.slider("ความกว้างทางเท้า (เมตร)", 0.5, 3.0, 1.5, step=0.1)
surface_opt    = st.sidebar.selectbox("สภาพพื้นผิว:", ["เรียบ (ดี)","ขรุขระ (แย่)"])
obstacles      = st.sidebar.slider("จำนวนสิ่งกีดขวางโดยประมาณ", 0, 10, 2)
has_ramp_sw    = st.sidebar.checkbox("มีทางลาดบริเวณทางเท้า", value=True)

# ─── ดึงข้อมูลสถานที่ที่เลือก ────────────────────────────────────────────────
start_info = df_places[df_places['display_th']==start_label].iloc[0]
end_info   = df_places[df_places['display_th']==end_label].iloc[0]

start_lat, start_lon = start_info['latitude'], start_info['longitude']
end_lat,   end_lon   = end_info['latitude'],   end_info['longitude']

bts_s = nearest_bts(start_lat, start_lon)
bts_e = nearest_bts(end_lat, end_lon)

lift_s, ramp_s, evac_s = get_accessibility(bts_s)
lift_e, ramp_e, evac_e = get_accessibility(bts_e)

bus_near_s = nearest_bus_stops(start_lat, start_lon, n=3)
bus_near_e = nearest_bus_stops(end_lat, end_lon, n=3)

crowd_s = get_crowd(str(bts_s.get('clean_name','')))
crowd_e = get_crowd(str(bts_e.get('clean_name','')))
avg_crowd = int((crowd_s + crowd_e) / 2)

dist_straight = haversine(start_lat, start_lon, end_lat, end_lon)
est_cost = int(16 + dist_straight/1000 * 2.5)
est_time = int(dist_straight/1000 * 6)

# ─── 🤖 AI Prediction 1: Route RF ────────────────────────────────────────────
transport_map = {
    "🚇 รถไฟฟ้า BTS":"BTS",
    "🚌 รถเมล์ชานต่ำ":"Bus",
    "🏥 สวัสดิการรถตู้รัฐ (โรงพยาบาล)":"BTS+Bus"
}
transport_str = transport_map[travel_mode]
el_val  = 1 if ("✅" in lift_s and "✅" in lift_e) else 0
rmp_val = 1 if ("✅" in ramp_s and "✅" in ramp_e) else 0
acc_exit= 1 if (el_val or rmp_val) else 0

ai_label, ai_prob, ai_imp = ai_route_predict(
    transport_str, el_val, rmp_val, acc_exit, est_cost, est_time,
    1 if "🚌" in travel_mode else 0,
    4 if prefer_safe else 3, avg_crowd, urgency_val,
    1 if prefer_safe else 0, 1 if prefer_cheap else 0
)

# ─── 🤖 AI Prediction 2: Sidewalk RF ─────────────────────────────────────────
surface_val  = 1 if "เรียบ" in surface_opt else 0
ramp_sw_val  = 1 if has_ramp_sw else 0
sw_label, sw_prob = ai_sidewalk_predict(sidewalk_width, surface_val, obstacles, ramp_sw_val)

# ─── ORS Real Road Route ──────────────────────────────────────────────────────
route_coords, route_dist_km, route_dur_min, route_steps = None, None, None, None
if ors_available:
    with st.spinner("📡 กำลังดึงเส้นทางจริงจาก OpenRouteService..."):
        route_coords, route_dist_km, route_dur_min, route_steps = \
            get_ors_route(start_lon, start_lat, end_lon, end_lat)

route_color = "green" if (ai_label==1 and sw_label==1) else ("orange" if ai_label==1 else "red")

# ─── LAYOUT: 3 คอลัมน์ ────────────────────────────────────────────────────────
col_ai, col_route, col_map = st.columns([1, 1, 2])

# ══════════════════════════════════════════════════════════════════════════════
# COL 1 — AI Analysis
# ══════════════════════════════════════════════════════════════════════════════
with col_ai:
    st.markdown("### 🤖 AI Analysis")
    st.write(f"**จาก:** {start_label.split(' (')[0]}")
    st.write(f"**ถึง:** {end_label.split(' (')[0]}")
    st.write(f"📏 ระยะตรง: **{dist_straight/1000:.2f} กม.**")

    st.markdown("---")
    st.markdown("#### AI Function 1: Route RF")
    st.caption("วิเคราะห์ความเหมาะสมเส้นทางจาก 300 rows training data")
    if ai_label == 1:
        st.success(f"✅ AI แนะนำเส้นทางนี้\nความเชื่อมั่น **{ai_prob*100:.1f}%**")
    else:
        st.error(f"⚠️ AI ไม่แนะนำ\nความเชื่อมั่นต่ำ ({ai_prob*100:.1f}%)")

    top3 = sorted(ai_imp.items(), key=lambda x: x[1], reverse=True)[:3]
    st.caption("🔍 ปัจจัยสำคัญ: " + ", ".join(
        [f"{factor_thai.get(k,k)} ({v*100:.0f}%)" for k,v in top3]))
    st.write(f"💸 ค่าโดยสารประมาณ: **{est_cost} บาท**")
    st.write(f"⏱️ เวลาประมาณ: **{est_time} นาที**")

    st.markdown("---")
    st.markdown("#### AI Function 2: Sidewalk RF")
    st.caption("ประเมินความปลอดภัยของทางเท้า (Footpath)")
    if sw_label == 1:
        st.success(f"🟢 ทางเท้าปลอดภัยสำหรับรถเข็น\nคะแนน: **{sw_prob*100:.1f}%**")
    else:
        st.warning(f"🔴 ทางเท้ามีความเสี่ยง\n(คะแนน {sw_prob*100:.1f}%) — แนะนำ GrabAssist")
    st.caption(f"กว้าง {sidewalk_width}ม. | {'เรียบ' if surface_val else 'ขรุขระ'} | "
               f"สิ่งกีดขวาง {obstacles} จุด | ทางลาด {'มี' if ramp_sw_val else 'ไม่มี'}")

    if route_dist_km:
        st.markdown("---")
        st.markdown("#### 🛣️ เส้นทางจริง (ORS)")
        st.write(f"📏 ระยะทางจริง: **{route_dist_km:.2f} กม.**")
        st.write(f"⏱️ เวลาเดิน: **{route_dur_min:.0f} นาที**")

# ══════════════════════════════════════════════════════════════════════════════
# COL 2 — Route Plan + Turn-by-Turn
# ══════════════════════════════════════════════════════════════════════════════
with col_route:
    st.markdown("### 📋 แผนการเดินทาง")

    # ─── BTS Mode ─────────────────────────────────────────────────────────────
    if "🚇" in travel_mode:
        walk_s = haversine(start_lat, start_lon, bts_s['lat'], bts_s['lng'])
        walk_e = haversine(end_lat, end_lon, bts_e['lat'], bts_e['lng'])
        mode_s = "🚶 เดินเท้า" if walk_s<=400 else "🚖 แนะนำ Grab/แท็กซี่"
        mode_e = "🚶 เดินเท้า" if walk_e<=400 else "🚖 แนะนำ Grab/แท็กซี่"

        st.info(f"**🟢 ขั้นที่ 1 — ทางเท้า (Footpath):**\n\n"
                f"{mode_s} ไปสถานี **{bts_s['clean_name']}** ({walk_s:.0f} ม.)")
        with st.expander("ℹ️ อารยสถาปัตย์สถานีต้นทาง"):
            st.write(f"🛗 ลิฟต์: {lift_s} | ♿ ทางลาด: {ramp_s} | 🚨 พื้นที่หนีภัย: {evac_s}")

        if bts_s['clean_name'] != bts_e['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2 — รถไฟฟ้า:**\n\n"
                    f"นั่งรถไฟฟ้า **{bts_s['clean_name']}** → **{bts_e['clean_name']}**")

        st.info(f"**🔴 ขั้นที่ 3 — ทางเท้า (Footpath):**\n\n"
                f"{mode_e} จากสถานี **{bts_e['clean_name']}** ไปปลายทาง ({walk_e:.0f} ม.)")
        with st.expander("ℹ️ อารยสถาปัตย์สถานีปลายทาง"):
            st.write(f"🛗 ลิฟต์: {lift_e} | ♿ ทางลาด: {ramp_e} | 🚨 พื้นที่หนีภัย: {evac_e}")

    # ─── Bus Mode ─────────────────────────────────────────────────────────────
    elif "🚌" in travel_mode:
        st.markdown("#### 🚌 แผนรถเมล์ชานต่ำ")
        if len(bus_near_s) > 0:
            for _, b in bus_near_s.iterrows():
                st.success(f"🚏 ป้ายต้นทาง: **{b['place_name']}** ({b['_d']:.0f} ม.)")
        else:
            st.warning("ไม่พบป้ายรถเมล์ในรัศมี 600 ม. — แนะนำใช้ BTS แทน")
        if len(bus_near_e) > 0:
            for _, b in bus_near_e.iterrows():
                st.success(f"🚏 ป้ายปลายทาง: **{b['place_name']}** ({b['_d']:.0f} ม.)")
        st.markdown("""
**📋 ขั้นตอน:**
1. 🚶 เดินไปป้ายรถเมล์ (ดูแผนที่)
2. 🚌 ขึ้น Thai Smile Bus (มีแรมป์ไฮโดรลิก + ล็อกล้อ)
3. 🏁 ลงรถที่ป้ายใกล้ปลายทาง
        """)

    # ─── Hospital Mode ────────────────────────────────────────────────────────
    else:
        st.markdown("#### 🏥 สวัสดิการรถรับ-ส่งผู้พิการ")
        is_hosp = any(kw in end_label.lower() for kw in ['hospital','โรงพยาบาล'])
        if is_hosp:
            st.success("✅ ยืนยันสิทธิ์บริการรถรับ-ส่งผู้พิการ")
            st.markdown("📞 **สายด่วน กทม.:** โทร **1555** หรือ **1479**\n\n"
                        "🚑 **สปสช.:** โทร **1330**")
        else:
            st.error("❌ บริการจำกัดเฉพาะการเดินทางไปสถานพยาบาลเท่านั้น")

    # ─── Turn-by-Turn (จาก ORS) ───────────────────────────────────────────────
    if route_steps:
        st.markdown("---")
        st.markdown("#### 🧭 Turn-by-Turn Navigation (เส้นทางจริง)")
        for i, step in enumerate(route_steps[:10]):
            st.write(f"{i+1}. {step['instruction']} ({step['distance']:.0f} ม.)")
        if len(route_steps) > 10:
            st.caption(f"(และอีก {len(route_steps)-10} ขั้นตอน...)")
    elif not ors_available:
        st.markdown("---")
        st.caption("💡 ใส่ ORS API Key ใน `ORS_API_KEY` เพื่อเปิดใช้เส้นทางจริง + Turn-by-Turn")

    # ─── Passenger Crowd Data ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 ความหนาแน่นผู้โดยสาร")
    clean = str(bts_s.get('clean_name',''))
    sub = df_passenger[df_passenger['Station'].str.contains(
        clean.split()[0] if clean else 'Siam', case=False, na=False)]
    if len(sub) > 0:
        avg_in = sub[sub['Time Period']=='Rush Hour']['Passengers In'].mean()
        st.metric("เฉลี่ยผู้โดยสาร (Rush Hour)", f"{avg_in:.0f} คน/ชม.")
        crowd_label = ("🟢 ไม่แออัด" if avg_in<300 else
                       "🟡 ปานกลาง" if avg_in<600 else "🔴 แออัดมาก")
        st.write(f"ระดับ: **{crowd_label}**")
    else:
        st.caption("ไม่พบข้อมูลผู้โดยสารสำหรับสถานีนี้")

# ══════════════════════════════════════════════════════════════════════════════
# COL 3 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with col_map:
    st.markdown("### 🗺️ แผนที่เส้นทางและทางเท้า")
    st.markdown("""<small>
    🟠 เส้นประ = ทางเท้า (Footpath) &nbsp;|&nbsp;
    🟢 เส้นทึบ = รถไฟฟ้า BTS &nbsp;|&nbsp;
    🔵 หมุด = สถานี BTS &nbsp;|&nbsp;
    🟣 หมุด = ป้ายรถเมล์ &nbsp;|&nbsp;
    ⚠️ หมุดส้ม = จุดอุปสรรค
    </small>""", unsafe_allow_html=True)

    m = folium.Map(
        location=[(start_lat+end_lat)/2, (start_lon+end_lon)/2],
        zoom_start=14, tiles='CartoDB positron'
    )

    # Feature Groups (Layer Control)
    fg_foot    = folium.FeatureGroup(name="🚶 ทางเท้า (Footpath)", show=True)
    fg_transit = folium.FeatureGroup(name="🚇 เส้นทางรถไฟฟ้า / รถเมล์", show=True)
    fg_bts_mk  = folium.FeatureGroup(name="🔵 สถานี BTS", show=True)
    fg_bus_mk  = folium.FeatureGroup(name="🟣 ป้ายรถเมล์", show=True)
    fg_obs     = folium.FeatureGroup(name="⚠️ จุดอุปสรรค (Simulated)", show=True)

    # หมุดต้นทาง-ปลายทาง
    folium.Marker([start_lat,start_lon],
        popup=folium.Popup(f"<b>🟢 ต้นทาง</b><br>{start_label.split(' (')[0]}", max_width=200),
        tooltip=f"ต้นทาง: {start_label.split(' (')[0]}",
        icon=folium.Icon(color='orange', icon='play', prefix='fa')
    ).add_to(m)
    folium.Marker([end_lat,end_lon],
        popup=folium.Popup(f"<b>🏁 ปลายทาง</b><br>{end_label.split(' (')[0]}", max_width=200),
        tooltip=f"ปลายทาง: {end_label.split(' (')[0]}",
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    # ─── BTS / สำรอง ──────────────────────────────────────────────────────────
    if "🚇" in travel_mode or ("🚌" in travel_mode and len(bus_near_s)==0):
        walk_s = haversine(start_lat, start_lon, bts_s['lat'], bts_s['lng'])
        walk_e = haversine(end_lat, end_lon, bts_e['lat'], bts_e['lng'])

        # สถานี BTS markers
        folium.Marker([bts_s['lat'],bts_s['lng']],
            popup=folium.Popup(
                f"<b>🚇 {bts_s['clean_name']}</b><br>ลิฟต์:{lift_s}<br>ทางลาด:{ramp_s}",
                max_width=200),
            tooltip=f"BTS: {bts_s['clean_name']}",
            icon=folium.Icon(color='blue', icon='train', prefix='fa')
        ).add_to(fg_bts_mk)
        folium.Marker([bts_e['lat'],bts_e['lng']],
            popup=folium.Popup(
                f"<b>🚇 {bts_e['clean_name']}</b><br>ลิฟต์:{lift_e}<br>ทางลาด:{ramp_e}",
                max_width=200),
            tooltip=f"BTS: {bts_e['clean_name']}",
            icon=folium.Icon(color='darkblue', icon='train', prefix='fa')
        ).add_to(fg_bts_mk)

        # ทางเท้าช่วงที่ 1 (Footpath)
        folium.PolyLine([[start_lat,start_lon],[bts_s['lat'],bts_s['lng']]],
            color='#e67e22', weight=4, dash_array='8,8',
            tooltip=f"🚶 ทางเท้า ช่วงที่ 1: {walk_s:.0f} ม."
        ).add_to(fg_foot)
        mid1 = ((start_lat+bts_s['lat'])/2, (start_lon+bts_s['lng'])/2)
        folium.CircleMarker(mid1, radius=7, color='#e67e22',
            fill=True, fill_color='#f39c12', fill_opacity=0.9,
            tooltip=f"🦮 จุดกึ่งกลางทางเท้า ช่วงที่ 1"
        ).add_to(fg_foot)

        # เส้นรถไฟฟ้า BTS
        if bts_s['clean_name'] != bts_e['clean_name']:
            folium.PolyLine([[bts_s['lat'],bts_s['lng']],[bts_e['lat'],bts_e['lng']]],
                color='#00aa44', weight=7,
                tooltip="🚇 เส้นทางรถไฟฟ้า BTS"
            ).add_to(fg_transit)

        # ทางเท้าช่วงที่ 3 (Footpath)
        folium.PolyLine([[bts_e['lat'],bts_e['lng']],[end_lat,end_lon]],
            color='#e67e22', weight=4, dash_array='8,8',
            tooltip=f"🚶 ทางเท้า ช่วงที่ 3: {walk_e:.0f} ม."
        ).add_to(fg_foot)
        mid3 = ((bts_e['lat']+end_lat)/2, (bts_e['lng']+end_lon)/2)
        folium.CircleMarker(mid3, radius=7, color='#e67e22',
            fill=True, fill_color='#f39c12', fill_opacity=0.9,
            tooltip="🦮 จุดกึ่งกลางทางเท้า ช่วงที่ 3"
        ).add_to(fg_foot)

    # ─── รถเมล์ ────────────────────────────────────────────────────────────────
    elif "🚌" in travel_mode:
        for _, b in bus_near_s.iterrows():
            folium.Marker([b['latitude'],b['longitude']],
                tooltip=f"🚏 {b['place_name']} ({b['_d']:.0f} ม.)",
                icon=folium.Icon(color='purple', icon='bus', prefix='fa')
            ).add_to(fg_bus_mk)
            folium.PolyLine([[start_lat,start_lon],[b['latitude'],b['longitude']]],
                color='#9b59b6', weight=3, dash_array='6,6',
                tooltip=f"🚶 ทางเท้าไปป้าย ({b['_d']:.0f} ม.)"
            ).add_to(fg_foot)
        for _, b in bus_near_e.iterrows():
            folium.Marker([b['latitude'],b['longitude']],
                tooltip=f"🚏 {b['place_name']} ({b['_d']:.0f} ม.)",
                icon=folium.Icon(color='purple', icon='bus', prefix='fa')
            ).add_to(fg_bus_mk)
            folium.PolyLine([[b['latitude'],b['longitude']],[end_lat,end_lon]],
                color='#9b59b6', weight=3, dash_array='6,6',
                tooltip=f"🚶 ทางเท้าจากป้าย ({b['_d']:.0f} ม.)"
            ).add_to(fg_foot)
        if len(bus_near_s)>0 and len(bus_near_e)>0:
            folium.PolyLine([
                [bus_near_s.iloc[0]['latitude'], bus_near_s.iloc[0]['longitude']],
                [bus_near_e.iloc[0]['latitude'], bus_near_e.iloc[0]['longitude']]],
                color='#8e44ad', weight=5,
                tooltip="🚌 เส้นทางรถเมล์ชานต่ำ"
            ).add_to(fg_transit)

    # ─── ORS Real Road Route (ถ้ามี API Key) ─────────────────────────────────
    if route_coords:
        folium.PolyLine(route_coords, color=route_color, weight=6, opacity=0.85,
            tooltip=f"🛣️ เส้นทางจริง (ORS) — AI Score {ai_prob*100:.1f}%"
        ).add_to(fg_transit)

    # ─── Simulated Obstacle Markers (จาก New Code) ───────────────────────────
    # สร้างจุดอุปสรรคอัตโนมัติตาม obstacles slider (กระจายตามเส้นทาง)
    if obstacles > 0:
        for i in range(min(obstacles, 5)):
            frac = (i + 1) / (obstacles + 1)
            obs_lat = start_lat + frac * (end_lat - start_lat)
            obs_lon = start_lon + frac * (end_lon - start_lon)
            # สุ่มเล็กน้อยเพื่อไม่ให้ซ้ำกัน
            obs_lat += np.random.uniform(-0.001, 0.001)
            obs_lon += np.random.uniform(-0.001, 0.001)
            folium.Marker([obs_lat, obs_lon],
                popup=folium.Popup(f"<b>⚠️ สิ่งกีดขวาง #{i+1}</b><br>AI ตรวจพบบนเส้นทางเดิน", max_width=180),
                tooltip=f"⚠️ Obstacle #{i+1} (Simulated by AI)",
                icon=folium.Icon(color='orange', icon='warning-sign', prefix='glyphicon')
            ).add_to(fg_obs)

    # เพิ่ม layers และ controls
    fg_foot.add_to(m)
    fg_transit.add_to(m)
    fg_bts_mk.add_to(m)
    fg_bus_mk.add_to(m)
    fg_obs.add_to(m)
    folium.LayerControl(position='topright').add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    st_folium(m, width="100%", height=620)

# ─── ส่วนล่าง: ตารางข้อมูล + Feature Importance ─────────────────────────────
st.markdown("---")
col_tbl, col_chart = st.columns(2)

with col_tbl:
    st.markdown("#### ♿ ข้อมูลอารยสถาปัตย์สถานี BTS (สายสีเขียว)")
    st.dataframe(
        df_bts[['clean_name','มีลิฟต์','ทางลาดสำหรับรถเข็น',
                'พื้นที่สำหรับหนีภัยของคนพิการ','ลิฟต์ราวบันไดสำหรับคนพิการ']].rename(columns={
            'clean_name':'สถานี','มีลิฟต์':'ลิฟต์',
            'ทางลาดสำหรับรถเข็น':'ทางลาด',
            'พื้นที่สำหรับหนีภัยของคนพิการ':'พื้นที่หนีภัย',
            'ลิฟต์ราวบันไดสำหรับคนพิการ':'ลิฟต์ราวบันได'}),
        use_container_width=True, height=260
    )

with col_chart:
    st.markdown("#### 🤖 ความสำคัญของตัวแปร (AI Feature Importance — Route RF)")
    imp_df = pd.DataFrame(list(ai_imp.items()), columns=['ตัวแปร','ความสำคัญ'])
    imp_df['ตัวแปร'] = imp_df['ตัวแปร'].map(factor_thai).fillna(imp_df['ตัวแปร'])
    imp_df = imp_df.sort_values('ความสำคัญ', ascending=False)
    st.bar_chart(imp_df.set_index('ตัวแปร')['ความสำคัญ'], use_container_width=True, height=260)

st.markdown("---")
st.caption(
    "🤖 AI: Random Forest (Route RF + Sidewalk RF) | "
    "🗺️ Routing: OpenRouteService (foot-walking) | "
    "📊 Data: BTS Accessibility CSV, Bus Stops CSV, Passenger Data CSV, RF Training CSV | "
    "Version 8.0 Merged"
)
