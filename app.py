"""
AI Accessibility Route Planner — Version 8.1 (Performance Optimized)
แก้ปัญหาช้า: ใช้ vectorized numpy แทน row-by-row haversine, precompute ทุกอย่างตั้งแต่ต้น
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

st.set_page_config(page_title="AI Accessibility Route Planner V8", layout="wide", page_icon="♿")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.custom-header {
    background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.60)),
        url("https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?semt=ais_hybrid&w=740&q=80");
    background-size:cover; background-position:center;
    padding:40px; border-radius:14px; color:white; text-align:center; margin-bottom:20px;
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
""", unsafe_allow_html=True)
st.write("---")

# ─── ORS ──────────────────────────────────────────────────────────────────────
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjM4ZTc2MTM5NzUyZjQ1ZjJhOTJjYTlhOTEwMzg5NmZlIiwiaCI6Im11cm11cjY0In0="
ors_available = False
try:
    import openrouteservice
    if len(ORS_API_KEY) > 20:
        ors_client = openrouteservice.Client(key=ORS_API_KEY)
        ors_available = True
except ImportError:
    pass

# ─── FAST vectorized haversine (numpy arrays) ─────────────────────────────────
def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    """คำนวณระยะทาง (เมตร) จากจุด 1 จุด ไปยัง array ของจุดต่างๆ พร้อมกันทีเดียว"""
    R = 6371000.0
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2 = np.radians(lat2_arr)
    lo2 = np.radians(lon2_arr)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def haversine_scalar(lat1, lon1, lat2, lon2):
    return float(haversine_vec(lat1, lon1, np.array([lat2]), np.array([lon2]))[0])

# ─── โหลด + Precompute ทุกอย่างครั้งเดียว ────────────────────────────────────
@st.cache_data
def load_and_precompute():
    base = "/mnt/user-data/uploads"
    df_places    = pd.read_csv(f"{base}/bangkok_places_bus_spot.csv")
    df_stations  = pd.read_csv(f"{base}/bts_station.csv")
    df_acc       = pd.read_csv(f"{base}/BTS_for_wheelchair_users_spreadsheet_-_BTS_green_line.csv")
    df_bus_stops = pd.read_csv(f"{base}/bangkok_bus_stops_coordinates.csv")
    df_rf        = pd.read_csv(f"{base}/wheelchair_random_forest_300rows.csv")

    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี','').str.strip()
    df_acc['clean_name']      = df_acc['สถานี'].str.replace('สถานี','').str.strip()

    df_bts = pd.merge(
        df_acc,
        df_stations[['clean_name','lat','lng','btsline','location']],
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)

    # Precompute arrays สำหรับ vectorized nearest lookup
    bts_lats = df_bts['lat'].values
    bts_lons = df_bts['lng'].values
    bus_lats  = df_bus_stops['latitude'].values
    bus_lons  = df_bus_stops['longitude'].values

    # Precompute nearest BTS + nearest bus stop สำหรับทุก place ล่วงหน้า
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

    display_names, nearest_bts_idx, nearest_bus_dist = [], [], []
    for _, row in df_places.iterrows():
        lat, lon = row['latitude'], row['longitude']
        # BTS nearest (vectorized)
        d_bts = haversine_vec(lat, lon, bts_lats, bts_lons)
        idx_b = int(np.argmin(d_bts))
        nearest_bts_idx.append(idx_b)
        min_bts_d = d_bts[idx_b]
        # Bus nearest (vectorized)
        d_bus = haversine_vec(lat, lon, bus_lats, bus_lons)
        min_bus_d = float(np.min(d_bus))
        nearest_bus_dist.append(min_bus_d)
        # display name
        th = th_name_map.get(row['place_name'], row['place_name'])
        suf = []
        if 'bts' in row['place_name'].lower() or min_bts_d <= 400: suf.append("BTS")
        if min_bus_d <= 300: suf.append("มีป้ายรถเมล์")
        display_names.append(f"{th} ({' / '.join(suf)})" if suf else th)

    df_places['display_th']      = display_names
    df_places['_bts_idx']        = nearest_bts_idx
    df_places['_min_bus_dist']   = nearest_bus_dist

    # Precompute crowd level per BTS station
    crowd_map = {}
    for stn in df_bts['clean_name']:
        word = stn.split()[0] if stn else ''
        sub  = df_passenger[df_passenger['Station'].str.contains(word, case=False, na=False)]
        if len(sub) == 0:
            crowd_map[stn] = 3
        else:
            avg = sub[sub['Time Period']=='Rush Hour']['Passengers In'].mean()
            if   avg < 300: crowd_map[stn] = 1
            elif avg < 600: crowd_map[stn] = 2
            elif avg < 900: crowd_map[stn] = 3
            else:           crowd_map[stn] = 4

    return df_places, df_bts, df_bus_stops, df_passenger, df_rf, bts_lats, bts_lons, bus_lats, bus_lons, crowd_map

(df_places, df_bts, df_bus_stops, df_passenger, df_rf,
 bts_lats, bts_lons, bus_lats, bus_lons, crowd_map) = load_and_precompute()

# ─── AI Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df_rf):
    # Model 1: Route RF (300 rows real data)
    le = LabelEncoder()
    df = df_rf.copy()
    df['Transport_Type_enc'] = le.fit_transform(df['Transport_Type'])
    feats = ['Elevator','Ramp','Accessible_Exit','Cost','Travel_Time',
             'BusSupport','Safety','Crowded_Level','Urgency',
             'Prefer_Safe','Prefer_Cheap','Transport_Type_enc']
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    clf1.fit(df[feats], df['Recommended'])
    # Model 2: Sidewalk RF (synthetic)
    X2 = np.array([
        [1.5,1,0,1],[0.7,0,3,0],[1.2,1,1,1],[1.8,1,0,1],[0.8,0,2,0],
        [2.0,1,0,1],[0.6,0,5,0],[1.4,1,1,1],[1.0,0,2,0],[2.5,1,0,1],
        [0.9,0,4,0],[1.6,1,1,1],[1.1,1,0,1],[0.5,0,6,0],[2.0,1,2,1],
    ])
    y2 = np.array([1,0,1,1,0,1,0,1,0,1,0,1,1,0,1])
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf2.fit(X2, y2)
    return clf1, clf2, le, feats

route_rf, sidewalk_rf, le_transport, rf_features = train_models(df_rf)

# ─── Fast lookup helpers (use precomputed arrays) ─────────────────────────────
def get_nearest_bts(lat, lon):
    d = haversine_vec(lat, lon, bts_lats, bts_lons)
    return df_bts.iloc[int(np.argmin(d))], float(np.min(d))

def get_nearest_bus_stops(lat, lon, n=3, max_dist=600):
    d = haversine_vec(lat, lon, bus_lats, bus_lons)
    mask = d <= max_dist
    if not mask.any():
        return pd.DataFrame()
    sub = df_bus_stops[mask].copy()
    sub['_d'] = d[mask]
    return sub.sort_values('_d').head(n)

def get_accessibility(row):
    lift = "✅ มี" if str(row.get('มีลิฟต์','')).strip()                    in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    ramp = "✅ มี" if str(row.get('ทางลาดสำหรับรถเข็น','')).strip()         in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    evac = "✅ มี" if str(row.get('พื้นที่สำหรับหนีภัยของคนพิการ','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    return lift, ramp, evac

def ai_route_predict(transport_type, elevator, ramp, acc_exit, cost, time_,
                     bus_sup, safety, crowd, urgency, pref_safe, pref_cheap):
    try:    t_enc = le_transport.transform([transport_type])[0]
    except: t_enc = 0
    row = pd.DataFrame([[elevator,ramp,acc_exit,cost,time_,bus_sup,safety,
                         crowd,urgency,pref_safe,pref_cheap,t_enc]], columns=rf_features)
    prob  = float(route_rf.predict_proba(row)[0][1])
    label = int(route_rf.predict(row)[0])
    imp   = dict(zip(rf_features, route_rf.feature_importances_))
    return label, prob, imp

def ai_sidewalk_predict(width, surface_val, obstacles, ramp_val):
    feats = np.array([[width, surface_val, obstacles, ramp_val]])
    prob  = float(sidewalk_rf.predict_proba(feats)[0][1])
    label = int(sidewalk_rf.predict(feats)[0])
    return label, prob

def get_ors_route(slon, slat, elon, elat):
    try:
        r = ors_client.directions(
            coordinates=[(slon,slat),(elon,elat)],
            profile='foot-walking', format='geojson')
        coords  = [[p[1],p[0]] for p in r['features'][0]['geometry']['coordinates']]
        summary = r['features'][0]['properties']['summary']
        steps   = r['features'][0]['properties']['segments'][0]['steps']
        return coords, summary['distance']/1000, summary['duration']/60, steps
    except:
        return None, None, None, None

factor_thai = {
    'Elevator':'ลิฟต์','Ramp':'ทางลาด','Accessible_Exit':'ทางออกอารยสถาปัตย์',
    'Cost':'ค่าใช้จ่าย','Travel_Time':'เวลาเดินทาง','Safety':'ความปลอดภัย',
    'Crowded_Level':'ความหนาแน่น','BusSupport':'รถเมล์ชานต่ำ',
    'Urgency':'ความเร่งด่วน','Prefer_Safe':'ความปลอดภัยสำคัญ',
    'Prefer_Cheap':'ราคาประหยัดสำคัญ','Transport_Type_enc':'ประเภทขนส่ง'
}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
place_list = sorted(df_places['display_th'].tolist())
def_start  = next((i for i,n in enumerate(place_list) if "อนุสาวรีย์" in n), 0)
def_end    = next((i for i,n in enumerate(place_list) if "จุฬา" in n), 1)

st.sidebar.header("🕹️ ตั้งค่าการเดินทาง")
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
obstacles      = st.sidebar.slider("จำนวนสิ่งกีดขวาง", 0, 10, 2)
has_ramp_sw    = st.sidebar.checkbox("มีทางลาดบริเวณทางเท้า", value=True)

# ─── ดึงข้อมูลสถานที่ (O(1) lookup จาก precomputed) ─────────────────────────
start_row = df_places[df_places['display_th']==start_label].iloc[0]
end_row   = df_places[df_places['display_th']==end_label].iloc[0]
slat, slon = start_row['latitude'], start_row['longitude']
elat, elon = end_row['latitude'],   end_row['longitude']

# nearest BTS (vectorized, fast)
bts_s, dist_bts_s = get_nearest_bts(slat, slon)
bts_e, dist_bts_e = get_nearest_bts(elat, elon)

lift_s, ramp_s, evac_s = get_accessibility(bts_s)
lift_e, ramp_e, evac_e = get_accessibility(bts_e)

# nearest bus stops (vectorized)
bus_near_s = get_nearest_bus_stops(slat, slon, n=3)
bus_near_e = get_nearest_bus_stops(elat, elon, n=3)

# crowd from precomputed map
crowd_s   = crowd_map.get(str(bts_s.get('clean_name','')), 3)
crowd_e   = crowd_map.get(str(bts_e.get('clean_name','')), 3)
avg_crowd = int((crowd_s + crowd_e) / 2)

dist_straight = haversine_scalar(slat, slon, elat, elon)
est_cost = int(16 + dist_straight/1000 * 2.5)
est_time = int(dist_straight/1000 * 6)

# ─── AI Predictions ──────────────────────────────────────────────────────────
transport_map = {"🚇 รถไฟฟ้า BTS":"BTS","🚌 รถเมล์ชานต่ำ":"Bus",
                 "🏥 สวัสดิการรถตู้รัฐ (โรงพยาบาล)":"BTS+Bus"}
transport_str = transport_map[travel_mode]
el_val   = 1 if ("✅" in lift_s and "✅" in lift_e) else 0
rmp_val  = 1 if ("✅" in ramp_s and "✅" in ramp_e) else 0
acc_exit = 1 if (el_val or rmp_val) else 0

ai_label, ai_prob, ai_imp = ai_route_predict(
    transport_str, el_val, rmp_val, acc_exit, est_cost, est_time,
    1 if "🚌" in travel_mode else 0,
    4 if prefer_safe else 3, avg_crowd, urgency_val,
    1 if prefer_safe else 0, 1 if prefer_cheap else 0)

surface_val = 1 if "เรียบ" in surface_opt else 0
ramp_sw_val = 1 if has_ramp_sw else 0
sw_label, sw_prob = ai_sidewalk_predict(sidewalk_width, surface_val, obstacles, ramp_sw_val)

# ─── ORS Route (only if key available) ───────────────────────────────────────
route_coords = route_dist_km = route_dur_min = route_steps = None
if ors_available:
    with st.spinner("📡 กำลังดึงเส้นทางจริงจาก OpenRouteService..."):
        route_coords, route_dist_km, route_dur_min, route_steps = \
            get_ors_route(slon, slat, elon, elat)

route_color = "green" if (ai_label==1 and sw_label==1) else ("orange" if ai_label==1 else "red")

# ─── LAYOUT ──────────────────────────────────────────────────────────────────
col_ai, col_route, col_map = st.columns([1, 1, 2])

# ── COL 1: AI Analysis ────────────────────────────────────────────────────────
with col_ai:
    st.markdown("### 🤖 AI Analysis")
    st.write(f"**จาก:** {start_label.split(' (')[0]}")
    st.write(f"**ถึง:** {end_label.split(' (')[0]}")
    st.write(f"📏 ระยะตรง: **{dist_straight/1000:.2f} กม.**")
    st.markdown("---")

    st.markdown("#### AI Function 1: Route RF")
    st.caption("วิเคราะห์เส้นทางจาก 300 rows training data (wheelchair_random_forest)")
    if ai_label == 1:
        st.success(f"✅ AI แนะนำเส้นทางนี้\nความเชื่อมั่น **{ai_prob*100:.1f}%**")
    else:
        st.error(f"⚠️ AI ไม่แนะนำ\n(ความเชื่อมั่น {ai_prob*100:.1f}%)")
    top3 = sorted(ai_imp.items(), key=lambda x: x[1], reverse=True)[:3]
    st.caption("🔍 ปัจจัยสำคัญ: " + ", ".join(
        [f"{factor_thai.get(k,k)} ({v*100:.0f}%)" for k,v in top3]))
    st.write(f"💸 ค่าโดยสาร: **{est_cost} บาท** | ⏱️ เวลา: **{est_time} นาที**")

    st.markdown("---")
    st.markdown("#### AI Function 2: Sidewalk RF")
    st.caption("ประเมินความปลอดภัยทางเท้า (Footpath) จากสภาพแวดล้อม")
    if sw_label == 1:
        st.success(f"🟢 ทางเท้าปลอดภัย — คะแนน **{sw_prob*100:.1f}%**")
    else:
        st.warning(f"🔴 ทางเท้ามีความเสี่ยง ({sw_prob*100:.1f}%) — แนะนำ GrabAssist")
    st.caption(f"กว้าง {sidewalk_width}ม. | {'เรียบ' if surface_val else 'ขรุขระ'} | "
               f"สิ่งกีดขวาง {obstacles} จุด | ทางลาด {'มี' if ramp_sw_val else 'ไม่มี'}")

    if route_dist_km:
        st.markdown("---")
        st.markdown("#### 🛣️ เส้นทางจริง (ORS)")
        st.write(f"📏 **{route_dist_km:.2f} กม.** | ⏱️ **{route_dur_min:.0f} นาที**")

# ── COL 2: Route Plan ─────────────────────────────────────────────────────────
with col_route:
    st.markdown("### 📋 แผนการเดินทาง")

    if "🚇" in travel_mode:
        walk_s = dist_bts_s
        walk_e = dist_bts_e
        mode_s = "🚶 เดินเท้า" if walk_s<=400 else "🚖 แนะนำ Grab/แท็กซี่"
        mode_e = "🚶 เดินเท้า" if walk_e<=400 else "🚖 แนะนำ Grab/แท็กซี่"
        st.info(f"**🟢 ขั้นที่ 1 — ทางเท้า:**\n\n"
                f"{mode_s} ไปสถานี **{bts_s['clean_name']}** ({walk_s:.0f} ม.)")
        with st.expander("ℹ️ อารยสถาปัตย์สถานีต้นทาง"):
            st.write(f"🛗 ลิฟต์: {lift_s} | ♿ ทางลาด: {ramp_s} | 🚨 พื้นที่หนีภัย: {evac_s}")
        if bts_s['clean_name'] != bts_e['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2 — รถไฟฟ้า:**\n\n"
                    f"**{bts_s['clean_name']}** → **{bts_e['clean_name']}**")
        st.info(f"**🔴 ขั้นที่ 3 — ทางเท้า:**\n\n"
                f"{mode_e} จาก **{bts_e['clean_name']}** ไปปลายทาง ({walk_e:.0f} ม.)")
        with st.expander("ℹ️ อารยสถาปัตย์สถานีปลายทาง"):
            st.write(f"🛗 ลิฟต์: {lift_e} | ♿ ทางลาด: {ramp_e} | 🚨 พื้นที่หนีภัย: {evac_e}")

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
        st.markdown("1. 🚶 เดินไปป้ายรถเมล์\n2. 🚌 ขึ้น Thai Smile Bus (แรมป์ไฮโดรลิก)\n3. 🏁 ลงที่ป้ายใกล้ปลายทาง")

    else:
        st.markdown("#### 🏥 สวัสดิการรถรับ-ส่งผู้พิการ")
        is_hosp = any(kw in end_label.lower() for kw in ['hospital','โรงพยาบาล'])
        if is_hosp:
            st.success("✅ ยืนยันสิทธิ์บริการรถรับ-ส่งผู้พิการ")
            st.markdown("📞 **สายด่วน กทม.:** โทร **1555 / 1479**\n\n🚑 **สปสช.:** โทร **1330**")
        else:
            st.error("❌ บริการจำกัดเฉพาะการเดินทางไปสถานพยาบาล")

    # Turn-by-Turn
    if route_steps:
        st.markdown("---")
        st.markdown("#### 🧭 Turn-by-Turn Navigation")
        for i, step in enumerate(route_steps[:10]):
            st.write(f"{i+1}. {step['instruction']} ({step['distance']:.0f} ม.)")
        if len(route_steps) > 10:
            st.caption(f"และอีก {len(route_steps)-10} ขั้นตอน...")

    # Crowd data
    st.markdown("---")
    st.markdown("#### 📈 ความหนาแน่นผู้โดยสาร")
    clean = str(bts_s.get('clean_name',''))
    sub   = df_passenger[df_passenger['Station'].str.contains(
        clean.split()[0] if clean else 'X', case=False, na=False)]
    if len(sub) > 0:
        avg_in = sub[sub['Time Period']=='Rush Hour']['Passengers In'].mean()
        st.metric("เฉลี่ย Rush Hour", f"{avg_in:.0f} คน/ชม.")
        st.write("🟢 ไม่แออัด" if avg_in<300 else ("🟡 ปานกลาง" if avg_in<600 else "🔴 แออัดมาก"))
    else:
        st.caption("ไม่พบข้อมูลผู้โดยสาร")

# ── COL 3: MAP ────────────────────────────────────────────────────────────────
with col_map:
    st.markdown("### 🗺️ แผนที่เส้นทางและทางเท้า")
    st.markdown("""<small>
    🟠 เส้นประ = ทางเท้า (Footpath) &nbsp;|&nbsp; 🟢 เส้นทึบ = รถไฟฟ้า / รถเมล์ &nbsp;|&nbsp;
    🔵 = สถานี BTS &nbsp;|&nbsp; 🟣 = ป้ายรถเมล์ &nbsp;|&nbsp; ⚠️ = สิ่งกีดขวาง
    </small>""", unsafe_allow_html=True)

    m = folium.Map(location=[(slat+elat)/2,(slon+elon)/2], zoom_start=14, tiles='CartoDB positron')
    fg_foot = folium.FeatureGroup(name="🚶 ทางเท้า (Footpath)", show=True)
    fg_tran = folium.FeatureGroup(name="🚇 เส้นทางรถไฟฟ้า/รถเมล์", show=True)
    fg_bts  = folium.FeatureGroup(name="🔵 สถานี BTS", show=True)
    fg_bus  = folium.FeatureGroup(name="🟣 ป้ายรถเมล์", show=True)
    fg_obs  = folium.FeatureGroup(name="⚠️ สิ่งกีดขวาง", show=True)

    # ต้นทาง / ปลายทาง
    folium.Marker([slat,slon], tooltip=f"ต้นทาง: {start_label.split(' (')[0]}",
        icon=folium.Icon(color='orange',icon='play',prefix='fa')).add_to(m)
    folium.Marker([elat,elon], tooltip=f"ปลายทาง: {end_label.split(' (')[0]}",
        icon=folium.Icon(color='red',icon='flag',prefix='fa')).add_to(m)

    if "🚇" in travel_mode or ("🚌" in travel_mode and len(bus_near_s)==0):
        # สถานี BTS
        folium.Marker([bts_s['lat'],bts_s['lng']],
            popup=folium.Popup(f"<b>{bts_s['clean_name']}</b><br>ลิฟต์:{lift_s}<br>ทางลาด:{ramp_s}",max_width=200),
            tooltip=f"BTS: {bts_s['clean_name']}",
            icon=folium.Icon(color='blue',icon='train',prefix='fa')).add_to(fg_bts)
        folium.Marker([bts_e['lat'],bts_e['lng']],
            popup=folium.Popup(f"<b>{bts_e['clean_name']}</b><br>ลิฟต์:{lift_e}<br>ทางลาด:{ramp_e}",max_width=200),
            tooltip=f"BTS: {bts_e['clean_name']}",
            icon=folium.Icon(color='darkblue',icon='train',prefix='fa')).add_to(fg_bts)
        # ทางเท้า 1
        folium.PolyLine([[slat,slon],[bts_s['lat'],bts_s['lng']]],
            color='#e67e22',weight=4,dash_array='8,8',
            tooltip=f"🚶 ทางเท้าช่วงที่ 1: {dist_bts_s:.0f} ม.").add_to(fg_foot)
        folium.CircleMarker([(slat+bts_s['lat'])/2,(slon+bts_s['lng'])/2],
            radius=7,color='#e67e22',fill=True,fill_color='#f39c12',fill_opacity=0.9,
            tooltip="🦮 จุดกึ่งกลางทางเท้า 1").add_to(fg_foot)
        # รถไฟฟ้า
        if bts_s['clean_name'] != bts_e['clean_name']:
            folium.PolyLine([[bts_s['lat'],bts_s['lng']],[bts_e['lat'],bts_e['lng']]],
                color='#00aa44',weight=7,tooltip="🚇 รถไฟฟ้า BTS").add_to(fg_tran)
        # ทางเท้า 3
        folium.PolyLine([[bts_e['lat'],bts_e['lng']],[elat,elon]],
            color='#e67e22',weight=4,dash_array='8,8',
            tooltip=f"🚶 ทางเท้าช่วงที่ 3: {dist_bts_e:.0f} ม.").add_to(fg_foot)
        folium.CircleMarker([(bts_e['lat']+elat)/2,(bts_e['lng']+elon)/2],
            radius=7,color='#e67e22',fill=True,fill_color='#f39c12',fill_opacity=0.9,
            tooltip="🦮 จุดกึ่งกลางทางเท้า 3").add_to(fg_foot)

    elif "🚌" in travel_mode:
        for _, b in bus_near_s.iterrows():
            folium.Marker([b['latitude'],b['longitude']],
                tooltip=f"🚏 {b['place_name']} ({b['_d']:.0f} ม.)",
                icon=folium.Icon(color='purple',icon='bus',prefix='fa')).add_to(fg_bus)
            folium.PolyLine([[slat,slon],[b['latitude'],b['longitude']]],
                color='#9b59b6',weight=3,dash_array='6,6',
                tooltip=f"🚶 ทางเท้าไปป้าย ({b['_d']:.0f} ม.)").add_to(fg_foot)
        for _, b in bus_near_e.iterrows():
            folium.Marker([b['latitude'],b['longitude']],
                tooltip=f"🚏 {b['place_name']} ({b['_d']:.0f} ม.)",
                icon=folium.Icon(color='purple',icon='bus',prefix='fa')).add_to(fg_bus)
            folium.PolyLine([[b['latitude'],b['longitude']],[elat,elon]],
                color='#9b59b6',weight=3,dash_array='6,6',
                tooltip=f"🚶 ทางเท้าจากป้าย ({b['_d']:.0f} ม.)").add_to(fg_foot)
        if len(bus_near_s)>0 and len(bus_near_e)>0:
            folium.PolyLine([
                [bus_near_s.iloc[0]['latitude'],bus_near_s.iloc[0]['longitude']],
                [bus_near_e.iloc[0]['latitude'],bus_near_e.iloc[0]['longitude']]],
                color='#8e44ad',weight=5,tooltip="🚌 เส้นทางรถเมล์").add_to(fg_tran)

    # ORS real route
    if route_coords:
        folium.PolyLine(route_coords,color=route_color,weight=6,opacity=0.85,
            tooltip=f"🛣️ เส้นทางจริง (ORS) — {ai_prob*100:.1f}%").add_to(fg_tran)

    # Obstacle markers
    if obstacles > 0:
        np.random.seed(42)
        for i in range(min(obstacles, 5)):
            frac = (i+1)/(obstacles+1)
            folium.Marker(
                [slat+frac*(elat-slat)+np.random.uniform(-0.001,0.001),
                 slon+frac*(elon-slon)+np.random.uniform(-0.001,0.001)],
                popup=f"⚠️ สิ่งกีดขวาง #{i+1}",
                tooltip=f"⚠️ Obstacle #{i+1} (AI Simulated)",
                icon=folium.Icon(color='orange',icon='warning-sign',prefix='glyphicon')
            ).add_to(fg_obs)

    for fg in [fg_foot, fg_tran, fg_bts, fg_bus, fg_obs]:
        fg.add_to(m)
    folium.LayerControl(position='topright').add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    st_folium(m, width="100%", height=620)

# ─── ส่วนล่าง ─────────────────────────────────────────────────────────────────
st.markdown("---")
col_tbl, col_chart = st.columns(2)
with col_tbl:
    st.markdown("#### ♿ ข้อมูลอารยสถาปัตย์สถานี BTS (สายสีเขียว)")
    st.dataframe(
        df_bts[['clean_name','มีลิฟต์','ทางลาดสำหรับรถเข็น',
                'พื้นที่สำหรับหนีภัยของคนพิการ','ลิฟต์ราวบันไดสำหรับคนพิการ']].rename(columns={
            'clean_name':'สถานี','มีลิฟต์':'ลิฟต์','ทางลาดสำหรับรถเข็น':'ทางลาด',
            'พื้นที่สำหรับหนีภัยของคนพิการ':'พื้นที่หนีภัย',
            'ลิฟต์ราวบันไดสำหรับคนพิการ':'ลิฟต์ราวบันได'}),
        use_container_width=True, height=260)
with col_chart:
    st.markdown("#### 🤖 Feature Importance (Route RF)")
    imp_df = pd.DataFrame(list(ai_imp.items()), columns=['ตัวแปร','ความสำคัญ'])
    imp_df['ตัวแปร'] = imp_df['ตัวแปร'].map(factor_thai).fillna(imp_df['ตัวแปร'])
    st.bar_chart(imp_df.sort_values('ความสำคัญ',ascending=False).set_index('ตัวแปร'),
                 use_container_width=True, height=260)

st.markdown("---")
st.caption("🤖 Random Forest AI (Route + Sidewalk) | 🗺️ OpenRouteService | 📊 CSV: BTS/Bus/Passenger | V8.1 Optimized")
