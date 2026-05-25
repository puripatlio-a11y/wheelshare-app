"""
AI Accessibility Route Planner for Wheelchair Users — Version 7.3 (Final Clean Path Fix)
แก้ไขปัญหา FileNotFoundError โดยการเพิ่ม Keyword-Based Matcher ดักจับไฟล์ passenger__1_ และ ThaiSmalieBus
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import re
import warnings
warnings.filterwarnings("ignore")

from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Accessibility Route Planner V7.3", layout="wide", page_icon="♿")

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

# ─── 🛠️ โหมดค้นหาไฟล์อัจฉริยะ ป้องกันปัญหาชื่อไฟล์ไม่ตรงและชื่อมีเลขห้อยท้าย ───
@st.cache_data
def load_all_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # สแกนไฟล์ทั้งหมดที่มีในระบบเพื่อทำ Map
    all_files_map = {}
    for root, dirs, files in os.walk(current_dir):
        for f in files:
            # ทำความสะอาดคีย์: แปลงเป็นตัวพิมพ์เล็กและดึงเฉพาะตัวอักษรกับตัวเลข
            clean_key = re.sub(r'[^a-z0-9]', '', f.lower())
            all_files_map[clean_key] = os.path.join(root, f)

    def find_file_by_keyword(keyword_target):
        """ค้นหาไฟล์ผ่าน Key หลัก ป้องกันปัญหาชื่อไฟล์ห้อยท้ายพวก __1__ หรือมีเว้นวรรคแปลกๆ"""
        clean_target = re.sub(r'[^a-z0-9]', '', keyword_target.lower())
        
        # 1. เช็คว่ามีคีย์ตรงๆ ไหม
        if clean_target in all_files_map:
            return all_files_map[clean_target]
            
        # 2. เช็คแบบ Partial Match (เช่น ส่ง 'passenger' ไปหาเจอรหัส 'bangkoktransitpassengerdata1csv')
        for key, full_path in all_files_map.items():
            if clean_target in key or key in clean_target:
                return full_path
                
        # คืนค่า Default หากเกิดเหตุสุดวิสัยหาไม่เจอจริงๆ
        return os.path.join(current_dir, keyword_target)

    # ─── เริ่มต้นโหลดข้อมูลจากไฟล์ (ใช้ Keyword Matcher ทั้งหมดเพื่อความชัวร์) ───
    
    # 1. ข้อมูลสถานที่หลักในกรุงเทพฯ
    df_places = pd.read_csv(find_file_by_keyword("bangkok_places_bus_spot.csv"))

    # 2. พิกัดสถานี BTS
    df_stations = pd.read_csv(find_file_by_keyword("bts_station.csv"))
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()

    # 3. ข้อมูลสิ่งอำนวยความสะดวก/อารยสถาปัตย์ BTS ของผู้พิการ
    df_acc = pd.read_csv(find_file_by_keyword("BTS for wheelchair users spreadsheet"))
    df_acc['clean_name'] = df_acc['สถานี'].str.replace('สถานี', '').str.strip()

    # รวมข้อมูล BTS
    df_bts = pd.merge(
        df_acc, df_stations[['clean_name', 'lat', 'lng', 'btsline', 'location']],
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)

    # 4. ข้อมูลพิกัดป้ายรถเมล์
    df_bus_stops = pd.read_csv(find_file_by_keyword("bangkok_bus_stops_coordinates.csv"))

    # 5. ข้อมูลสถิติจำนวนผู้โดยสาร (แก้บั๊ก __1__ ดึงผ่านคีย์คำว่า passenger)
    df_passenger = pd.read_csv(find_file_by_keyword("passenger"))

    # 6. ข้อมูลสำหรับเทรนโมเดล AI Random Forest
    df_rf = pd.read_csv(find_file_by_keyword("wheelchair_random_forest_300rows.csv"))

    # 7. สายรถเมล์ของไทยสมายล์บัส (ดึงผ่านคีย์คำว่า ThaiSmalieBus)
    df_bus_routes = None
    bus_route_path = find_file_by_keyword("ThaiSmalieBus")
    if os.path.exists(bus_route_path):
        df_bus_routes = pd.read_csv(bus_route_path)

    return df_places, df_bts, df_bus_stops, df_passenger, df_rf, df_bus_routes

# เรียกใช้ฟังก์ชันดึงข้อมูลที่มีระบบแมตช์คำอัจฉริยะ
df_places, df_bts, df_bus_stops, df_passenger, df_rf, df_bus_routes = load_all_data()


# ─── 🤖 AI FUNCTION: Train Random Forest Model ───────────────────────────────
@st.cache_resource
def train_random_forest(df_rf):
    """AI Function: เทรน Random Forest Classifier เพื่อทำนาย Recommended (1=แนะนำ, 0=ไม่แนะนำ)"""
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
    """🤖 AI Predict: ใช้ Model ทำนายผลลัพธ์และความเชื่อมั่นของเส้นทางสำหรับรถเข็น"""
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


# ─── ฟังก์ชันคำนวณตำแหน่ง ─────────────────────────────────────────────────────
def nearest_bus_stops(lat, lon, df_bus_stops, n=3, max_dist=600):
    """ค้นหาป้ายรถเมล์ที่ใกล้ที่สุดในรัศมีกำหนด"""
    df_bus_stops = df_bus_stops.copy()
    df_bus_stops['dist'] = df_bus_stops.apply(
        lambda r: haversine(lat, lon, r['latitude'], r['longitude']), axis=1)
    nearby = df_bus_stops[df_bus_stops['dist'] <= max_dist].sort_values('dist')
    return nearby.head(n)

def nearest_bts(lat, lon):
    """หาสถานี BTS สีเขียวที่ใกล้จุดพิกัดมากที่สุด"""
    df_bts['dist'] = df_bts.apply(
        lambda r: haversine(lat, lon, r['lat'], r['lng']), axis=1)
    return df_bts.sort_values('dist').iloc[0]


# ─── จับคู่และแสดงชื่อภาษาไทย ──────────────────────────────────────────────────
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

def make_display_name(row):
    name = row['place_name']
    th = th_name_map.get(name, name)
    suffixes = []
    bts_dist = nearest_bts(row['latitude'], row['longitude'])['dist']
    if 'bts' in name.lower() or bts_dist <= 400:
        suffixes.append("BTS")
    nearby_bus = nearest_bus_stops(row['latitude'], row['longitude'], df_bus_stops, n=1, max_dist=300)
    if len(nearby_bus) > 0:
        suffixes.append("มีป้ายรถเมล์ใกล้")
    return f"{th} ({' / '.join(suffixes)})" if suffixes else th

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


# ─── ประมวลผลข้อมูลเส้นทาง ────────────────────────────────────────────────────
start_info = df_places[df_places['display_th'] == start_label].iloc[0]
end_info   = df_places[df_places['display_th'] == end_label].iloc[0]

bts_start = nearest_bts(start_info['latitude'], start_info['longitude'])
bts_end   = nearest_bts(end_info['latitude'], end_info['longitude'])

def get_accessibility(bts_row):
    has_lift  = "✅ มี" if str(bts_row.get('มีลิฟต์','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    has_ramp  = "✅ มี" if str(bts_row.get('ทางลาดสำหรับรถเข็น','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    evac_area = "✅ มี" if str(bts_row.get('พื้นที่สำหรับหนีภัยของคนพิการ','')).strip() in ['1','1.0','มี','Yes'] else "❌ ไม่มี"
    return has_lift, has_ramp, evac_area

lift_s, ramp_s, evac_s = get_accessibility(bts_start)
lift_e, ramp_e, evac_e = get_accessibility(bts_end)

bus_near_start = nearest_bus_stops(start_info['latitude'], start_info['longitude'], df_bus_stops, n=3)
bus_near_end   = nearest_bus_stops(end_info['latitude'],   end_info['longitude'],   df_bus_stops, n=3)

def get_crowd_level(station_name_en):
    if df_passenger is None or len(df_passenger) == 0:
        return 2
    sub = df_passenger[df_passenger['Station'].str.contains(
        station_name_en.split()[0] if station_name_en else '', case=False, na=False)]
    if len(sub) == 0: return 2
    avg_in = sub['Passengers In'].mean()
    if avg_in < 400: return 1
    if avg_in < 800: return 2
    return 3

crowd_s = get_crowd_level(str(bts_start.get('clean_name','')))
crowd_e = get_crowd_level(str(bts_end.get('clean_name','')))
avg_crowd = int((crowd_s + crowd_e) / 2)


# ─── ทำนายความเหมาะสมด้วย AI ─────────────────────────────────────────────────
transport_map = {"🚇 รถไฟฟ้า BTS": "BTS", "🚌 รถเมล์ชานต่ำ": "Bus", "🏥 สวัสดิการรถตู้รัฐ (โรงพยาบาล)": "BTS+Bus"}
transport_type_str = transport_map[travel_mode]

el_s  = 1 if "✅" in lift_s else 0
rmp_s = 1 if "✅" in ramp_s else 0
el_e  = 1 if "✅" in lift_e else 0
rmp_e = 1 if "✅" in ramp_e else 0
elevator_val = 1 if (el_s and el_e) else 0
ramp_val     = 1 if (rmp_s and rmp_e) else 0

dist_total = haversine(start_info['latitude'], start_info['longitude'], end_info['latitude'], end_info['longitude'])
est_cost = int(16 + dist_total / 1000 * 2.5)
est_time = int(dist_total / 1000 * 6)

ai_label, ai_prob, ai_importances = ai_predict_route(
    transport_type=transport_type_str, elevator=elevator_val, ramp=ramp_val,
    accessible_exit=1 if (elevator_val or ramp_val) else 0, cost=est_cost, travel_time=est_time,
    bus_support=1 if "🚌" in travel_mode else 0, safety=4 if prefer_safe else 3,
    crowded_level=avg_crowd, urgency=urgency_val, prefer_safe=1 if prefer_safe else 0, prefer_cheap=1 if prefer_cheap else 0
)


# ─── LAYOUT การจัดวางแอป ──────────────────────────────────────────────────────
col_info, col_map = st.columns([1, 2])

with col_info:
    st.markdown("### 📊 ผลการวิเคราะห์เส้นทาง")
    st.markdown(f"**ต้นทาง:** {start_label.split(' (')[0]} → **ปลายทาง:** {end_label.split(' (')[0]}")
    st.markdown(f"**ระยะขจัดตรง:** {dist_total/1000:.2f} กม.")

    st.markdown("---")
    st.markdown("#### 🤖 ผลประเมินโดย AI (Random Forest)")

    if ai_label == 1:
        st.success(f"✅ AI แนะนำเส้นทางนี้ — ความน่าเชื่อถือ **{ai_prob*100:.1f}%**")
    else:
        st.error(f"⚠️ AI ไม่แนะนำเนื่องจากข้อจำกัดทางสถาปัตยกรรม ({ai_prob*100:.1f}%)")

    factor_thai = {
        'Elevator':'ลิฟต์', 'Ramp':'ทางลาด', 'Accessible_Exit':'ทางออกคนพิการ', 'Cost':'ค่าใช้จ่าย',
        'Travel_Time':'เวลาเดินทาง', 'Safety':'ความปลอดภัย', 'Crowded_Level':'ความหนาแน่นผู้โดยสาร',
        'BusSupport':'การรองรับชานต่ำ', 'Urgency':'ความเร่งด่วน', 'Prefer_Safe':'เน้นปลอดภัย', 'Prefer_Cheap':'เน้นประหยัด'
    }
    top3 = sorted(ai_importances.items(), key=lambda x: x[1], reverse=True)[:3]
    st.caption(f"🔍 ปัจจัยหลักที่ AI ใช้คิด: " + ", ".join([f"{factor_thai.get(k,k)} ({v*100:.0f}%)" for k,v in top3]))
    st.markdown(f"💰 คาดการณ์ค่าใช้จ่าย: **{est_cost} บาท** | ⏱️ เวลาเดินทางโดยประมาณ: **{est_time} นาที**")

    st.markdown("---")

    # แผนการเดินทางตามรูปแบบที่เลือก
    if "🚇" in travel_mode:
        st.markdown("#### 🚇 แผนการเดินรถไฟฟ้า BTS")
        walk_s = haversine(start_info['latitude'], start_info['longitude'], bts_start['lat'], bts_start['lng'])
        walk_e = haversine(end_info['latitude'], end_info['longitude'], bts_end['lat'], bts_end['lng'])

        st.info(f"🚶 **เฟส 1 - ทางเท้า (Footpath):** เดินเท้าเชื่อมต่อไปยังสถานี **{bts_start['clean_name']}** ({walk_s:.0f} เมตร)")
        with st.expander("ตรวจสอบอารยสถาปัตย์สถานีต้นทาง"):
            st.write(f"🛗 ลิฟต์สำหรับรถเข็น: {lift_s} | 📐 ทางลาดชัน: {ramp_s}")
            st.write(f"🚨 พื้นที่เซฟตี้หนีภัย: {evac_s}")

        if bts_start['clean_name'] != bts_end['clean_name']:
            st.info(f"🚇 **เฟส 2 - ตัวขบวนรถไฟฟ้า:** นั่งรถไฟฟ้าบีทีเอสขบวนชานชลาเสมอระดับจาก {bts_start['clean_name']} ไปยังสถานี {bts_end['clean_name']}")

        st.info(f"🚶 **เฟส 3 - ทางเท้าปลายทาง:** เดินจากสถานี **{bts_end['clean_name']}** เข้าสู่จุดหมาย ({walk_e:.0f} เมตร)")
        with st.expander("ตรวจสอบอารยสถาปัตย์สถานีปลายทาง"):
            st.write(f"🛗 ลิฟต์สำหรับรถเข็น: {lift_e} | 📐 ทางลาดชัน: {ramp_e}")

    elif "🚌" in travel_mode:
        st.markdown("#### 🚌 แผนการเดินทางรถเมล์ชานต่ำ")
        if len(bus_near_start) > 0 and len(bus_near_end) > 0:
            st.success(f"🚏 ป้ายขึ้นรถแนะนำ: **{bus_near_start.iloc[0]['place_name']}** ({bus_near_start.iloc[0]['dist']:.0f} ม.)")
            st.success(f"🚏 ป้ายลงรถแนะนำ: **{bus_near_end.iloc[0]['place_name']}** ({bus_near_end.iloc[0]['dist']:.0f} ม.)")
            st.write("💡 แนะนำบริการของ *ไทยสมายล์บัส* ซึ่งเป็นรถบัสไฟฟ้าชานต่ำ ชานพับไฮโดรลิกลาดลงพื้น และมีเบลท์ล็อกวีลแชร์")
        else:
            st.warning("⚠️ ไม่พบรัศมีป้ายรถเมล์ในเกณฑ์ปลอดภัย แนะนำให้เปลี่ยนระบบไปใช้รถไฟฟ้าเป็นหลัก")

    else:
        st.markdown("#### 🏥 บริการรถรับส่งสวัสดิการภาครัฐ")
        if any(kw in end_label.lower() for kw in ['hospital','โรงพยาบาล','รพ.']):
            st.success("🚑 ปลายทางเข้าเกณฑ์สถานพยาบาล สามารถใช้สิทธิ์โทรจองรถตู้ทางลาดสวัสดิการ 1555 หรือ 1479")
        else:
            st.error("❌ ข้อมูลสิทธิ์สวัสดิการนี้ สงวนสิทธิ์เดินทางเฉพาะกรณีพบแพทย์หรือไปโรงพยาบาลเท่านั้น")

    # ส่วนสถิติความหนาแน่น
    st.markdown("---")
    st.markdown("#### 📈 ข้อมูลความหนาแน่นสถานี")
    st.write(f"ระดับความหนาแน่นสถานี {bts_start['clean_name']}: " + ("🟢 ต่ำ-ปลอดภัย" if crowd_s==1 else ("🟡 ปานกลาง" if crowd_s==2 else "🔴 หนาแน่นสูง")))


with col_map:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดเส้นทางและทางเท้า")
    st.markdown("<small>🟠 เส้นประ = ทางเท้าสัญจรคนพิการ (Footpath) | 🟢 เส้นทึบ = ทางรถไฟฟ้า</small>", unsafe_allow_html=True)

    c_lat = (start_info['latitude'] + end_info['latitude']) / 2
    c_lon = (start_info['longitude'] + end_info['longitude']) / 2
    m = folium.Map(location=[c_lat, c_lon], zoom_start=14, tiles='CartoDB positron')

    # มาร์กเกอร์จุดรับส่ง
    folium.Marker([start_info['latitude'], start_info['longitude']], tooltip="จุดต้นทาง", icon=folium.Icon(color='orange', icon='play')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], tooltip="จุดปลายทาง", icon=folium.Icon(color='red', icon='flag')).add_to(m)

    if "🚇" in travel_mode:
        folium.Marker([bts_start['lat'], bts_start['lng']], tooltip=f"BTS {bts_start['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([bts_end['lat'], bts_end['lng']], tooltip=f"BTS {bts_end['clean_name']}", icon=folium.Icon(color='darkblue', icon='train', prefix='fa')).add_to(m)
        
        # วาด Footpath เส้นประทางเดิน
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [bts_start['lat'], bts_start['lng']]], color='#e67e22', weight=4, dash_array='7,7').add_to(m)
        folium.PolyLine([[bts_end['lat'], bts_end['lng']], [end_info['latitude'], end_info['longitude']]], color='#e67e22', weight=4, dash_array='7,7').add_to(m)
        # เส้นวิ่งรถไฟฟ้า
        folium.PolyLine([[bts_start['lat'], bts_start['lng']], [bts_end['lat'], bts_end['lng']]], color='#2ecc71', weight=6).add_to(m)

    elif "🚌" in travel_mode and len(bus_near_start) > 0:
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [bus_near_start.iloc[0]['latitude'], bus_near_start.iloc[0]['longitude']]], color='#e67e22', weight=4, dash_array='7,7').add_to(m)
        folium.PolyLine([[bus_near_start.iloc[0]['latitude'], bus_near_start.iloc[0]['longitude']], [bus_near_end.iloc[0]['latitude'], bus_near_end.iloc[0]['longitude']]], color='#9b59b6', weight=5).add_to(m)
        folium.PolyLine([[bus_near_end.iloc[0]['latitude'], bus_near_end.iloc[0]['longitude']], [end_info['latitude'], end_info['longitude']]], color='#e67e22', weight=4, dash_array='7,7').add_to(m)

    st_folium(m, width="100%", height=550)

# แสดงฟีเจอร์บาร์ชาร์ตใต้กราฟ
st.markdown("---")
col_g1, col_g2 = st.columns(2)
with col_g1:
    st.markdown("#### ♿ อารยสถาปัตย์ภาพรวมระบบ")
    st.dataframe(df_bts[['clean_name','มีลิฟต์','ทางลาดสำหรับรถเข็น']].rename(columns={'clean_name':'สถานี'}), use_container_width=True, height=200)
with col_g2:
    st.markdown("#### 🤖 ปัจจัยที่ส่งผลต่อโมเดล AI")
    imp_df = pd.DataFrame(list(ai_importances.items()), columns=['ตัวแปร','น้ำหนัก']).sort_values('น้ำหนัก', ascending=False)
    imp_df['ตัวแปร'] = imp_df['ตัวแปร'].map(factor_thai).fillna(imp_df['ตัวแปร'])
    st.bar_chart(imp_df.set_index('ตัวแปร')['น้ำหนัก'], use_container_width=True, height=200)
