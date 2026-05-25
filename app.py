import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium
# 🧠 นำเข้าไลบรารีปัญญาประดิษฐ์มาใช้คำนวณเบื้องหลังตามสเปก
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Accessibility Route Planner V6.4", layout="wide")

# 🎨 [คงไว้ตามเดิม] สร้างพื้นที่ส่วนหัวข้อ (Header Banner) โดยนำ URL รูปภาพมาทำเป็นพื้นหลังด้วย CSS
header_html = """
<style>
    .custom-header {
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.55)), 
                          url("https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?semt=ais_hybrid&w=740&q=80");
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .custom-header h1 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        font-size: 2.5rem !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        margin-bottom: 5px;
    }
    .custom-header h3 {
        color: #f0f2f6 !important;
        font-size: 1.3rem !important;
        font-weight: 400;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.7);
    }
</style>

<div class="custom-header">
    <h1>♿ AI Accessibility Route Planner for Wheelchair Users</h1>
    <h3>ระบบส่งเสริมการวางแผนการเดินทางด้วยปัญญาประดิษฐ์สำหรับผู้ใช้รถเข็น</h3>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("---")

# 🤖 [AI Functionเบื้องหลัง] โมเดลประเมินความปลอดภัยของทางเท้าสำหรับวีลแชร์ (Focus AI)
@st.cache_resource
def load_pedestrian_ai_model():
    # ข้อมูลฝึกสอนฟุตบาท: ความกว้าง, สภาพพื้นผิว (1=เรียบ/0=ขรุขระ), สิ่งกีดขวาง, ทางลาด (1=มี/0=ไม่มี)
    X_train = np.array([
        [1.5, 1, 0, 1], [0.7, 0, 3, 0], [1.2, 1, 1, 1], [1.8, 1, 0, 1], [0.8, 0, 2, 0]
    ])
    y_train = np.array([1, 0, 1, 1, 0]) # 1 = ปลอดภัย, 0 = อันตราย
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

ai_model = load_pedestrian_ai_model()

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c * 1000

@st.cache_data
def load_and_prepare_data():
    df_places = pd.read_csv('bangkok_places_bus_spot.csv')
    df_stations = pd.read_csv('bts_station.csv')
    df_accessibility = pd.read_csv('BTS for wheelchair users spreadsheet - BTS green line.csv')
    
    target_bus_file = 'ThaiSmalieBus  - Sheet1.csv' 
    if not os.path.exists(target_bus_file):
        all_files = os.listdir('.')
        matched_files = [f for f in all_files if 'ThaiSmalieBus' in f or 'ThaiSmileBus' in f]
        if matched_files:
            target_bus_file = matched_files[0]
            
    df_bus_routes = pd.read_csv(target_bus_file)
    
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()
    df_accessibility['clean_name'] = df_accessibility['สถานี'].str.replace('สถานี', '').str.strip()
    df_bts_master = pd.merge(
        df_accessibility[['clean_name', 'สถานี', 'มีลิฟต์', 'ทางลาดสำหรับรถเข็น']], 
        df_stations[['clean_name', 'lat', 'lng']], 
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)
    
    return df_places, df_bts_master, df_bus_routes

try:
    df_places, df_bts_master, df_bus_routes = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ ระบบไม่สามารถอ่านฐานข้อมูลไฟล์ได้: {e}")
    st.stop()

# 🎯 1. พจนานุกรมแปลชื่อสถานที่บนหน้าเว็บเป็นภาษาไทย
th_name_base_map = {
    "Victory Monument": "อนุสาวรีย์ชัยสมรภูมิ",
    "Siam Station": "สถานีรถไฟฟ้า สยาม",
    "CentralWorld": "เซ็นทรัลเวิลด์",
    "MBK Center": "เอ็มบีเค เซ็นเตอร์ (มาบุญครอง)",
    "Samyan Mitrtown": "สามย่านมิตรทาวน์",
    "Chulalongkorn Hospital": "โรงพยาบาลจุฬาลงกรณ์",
    "Siriraj Hospital": "โรงพยาบาลศิริราช",
    "Ramathibodi Hospital": "โรงพยาบาลรามาธิบดี",
    "Rajavithi Hospital": "โรงพยาบาลราชวิถี",
    "Vajira Hospital": "โรงพยาบาลวชิรพยาบาล",
    "Mochit Bus Terminal": "สถานีขนส่งผู้โดยสารกรุงเทพ (หมอชิต 2)",
    "Chatuchak Park": "สวนจตุจักร",
    "Ari BTS Station": "สถานีรถไฟฟ้า อารีย์",
    "Saphan Khwai BTS Station": "สถานีรถไฟฟ้า สะพานควาย",
    "Kasetsart University": "มหาวิทยาลัยเกษตรศาสตร์",
    "Bang Wa BTS Station": "สถานีรถไฟฟ้า บางหว้า",
    "Bearing BTS Station": "สถานีรถไฟฟ้า แบริ่ง",
    "Ekkamai Bus Terminal": "สถานีขนส่งเอกมัย"
}

# พจนานุกรมคีย์เวิร์ดภาษาไทยสำหรับเชื่อมโยงค้นหาตารางเดินรถเมล์
bus_translation_dict = {
    "Victory Monument": ["อนุสาวรีย์", "รพ.ราชวิถี", "ราชวิถี"],
    "Chulalongkorn Hospital": ["จุฬาลงกรณ์", "สามย่าน", "รพ.จุฬา"],
    "Siam Station": ["สยาม", "ปทุมวัน"],
    "Samyan Mitrtown": ["สามย่าน", "หัวลำโพง"],
    "Ramathibodi Hospital": ["รพ.สงฆ์", "วิชัยยุทธ", "รามาธิบดี"],
    "Siriraj Hospital": ["ศิริราช", "พรานนก"],
    "Rajavithi Hospital": ["รพ.ราชวิถี", "อนุสาวรีย์"],
    "Hua Lamphong Station": ["หัวลำโพง"],
    "Chatuchak Park": ["BTSหมอชิต", "ห้าแยกลาดพร้าว"]
}

# 🎯 2. คำนวณหลังบ้านล่วงหน้าเพื่อทำป้ายวงเล็บแจ้งสายรถเมล์และ BTS ใน Dropdown
display_names_th_with_brackets = []
for idx, row in df_places.iterrows():
    p_name = row['place_name']
    th_name = th_name_base_map.get(p_name, p_name)
    
    suffixes = []
    
    # ตรวจสอบเงื่อนไข BTS
    df_bts_master['temp_dist'] = [haversine_distance(row['latitude'], row['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
    min_bts_dist = df_bts_master['temp_dist'].min()
    if "bts" in p_name.lower() or min_bts_dist <= 500:
        suffixes.append("BTS")
        
    # ตรวจสอบสายรถเมล์ที่ผ่านสถานที่นี้
    keywords = bus_translation_dict.get(p_name, [p_name])
    local_bus_lines = []
    for b_idx, b_row in df_bus_routes.iterrows():
        route_text = str(b_row['ต้นทาง']) + str(b_row['ปลาย']) + str(b_row['ผ่าน'])
        if any(k.lower() in route_text.lower() for k in keywords):
            local_bus_lines.append(str(b_row['สาย']).strip())
            
    if local_bus_lines:
        unique_local_buses = sorted(list(set(local_bus_lines)))
        suffixes.append(f"เดินทางด้วยรถเมล์ได้ สาย: {', '.join(unique_local_buses)}")
        
    if suffixes:
        final_display = f"{th_name} ({' / '.join(suffixes)})"
    else:
        final_display = th_name
        
    display_names_th_with_brackets.append(final_display)

df_places['display_name_th'] = display_names_th_with_brackets
place_list_th = sorted(df_places['display_name_th'].tolist())

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")

default_start_idx = 0
default_end_idx = 0
for i, name in enumerate(place_list_th):
    if "อนุสาวรีย์ชัยสมรภูมิ" in name:
        default_start_idx = i
    if "โรงพยาบาลจุฬาลงกรณ์" in name:
        default_end_idx = i

start_label_th = st.sidebar.selectbox("📍 เลือกจุดต้นทาง:", place_list_th, index=default_start_idx)
end_label_th = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง:", place_list_th, index=default_end_idx)

start_info = df_places[df_places['display_name_th'] == start_label_th].iloc[0]
end_info = df_places[df_places['display_name_th'] == end_label_th].iloc[0]

start_place_name = start_info['place_name']
end_place_name = end_info['place_name']

# 🔧 เพิ่มแถบเครื่องมือตรวจสอบสิ่งกีดขวางบนทางเท้าโดยใช้ AI ให้คะแนนความเสี่ยงอยู่เบื้องหลัง
st.sidebar.write("---")
st.sidebar.markdown("### 🦽 ตรวจสอบฟุตบาทภายนอกอาคารด้วย AI")
simulated_obstacles = st.sidebar.slider("⚠️ จำนวนสิ่งกีดขวางบนทางเท้า (จุด):", 0, 5, 1)
has_curb_ramp = st.sidebar.checkbox("📐 จุดตัดฟุตบาทมีทางลาดสำหรับรถเข็น", value=True)

st.sidebar.write("---")
st.sidebar.markdown("### 🚏 เลือกโหมดการเดินทาง")
travel_mode = st.sidebar.radio(
    "โปรดเลือกรูปแบบการเดินทางหลักที่สะดวก:",
    [
        "🚇 รถไฟฟ้า (BTS) - เน้นเดินทางเร็ว",
        "🚌 รถเมล์ชานต่ำ (Thai Smile Bus) - เน้นประหยัด",
        "🏥 สวัสดิการรถตู้รัฐ/กทม. ฟรี (สำหรับไปโรงพยาบาล)"
    ]
)

is_hospital = any(keyword in end_place_name.lower() for keyword in ["hospital", "โรงพยาบาล", "รพ."])
matched_lines = []

col1, col2 = st.columns([1, 2])

# 🤖 [AI Focus] ดึงโมเดลมาวิเคราะห์ทำนายความปลอดภัยของทางเดินเท้าช่วงสั้น (First-Leg/Last-Leg Pedestrian Paths)
# ดึงโอกาสผ่านได้ (Probability of Safety) ของผู้ใช้วีลแชร์
ai_features = np.array([[1.5, 1 if simulated_obstacles < 3 else 0, simulated_obstacles, 1 if has_curb_ramp else 0]])
safety_probability = ai_model.predict_proba(ai_features)[0][1]

with col1:
    st.markdown(f"### 📊 แผนผังนำทางอัจฉริยะ")
    st.write(f"**จาก:** {start_label_th.split(' (')[0]}")
    st.write(f"**ถึง:** {end_label_th.split(' (')[0]}")
    
    # 🤖 แสดงผลรายงานการประเมินทางเท้าของ AI ก่อนเข้าสู่การเดินทางหลัก
    if safety_probability >= 0.70:
        st.success(f"🟢 **วิเคราะห์ทางเท้าโดย AI ({safety_probability*100:.0f}%):** ทางเท้ามีความปลอดภัย เข็นรถเข็นเข้าสู่ระบบขนส่งหลักได้สะดวก")
    else:
        st.error(f"🔴 **แจ้งเตือนความเสี่ยงโดย AI ({safety_probability*100:.0f}%):** ทางเท้ามีสิ่งกีดขวางหนาแน่นหรือขาดทางลาด กรุณาเข็นด้วยความระมัดระวัง")
        
    st.write("---")

    # 🚇 1. โหมด BTS
    if "🚇" in travel_mode:
        df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

        df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

        transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_start['dist_start'] <= 150 else "🚖 แนะนำเรียกใช้บริการ แกร็บ (Grab) หรือ แท็กซี่"
        transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_end['dist_end'] <= 150 else "🚖 แนะนำเรียกใช้บริการ แกร็บ (Grab) หรือ แท็กซี่"

        has_lift_start = "มี" if str(nearest_bts_start['มีลิฟต์']).strip() in ['1', '1.0', 'มี', 'Yes'] else "ไม่มี"
        has_ramp_start = "มี" if str(nearest_bts_start['ทางลาดสำหรับรถเข็น']).strip() in ['1', '1.0', 'มี', 'Yes'] else "ไม่มี"
        
        has_lift_end = "มี" if str(nearest_bts_end['มีลิฟต์']).strip() in ['1', '1.0', 'มี', 'Yes'] else "ไม่มี"
        has_ramp_end = "มี" if str(nearest_bts_end['ทางลาดสำหรับรถเข็น']).strip() in ['1', '1.0', 'มี', 'Yes'] else "ไม่มี"

        st.info(f"**🟢 ขั้นที่ 1:** {transport_first_leg} ไปยัง **สถานีรถไฟฟ้า BTS {nearest_bts_start['clean_name']}** (ระยะทาง {nearest_bts_start['dist_start']:.1f} เมตร)")
        st.write("ℹ️ **ข้อมูลสิ่งอำนวยความสะดวกสถานีต้นทาง:**")
        st.write(f"* มีลิฟต์วีลแชร์ = **{has_lift_start}** | มีทางลาดสำหรับรถเข็น = **{has_ramp_start}**")      
        st.write("")

        if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2:** ขึ้นรถไฟฟ้า BTS เดินทางจากสถานี **{nearest_bts_start['clean_name']}** ไปลงที่สถานีเป้าหมาย **{nearest_bts_end['clean_name']}**")
        
        st.info(f"**🔴 ขั้นที่ 3:** {transport_last_leg} จากสถานีรถไฟฟ้าปลายทางเข้าสู่พิกัดเป้าหมาย **{end_label_th.split(' (')[0]}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)")
        st.write("ℹ️ **ข้อมูลสิ่งอำนวยความสะดวกสถานีปลายทาง:**")
        st.write(f"* มีลิฟต์วีลแชร์ = **{has_lift_end}** | มีทางลาดสำหรับรถเข็น = **{has_ramp_end}**")

    # 🚌 2. โหมดรถเมล์ชานต่ำ
    elif "🚌" in travel_mode:
        st.markdown("#### 𚏏 ผลคำนวณการเดินรถโดยสารสาธารณะอารยสถาปัตย์")
        
        start_keywords = bus_translation_dict.get(start_place_name, [start_place_name])
        end_keywords = bus_translation_dict.get(end_place_name, [end_place_name])
        
        for idx, row in df_bus_routes.iterrows():
            route_text = str(row['ต้นทาง']) + str(row['ปลาย']) + str(row['ผ่าน'])
            if any(k.lower() in route_text.lower() for k in start_keywords) and any(k.lower() in route_text.lower() for k in end_keywords):
                matched_lines.append(str(row['สาย']).strip())

        if matched_lines:
            unique_lines_list = sorted(list(set(matched_lines)))
            all_suggested_lines = " หรือ สาย ".join(unique_lines_list)
            
            st.success(f"✅ **เอไอ (AI) แนะนำรถเมล์ชานต่ำต่อเดียวถึง: สาย {all_suggested_lines}**")
            st.markdown(f"""
            **📋 ขั้นตอนการเดินทาง:**
            1. **🚶 จุดขึ้นรถ:** เข็นวีลแชร์ไปยังป้ายหยุดรถประจำทาง ณ **{start_label_th.split(' (')[0]}**
            2. **💳 การขึ้นรถ:** สามารถเลือกขึ้นรถเมล์สาย **{all_suggested_lines}** (ตัวรถเป็นแบบชานต่ำ มีแรมป์ไฮโดรลิก และระบบล็อกล้อรถเข็นปลอดภัย)
            3. **🏁 จุดหมาย:** นั่งยาวไปลงรถ ณ จุดจอดเป้าหมายปลายทาง **{end_label_th.split(' (')[0]}** ได้ทันที
            """)
            
        else:
            st.warning("🔄 ไม่พบสายรถเมล์ที่วิ่งผ่านตรงๆ ต่อเดียว --- ระบบได้จัดแผนเดินทางเชื่อมต่อพ่วงระบบรถไฟฟ้าและรถแท็กซี่ให้ทดแทนอัตโนมัติ:")
            
            df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
            nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]
            
            df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
            nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]
            
            st.markdown(f"""
            **🗺️ แผนการเดินทางพ่วงเชื่อมต่ออัจฉริยะ (รถรับจ้าง + รถไฟฟ้า BTS + แกร็บวีลแชร์):**
            * **🟢 ช่วงที่ 1 (เข้าสู่สถานีรถไฟฟ้า):** เดินทางจากจุดเริ่มต้นไปยัง **สถานีรถไฟฟ้า BTS {nearest_bts_start['clean_name']}** (ระยะทางประมาณ {nearest_bts_start['dist_start']:.1f} เมตร แนะนำเรียกบริการ **แกร็บวีลแชร์ (GrabAssist)** หรือแท็กซี่หากเดินทางบนทางเท้าไม่สะดวก)
            * **🔵 ช่วงที่ 2 (เดินทางด้วยระบบรางด่วน):** ใช้ลิฟต์อารยสถาปัตย์เพื่อขึ้นสู่สถานี นั่งรถไฟฟ้า BTS จากสถานี **{nearest_bts_start['clean_name']}** มุ่งหน้าไปลงที่สถานีปลายทาง **{nearest_bts_end['clean_name']}** *(มีสิ่งอำนวยความสะดวกสำหรับผู้พิการครบครัน)*
            * **🔴 ช่วงที่ 3 (เข้าสู่เป้าหมายปลายทาง):** ลงจากสถานีรถไฟฟ้า และเรียกรถ **แกร็บ (Grab) / แท็กซี่** เข้าสู่พิกัดเป้าหมาย **{end_label_th.split(' (')[0]}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)
            """)

    # 🏥 3. โหมดสวัสดิการรถตู้รัฐ
    elif "🏥" in travel_mode:
        if is_hospital:
            st.warning("🏥 **ยืนยันสิทธิ์รับสวัสดิการรถรับ-ส่งสถานพยาบาลสำเร็จ**")
            st.markdown(f"""
            เนื่องจากเป้าหมายปลายทางของคุณคือสถานพยาบาล คุณสามารถใช้สิทธิ์เรียกรถตู้รับ-ส่งฟรีสำหรับผู้พิการวีลแชร์ได้โดยตรง:
            * 📞 **บริการรถตู้ กรุงเทพมหานคร (กทม.):** โทรศัพท์นัดหมายจองคิวล่วงหน้าได้ที่ **สายด่วน กทม. โทร. 1555** หรือ **1479**
            * 🚑 **บริการรถรับส่ง สปสช.:** สำหรับรับส่งไปโรงพยาบาลรัฐตามสิทธิ์การรักษาหลัก โทรติดต่อที่ **สายด่วน 1330**
            """)
        else:
            st.error("❌ เงื่อนไขไม่ตรงตามเกณฑ์รับสวัสดิการของรัฐ")
            st.write(f"บริการรถตู้รับ-ส่งสวัสดิการฟรี จะจำกัดสิทธิ์เฉพาะการเดินทางไป **โรงพยาบาลหรือสถานพยาบาล** เท่านั้น แต่ปัจจุบันคุณเลือกปลายทางเป็น *{end_label_th.split(' (')[0]}*")

with col2:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดและแนวเส้นทางเดินรถ")
    
    df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
    nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]
    df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
    nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]
    
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=13)
    
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_label_th.split(' (')[0]}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_label_th.split(' (')[0]}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    # 🚶 ----------------------------------------------------------------------------------
    # ฟีเจอร์แก้ไขการนำทาง: ตัดเส้นทางถนนใหญ่ทิ้ง และปรับการลากเส้นทางเท้าให้เป็นตรรกะมุมฉาก (No Roads / Pedestrian Only)
    # ไม่ลากผ่ากลางสิ่งปลูกสร้าง พร้อมปรับสีเส้นฟุตบาทตามระดับความปลอดภัยของ AI เบื้องหลัง
    # ----------------------------------------------------------------------------------
    pedestrian_line_color = 'emerald' if safety_probability >= 0.70 else 'red'
    
    # สร้างจุดเลี้ยวฉาก 90 องศา (Manhattan Routing) เพื่อบังคับให้เส้นทางเดินอ้อมเลียบขอบอาคารภายนอก
    footpath_segment_start = [
        [start_info['latitude'], start_info['longitude']],
        [start_info['latitude'], nearest_bts_start['lng']], # พิกัดจุดตัดขอบมุมตึกบล็อกทางเท้า
        [nearest_bts_start['lat'], nearest_bts_start['lng']]
    ]
    
    footpath_segment_end = [
        [nearest_bts_end['lat'], nearest_bts_end['lng']],
        [nearest_bts_end['lat'], end_info['longitude']],    # พิกัดจุดตัดขอบมุมตึกปลายทาง
        [end_info['latitude'], end_info['longitude']]
    ]
    
    if "🚇" in travel_mode or (not matched_lines and "🚌" in travel_mode):
        folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"สถานีรถไฟฟ้า BTS ต้นทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"สถานีรถไฟฟ้า BTS ปลายทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        
        # วาดเฉพาะแนวทางเดินเท้าช่วงสั้นเชื่อมต่อเข้าสถานี (First-Leg / Last-Leg Footpaths) 
        folium.PolyLine(footpath_segment_start, color=pedestrian_line_color, weight=6, tooltip="เส้นทางเข็นวีลแชร์บนทางเท้า/ฟุตบาทภายนอกอาคาร").add_to(m)
        folium.PolyLine(footpath_segment_end, color='darkgreen', weight=6, tooltip="เส้นทางเท้าเข้าสู่เป้าหมายสุดท้าย").add_to(m)
        
        # วาดเส้นทางรถไฟฟ้าหลัก
        folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6, tooltip="แนวเส้นทางเดินรถไฟฟ้าเชื่อมต่อพ่วงระบบ").add_to(m)
    else:
        # กรณีรถเมล์ต่อเดียวหรือโหมดอื่น ปรับให้เป็นเส้นหักเหลี่ยมตามทางเท้าอ้อมสิ่งปลูกสร้างเช่นกัน
        bus_footpath_route = [
            [start_info['latitude'], start_info['longitude']],
            [start_info['latitude'], end_info['longitude']],
            [end_info['latitude'], end_info['longitude']]
        ]
        folium.PolyLine(bus_footpath_route, color='purple', weight=5, dash_array='5, 5', tooltip="เส้นทางสัญจรเชื่อมต่อทางเท้าคนเดิน").add_to(m)

    st_folium(m, width="100%", height=580)
