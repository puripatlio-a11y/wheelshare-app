import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V5.7", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบวางแผนเส้นทางอัจฉริยะ (เวอร์ชันพ่วงต่อ รถเมล์ + BTS + Grab สำหรับวีลแชร์)")
st.write("---")

def haversine_distance(lat1, import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V5.8", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบวางแผนเส้นทางอัจฉริยะ (เวอร์ชันพ่วงต่อ รถเมล์ + BTS + Grab สำหรับวีลแชร์)")
st.write("---")

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

# 🎯 พจนานุกรมแปลชื่อสถานที่จากภาษาอังกฤษในไฟล์ ให้เป็นภาษาไทยทั้งหมดบนหน้าเว็บ
th_name_map = {
    "Victory Monument": "📍 อนุสาวรีย์ชัยสมรภูมิ",
    "Siam Station": "🚇 สถานีรถไฟฟ้า Siam (สยาม)",
    "CentralWorld": "🛍️ เซ็นทรัลเวิลด์ (CentralWorld)",
    "MBK Center": "🛍️ เอ็มบีเค เซ็นเตอร์ (MBK Center)",
    "Samyan Mitrtown": "🏢 สามย่านมิตรทาวน์",
    "Chulalongkorn Hospital": "🏥 โรงพยาบาลจุฬาลงกรณ์",
    "Siriraj Hospital": "🏥 โรงพยาบาลศิริราช",
    "Ramathibodi Hospital": "🏥 โรงพยาบาลรามาธิบดี",
    "Rajavithi Hospital": "🏥 โรงพยาบาลราชวิถี",
    "Vajira Hospital": "🏥 โรงพยาบาลวชิรพยาบาล",
    "Mochit Bus Terminal": "🚌 สถานีขนส่งผู้โดยสารกรุงเทพ (หมอชิต 2)",
    "Chatuchak Park": "🌳 สวนจตุจักร",
    "Ari BTS Station": "🚇 สถานีรถไฟฟ้า Ari (อารีย์)",
    "Saphan Khwai BTS Station": "🚇 สถานีรถไฟฟ้า สะพานควาย",
    "Kasetsart University": "🎓 มหาวิทยาลัยเกษตรศาสตร์",
    "Bang Wa BTS Station": "🚇 สถานีรถไฟฟ้า บางหว้า",
    "Bearing BTS Station": "🚇 สถานีรถไฟฟ้า แบริ่ง",
    "Ekkamai Bus Terminal": "🚌 สถานีขนส่งเอกมัย"
}

df_places['display_name_th'] = df_places['place_name'].map(lambda x: th_name_map.get(x, x))
place_list_th = sorted(df_places['display_name_th'].tolist())

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")

default_start_idx = place_list_th.index("📍 อนุสาวรีย์ชัยสมรภูมิ") if "📍 อนุสาวรีย์ชัยสมรภูมิ" in place_list_th else 0
default_end_idx = place_list_th.index("🏥 โรงพยาบาลจุฬาลงกรณ์") if "🏥 โรงพยาบาลจุฬาลงกรณ์" in place_list_th else 0

start_label_th = st.sidebar.selectbox("📍 เลือกจุดต้นทาง:", place_list_th, index=default_start_idx)
end_label_th = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง:", place_list_th, index=default_end_idx)

start_info = df_places[df_places['display_name_th'] == start_label_th].iloc[0]
end_info = df_places[df_places['display_name_th'] == end_label_th].iloc[0]

start_place_name = start_info['place_name']
end_place_name = end_info['place_name']

st.sidebar.write("---")
st.sidebar.markdown("### 𚏏 เลือกโหมดการเดินทาง")
travel_mode = st.sidebar.radio(
    "โปรดเลือกรูปแบบการเดินทางหลักที่สะดวก:",
    [
        "🚇 รถไฟฟ้า (BTS) - เน้นเดินทางเร็ว",
        "🚌 รถเมล์ชานต่ำ (Thai Smile Bus) - เน้นประหยัด",
        "🏥 สวัสดิการรถตู้รัฐ/กทม. ฟรี (สำหรับไปโรงพยาบาล)"
    ]
)

is_hospital = any(keyword in end_place_name.lower() for keyword in ["hospital", "โรงพยาบาล", "รพ."])

# ⚡ [แก้ไขจุด NameError] ประกาศตัวแปร matched_lines ไว้ล่วงหน้าเป็นค่าเริ่มต้นที่นี่ เพื่อให้แผนผังด้านล่างเรียกใช้ได้โดยไม่พัง
matched_lines = []

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### 📊 แผนผังนำทางอัจฉริยะ")
    st.write(f"**จาก:** {start_label_th} ➡️ **ถึง:** {end_label_th}")
    st.write("---")

    # 🚇 1. โหมด BTS
    if "🚇" in travel_mode:
        df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

        df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

        transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_start['dist_start'] <= 150 else "🚖 แนะนำเรียก GrabInter / Taxi"
        transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_end['dist_end'] <= 150 else "𚖖 แนะนำเรียก GrabInter / Taxi"

        st.info(f"**🟢 ขั้นที่ 1:** {transport_first_leg} ไปยัง **สถานี BTS {nearest_bts_start['clean_name']}** (ระยะทาง {nearest_bts_start['dist_start']:.1f} เมตร)")
        st.write(f"ℹ️ *สิ่งอำนวยความสะดวก: มีลิฟต์ = {nearest_bts_start['มีลิฟต์']}, มีทางลาด = {nearest_bts_start['ทางลาดสำหรับรถเข็น']}*")
        
        if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2:** ขึ้นรถไฟฟ้าเดินทางจากสถานี **{nearest_bts_start['clean_name']}** ไปลงที่สถานี **{nearest_bts_end['clean_name']}**")
        
        st.info(f"**🔴 ขั้นที่ 3:** {transport_last_leg} จากสถานีปลายทางเข้าสู่เป้าหมาย **{end_label_th}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)")

    # 🚌 2. โหมดรถเมล์ชานต่ำ (ถ้าไม่มีรถเมล์สายตรง ระบบจะคำนวณแผนเชื่อมพ่วง BTS + Grab ให้ทันที)
    elif "🚌" in travel_mode:
        st.markdown("#### 🚏 ผลคำนวณการเดินรถโดยสารสาธารณะอารยสถาปัตย์")
        
        translation_dict = {
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
        
        start_keywords = translation_dict.get(start_place_name, [start_place_name])
        end_keywords = translation_dict.get(end_place_name, [end_place_name])
        
        for idx, row in df_bus_routes.iterrows():
            route_text = str(row['ต้นทาง']) + str(row['ปลาย']) + str(row['ผ่าน'])
            if any(k.lower() in route_text.lower() for k in start_keywords) and any(k.lower() in route_text.lower() for k in end_keywords):
                matched_lines.append(row['สาย'])

        # กรณีที่เจอสายรถเมล์ตรงวิ่งร่วมกัน
        if matched_lines:
            unique_lines_list = sorted(list(set(matched_lines)))
            all_suggested_lines = " หรือ ".join(unique_lines_list)
            
            st.success(f"✅ **AI แนะนำรถเมล์ชานต่ำต่อเดียวถึง: สาย {all_suggested_lines}**")
            st.markdown(f"""
            **📋 ขั้นตอนการเดินทาง:**
            1. **🚶 จุดขึ้นรถ:** เข็นวีลแชร์ไปยังจุดจอดรถประจำทาง ณ **{start_label_th}**
            2. **💳 การขึ้นรถ:** สามารถเลือกขึ้นรถสาย **{all_suggested_lines}** ตัวรถเป็นแบบชานต่ำ (Low-Floor) มีทางลาดไฮโดรลิก และพื้นที่ล็อกรถเข็นผู้พิการปลอดภัย
            3. **🏁 จุดหมาย:** นั่งยาวไปลงรถ ณ จุดจอดเป้าหมาย **{end_label_th}** ได้ทันที
            """)
            
        # กรณีไม่เจอสายรถเมล์วิ่งผ่านตรงๆ -> ปรับให้เป็นทริปพ่วงโหมดเชื่อมต่อให้วีลแชร์โดยอัตโนมัติ
        else:
            st.warning("🔄 ไม่พบสายรถเมล์ที่วิ่งผ่านตรงๆ ต่อเดียว --- AI ได้จัดแผนเดินทางพ่วงระบบเชื่อมต่อให้คุณอัตโนมัติ:")
            
            df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
            nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]
            
            df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
            nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]
            
            st.markdown(f"""
            **🗺️ แผนการเดินทางพ่วงเชื่อมต่ออัจฉริยะ (รถเมล์ + BTS + Grab):**
            * **🟢 ช่วงที่ 1 (พ่วงรถจ้าง/เข็น):** เดินทางจากจุดเริ่มต้นไปยัง **สถานี BTS {nearest_bts_start['clean_name']}** (ระยะทางประมาณ {nearest_bts_start['dist_start']:.1f} เมตร สามารถเรียกใช้บริการ **GrabAssist** หรือรถแท็กซี่ที่มีพื้นที่เก็บรถเข็นได้หากไกลเกินไป)
            * **🔵 ช่วงที่ 2 (พ่วงรถไฟฟ้าด่วน):** ใช้ลิฟต์อารยสถาปัตย์ขึ้นสถานีพ่วงต่อ เดินทางด้วยรถไฟฟ้า BTS จากสถานี **{nearest_bts_start['clean_name']}** มุ่งหน้าสู่สถานีปลายทาง **{nearest_bts_end['clean_name']}** *(สิ่งอำนวยความสะดวก: มีลิฟต์วีลแชร์บริการประจำสถานี)*
            * **🔴 ช่วงที่ 3 (เข้าสู่เป้าหมาย):** ลงจากสถานีรถไฟฟ้า และเรียกบริการรถ **Grab / Taxi** หรือเข็นรถวีลแชร์ต่อเนื่องเข้าสู่พิกัดเป้าหมาย **{end_label_th}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)
            """)

    # 🏥 3. โหมดสวัสดิการรถตู้รัฐ
    elif "🏥" in travel_mode:
        if is_hospital:
            st.warning("🏥 **ยืนยันสิทธิ์สวัสดิการรับ-ส่งสถานพยาบาลสำเร็จ**")
            st.markdown(f"""
            เนื่องจากปลายทางของคุณคือสถานพยาบาล คุณสามารถใช้สิทธิ์เรียกรถรับ-ส่งฟรีสำหรับผู้พิการได้:
            * 📞 **บริการรถตู้ กทม. (วีลแชร์):** โทรนัดหมายล่วงหน้าได้ที่ **สายด่วน กทม. โทร. 1555** หรือ **1479**
            * 🚑 **บริการรถรับส่ง สปสช.:** สำหรับไปโรงพยาบาลรัฐตามสิทธิ์ โทรติดต่อที่ **สายด่วน 1330**
            """)
        else:
            st.error("❌ เงื่อนไขไม่ตรงตามเกณฑ์สวัสดิการ")
            st.write(f"สวัสดิการรถตู้รับ-ส่งฟรี จะจำกัดสิทธิ์เฉพาะการเดินทางไป **โรงพยาบาล** เท่านั้น แต่ปัจจุบันคุณเลือกปลายทางเป็น *{end_label_th}*")

with col2:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดและแนวเส้นทางเดินรถ")
    
    # ดึงค่าพิกัดความปลอดภัยสำหรับแผนที่พ่วงต่อ
    df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
    nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]
    df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
    nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]
    
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=13)
    
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_label_th}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_label_th}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    # วาดแนวเส้นเชื่อมโยงบนแผนที่
    if "🚇" in travel_mode or (not matched_lines and "🚌" in travel_mode):
        folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS ต้นทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6, tooltip="แนวเส้นทางเดินรถไฟฟ้าพ่วงต่อ").add_to(m)
    else:
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='purple', weight=5, dash_array='5, 5').add_to(m)

    st_folium(m, width="100%", height=580), lat2, lon2):
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
    
    # ดึงไฟล์ตารางรถเมล์แบบปลอดภัย
    target_bus_file = 'ThaiSmalieBus  - Sheet1.csv' 
    if not os.path.exists(target_bus_file):
        all_files = os.listdir('.')
        matched_files = [f for f in all_files if 'ThaiSmalieBus' in f or 'ThaiSmileBus' in f]
        if matched_files:
            target_bus_file = matched_files[0]
            
    df_bus_routes = pd.read_csv(target_bus_file)
    
    # ล้างข้อมูลและจัดหมวดหมู่ BTS
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

# 🎯 พจนานุกรมแปลชื่อสถานที่จากอังกฤษในไฟล์ ให้กลายเป็นภาษาไทยบนหน้าเว็บ Dropdown
th_name_map = {
    "Victory Monument": "อนุสาวรีย์ชัยสมรภูมิ",
    "Siam Station": "สถานีรถไฟฟ้า Siam (สยาม)",
    "CentralWorld": "เซ็นทรัลเวิลด์ (CentralWorld)",
    "MBK Center": "เอ็มบีเค เซ็นเตอร์ (MBK Center)",
    "Samyan Mitrtown": "สามย่านมิตรทาวน์",
    "Chulalongkorn Hospital": "โรงพยาบาลจุฬาลงกรณ์",
    "Siriraj Hospital": "โรงพยาบาลศิริราช",
    "Ramathibodi Hospital": "โรงพยาบาลรามาธิบดี",
    "Rajavithi Hospital": "โรงพยาบาลราชวิถี",
    "Vajira Hospital": "โรงพยาบาลวชิรพยาบาล",
    "Mochit Bus Terminal": "สถานีขนส่งผู้โดยสารกรุงเทพ (หมอชิต 2)",
    "Chatuchak Park": "สวนจตุจักร",
    "Ari BTS Station": "สถานีรถไฟฟ้า Ari (อารีย์)",
    "Saphan Khwai BTS Station": "สถานีรถไฟฟ้า สะพานควาย",
    "Kasetsart University": "มหาวิทยาลัยเกษตรศาสตร์",
    "Bang Wa BTS Station": "สถานีรถไฟฟ้า บางหว้า",
    "Bearing BTS Station": "สถานีรถไฟฟ้า แบริ่ง",
    "Ekkamai Bus Terminal": "สถานีขนส่งเอกมัย"
}

# แปลงชื่อสถานที่ใน DataFrame เพื่อแสดงผลภาษาไทย
df_places['display_name_th'] = df_places['place_name'].map(lambda x: th_name_map.get(x, x))
place_list_th = sorted(df_places['display_name_th'].tolist())

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")

default_start_idx = place_list_th.index("📍 อนุสาวรีย์ชัยสมรภูมิ") if "📍 อนุสาวรีย์ชัยสมรภูมิ" in place_list_th else 0
default_end_idx = place_list_th.index("🏥 โรงพยาบาลจุฬาลงกรณ์") if "🏥 โรงพยาบาลจุฬาลงกรณ์" in place_list_th else 0

# หน้าตัวเลือก Dropdown จะกลายเป็นภาษาไทยอ่านง่ายทันที
start_label_th = st.sidebar.selectbox("📍 เลือกจุดต้นทาง:", place_list_th, index=default_start_idx)
end_label_th = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง:", place_list_th, index=default_end_idx)

# ดึงข้อมูลแถวแบบต้นฉบับกลับมาเพื่อคำนวณหลังบ้าน
start_info = df_places[df_places['display_name_th'] == start_label_th].iloc[0]
end_info = df_places[df_places['display_name_th'] == end_label_th].iloc[0]

start_place_name = start_info['place_name']
end_place_name = end_info['place_name']

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

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### 📊 แผนผังนำทางอัจฉริยะ")
    st.write(f"**จาก:** {start_label_th} ➡️ **ถึง:** {end_label_th}")
    st.write("---")

    # 🚇 1. โหมด BTS
    if "🚇" in travel_mode:
        df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

        df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

        transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_start['dist_start'] <= 150 else "🚖 แนะนำเรียก GrabAssist / Taxi"
        transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_end['dist_end'] <= 150 else "🚖 แนะนำเรียก GrabAssist / Taxi"

        st.info(f"**🟢 ขั้นที่ 1:** {transport_first_leg} ไปยัง **สถานี BTS {nearest_bts_start['clean_name']}** (ระยะทาง {nearest_bts_start['dist_start']:.1f} เมตร)")
        st.write(f"ℹ️ *สิ่งอำนวยความสะดวกสถานีต้นทาง: มีลิฟต์ = {nearest_bts_start['มีลิฟต์']}, มีทางลาด = {nearest_bts_start['ทางลาดสำหรับรถเข็น']}*")
        
        if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2:** ขึ้นรถไฟฟ้าเดินทางจากสถานี **{nearest_bts_start['clean_name']}** ไปลงที่สถานี **{nearest_bts_end['clean_name']}**")
        
        st.info(f"**🔴 ขั้นที่ 3:** {transport_last_leg} จากสถานีปลายทางเข้าสู่เป้าหมาย **{end_label_th}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)")

    # 🚌 2. โหมดรถเมล์ชานต่ำ (เวอร์ชันฉลาดล้ำ: ไม่มีรถเมล์ตรง -> สลับไปพ่วงต่อ BTS หรือระบบรถรับจ้างทันที)
    elif "🚌" in travel_mode:
        st.markdown("#### 🚏 ผลคำนวณการเดินรถโดยสารสาธารณะอารยสถาปัตย์")
        
        # คีย์เวิร์ดภาษาไทยเพื่อสแกนจับคู่กับตารางเดินรถเมล์
        translation_dict = {
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
        
        start_keywords = translation_dict.get(start_place_name, [start_place_name])
        end_keywords = translation_dict.get(end_place_name, [end_place_name])
        
        matched_lines = []
        for idx, row in df_bus_routes.iterrows():
            route_text = str(row['ต้นทาง']) + str(row['ปลาย']) + str(row['ผ่าน'])
            if any(k.lower() in route_text.lower() for k in start_keywords) and any(k.lower() in route_text.lower() for k in end_keywords):
                matched_lines.append(row['สาย'])

        # กรณีที่ 1: ตรวจพบว่ามีสายรถเมล์ชานต่ำวิ่งผ่านร่วมกันแบบต่อเดียวถึงตรงๆ
        if matched_lines:
            unique_lines_list = sorted(list(set(matched_lines)))
            all_suggested_lines = " หรือ ".join(unique_lines_list)
            
            st.success(f"✅ **AI แนะนำรถเมล์ชานต่ำต่อเดียวถึง: สาย {all_suggested_lines}**")
            st.markdown(f"""
            **📋 ขั้นตอนการเดินทาง:**
            1. **🚶 จุดขึ้นรถ:** เข็นวีลแชร์ไปยังจุดจอดรถประจำทาง ณ **{start_label_th}**
            2. **💳 การขึ้นรถ:** สามารถเลือกขึ้นรถสาย **{all_suggested_lines}** ตัวรถเป็นแบบชานต่ำ (Low-Floor) มีทางลาดไฮโดรลิก และพื้นที่ล็อกรถเข็นผู้พิการปลอดภัย
            3. **🏁 จุดหมาย:** นั่งยาวไปลงรถ ณ จุดจอดเป้าหมาย **{end_label_th}** ได้ทันที
            """)
            
        # กรณีที่ 2: ⚡ [แก้จุดปัญหาตามสั่ง] ไม่มีรถเมล์ตรง ให้ AI จัดทริปพ่วงโหมด BTS และ Taxi/Grab อัตโนมัติ!
        else:
            st.warning("🔄 ไม่พบสายรถเมล์ที่วิ่งผ่านตรงๆ ต่อเดียว --- AI ได้จัดแผนเดินทางพ่วงระบบเชื่อมต่อให้คุณอัตโนมัติ:")
            
            # ค้นหาสถานี BTS ที่อยู่ใกล้จุดเริ่มต้นและปลายทางเพื่อนำมาพ่วงประสานงาน
            df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
            nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]
            
            df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
            nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]
            
            st.markdown(f"""
            **🗺️ แผนการเดินทางพ่วงเชื่อมต่ออัจฉริยะ (รถเมล์ + BTS + Grab):**
            * **🟢 ช่วงที่ 1 (พ่วงรถจ้าง/เข็น):** เดินทางจากจุดเริ่มต้นไปยัง **สถานี BTS {nearest_bts_start['clean_name']}** (ระยะทางประมาณ {nearest_bts_start['dist_start']:.1f} เมตร สามารถเรียกใช้บริการ **GrabAssist** หรือรถแท็กซี่ที่มีพื้นที่เก็บรถเข็นได้หากไกลเกินไป)
            * **🔵 ช่วงที่ 2 (พ่วงรถไฟฟ้าด่วน):** ใช้ลิฟต์อารยสถาปัตย์ขึ้นสถานีพ่วงต่อ เดินทางด้วยรถไฟฟ้า BTS จากสถานี **{nearest_bts_start['clean_name']}** มุ่งหน้าสู่สถานีปลายทาง **{nearest_bts_end['clean_name']}** *(สิ่งอำนวยความสะดวก: มีลิฟต์วีลแชร์บริการประจำสถานี)*
            * **🔴 ช่วงที่ 3 (เข้าสู่เป้าหมาย):** ลงจากสถานีรถไฟฟ้า และเรียกบริการรถ **Grab / Taxi** หรือเข็นรถวีลแชร์ต่อเนื่องเข้าสู่พิกัดเป้าหมาย **{end_label_th}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)
            """)

    # 🏥 3. โหมดสวัสดิการรถตู้รัฐ
    elif "🏥" in travel_mode:
        if is_hospital:
            st.warning("🏥 **ยืนยันสิทธิ์สวัสดิการรับ-ส่งสถานพยาบาลสำเร็จ**")
            st.markdown(f"""
            เนื่องจากปลายทางของคุณคือสถานพยาบาล คุณสามารถใช้สิทธิ์เรียกรถรับ-ส่งฟรีสำหรับผู้พิการได้:
            * 📞 **บริการรถตู้ กทม. (วีลแชร์):** โทรนัดหมายล่วงหน้าได้ที่ **สายด่วน กทม. โทร. 1555** หรือ **1479**
            * 🚑 **บริการรถรับส่ง สปสช.:** สำหรับไปโรงพยาบาลรัฐตามสิทธิ์ โทรติดต่อที่ **สายด่วน 1330**
            """)
        else:
            st.error("❌ เงื่อนไขไม่ตรงตามเกณฑ์สวัสดิการ")
            st.write(f"สวัสดิการรถตู้รับ-ส่งฟรี จะจำกัดสิทธิ์เฉพาะการเดินทางไป **โรงพยาบาล** เท่านั้น แต่ปัจจุบันคุณเลือกปลายทางเป็น *{end_label_th}*")

with col2:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดและแนวเส้นทางเดินรถ")
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=12)
    
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_label_th}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_label_th}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    # วาดแนวเส้นเชื่อมต่อโครงข่ายบนแผนที่ตามโหมดทริป
    if "🚇" in travel_mode or (not matched_lines and "🚌" in travel_mode):
        folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS ต้นทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6, tooltip="เส้นทางเชื่อมรถไฟฟ้าพ่วงต่อ").add_to(m)
    else:
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='purple', weight=5, dash_array='5, 5').add_to(m)

    st_folium(m, width="100%", height=580)
