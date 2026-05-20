import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V5.6", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบวางแผนเส้นทางอัจฉริยะ (เวอร์ชันแลนด์มาร์กหลัก ค้นหาสายรถเมล์ร่วมอัตโนมัติ)")
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
    # 🎯 ใช้ไฟล์เจาะจงที่คุณระบุ ซึ่งเป็นไฟล์แลนด์มาร์กล้วน ไม่มีชื่อป้ายรถเมล์ย่อยผสม
    df_places = pd.read_csv('bangkok_places_bus_spot.csv')
    df_stations = pd.read_csv('bts_station.csv')
    df_accessibility = pd.read_csv('BTS for wheelchair users spreadsheet - BTS green line.csv')
    
    # ระบบค้นหาและสแกนชื่อไฟล์ตารางเดินรถเมล์อัตโนมัติ ป้องกันปัญหาเว้นวรรคสเปซไม่ตรงกันแล้วแอปพัง
    target_bus_file = 'ThaiSmalieBus  - Sheet1.csv' 
    if not os.path.exists(target_bus_file):
        all_files = os.listdir('.')
        matched_files = [f for f in all_files if 'ThaiSmalieBus' in f or 'ThaiSmileBus' in f]
        if matched_files:
            target_bus_file = matched_files[0]
            
    df_bus_routes = pd.read_csv(target_bus_file)
    
    # จัดการความสะอาดและเชื่อมข้อมูล BTS ลิฟต์วีลแชร์
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
    st.info("💡 คำแนะนำวิศวกร: โปรดตรวจสอบว่าไฟล์ตารางเดินรถเมล์ (ThaiSmalieBus) และไฟล์จุดพิกัด ได้ใส่ไว้ในโฟลเดอร์โปรเจกต์ครบถ้วนแล้ว")
    st.stop()

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")
place_list = sorted(df_places['place_name'].tolist())

# ตั้งค่าจุดเริ่มต้นเริ่มต้น (Defaults) ให้พร้อมใช้และแสดงผลทันที
default_start = place_list.index("Victory Monument") if "Victory Monument" in place_list else 0
hospital_options = [x for x in place_list if any(h in x.lower() for h in ["โรงพยาบาล", "รพ.", "hospital"])]
default_end = place_list.index(hospital_options[0]) if hospital_options else 0

# 🌟 ผลลัพธ์: Dropdown ตรงนี้จะคลีนมาก มีแต่สถานที่เด่นๆ ไม่มีชื่อป้ายรถเมล์ย่อยมากวนใจ
start_place_name = st.sidebar.selectbox("📍 เลือกจุดต้นทาง (Origin):", place_list, index=default_start)
end_place_name = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง (Destination):", place_list, index=default_end)

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

start_info = df_places[df_places['place_name'] == start_place_name].iloc[0]
end_info = df_places[df_places['place_name'] == end_place_name].iloc[0]
is_hospital = any(keyword in end_place_name.lower() for keyword in ["hospital", "โรงพยาบาล", "รพ."])

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### 📊 แผนผังการนำทางโหมด: {travel_mode.split(' - ')[0]}")
    st.write(f"**📍 จาก:** {start_place_name} ➡️ **🏁 ถึง:** {end_place_name}")
    st.write("---")

    # 🎯 OPTION 1: รถไฟฟ้า BTS
    if "🚇" in travel_mode:
        df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

        df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

        transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_start['dist_start'] <= 100 else "🚖 เรียก Taxi / Grab Assistance"
        transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_end['dist_end'] <= 100 else "🚖 เรียก Taxi / Grab Assistance"

        st.info(f"**🟢 ขั้นที่ 1:** {transport_first_leg} ไปยัง **สถานี BTS {nearest_bts_start['clean_name']}** (ระยะทาง {nearest_bts_start['dist_start']:.1f} เมตร)")
        st.write(f"ℹ️ *สิ่งอำนวยความสะดวก: มีลิฟต์ = {nearest_bts_start['มีลิฟต์']}, มีทางลาด = {nearest_bts_start['ทางลาดสำหรับรถเข็น']}*")
        
        if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2:** ขึ้นรถไฟฟ้าจากสถานี **{nearest_bts_start['clean_name']}** ไปลงที่สถานี **{nearest_bts_end['clean_name']}**")
        
        st.info(f"**🔴 ขั้นที่ 3:** {transport_last_leg} จากสถานีปลายทางเข้าสู่เป้าหมาย **{end_place_name}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)")

    # 🎯 OPTION 2: รถเมล์ชานต่ำ (คำนวณและแนะนำทุกสายที่ผ่านร่วมกันอัตโนมัติ)
    elif "🚌" in travel_mode:
        st.markdown("#### 🚏 สายรถเมล์ไทยสมายล์บัสที่ผ่านเส้นทางของคุณ")
        
        # ตัวแปลคีย์เวิร์ดเพื่อคุยกับไฟล์ตารางรถเมล์ภาษาไทยได้แม่นยำขึ้น
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
        
        # ค้นหาหลังบ้านว่ามีสายไหนวิ่งผ่านทั้งสองจุดร่วมกันบ้าง
        matched_lines = []
        for idx, row in df_bus_routes.iterrows():
            route_text = str(row['ต้นทาง']) + str(row['ปลาย']) + str(row['ผ่าน'])
            
            match_start = any(k.lower() in route_text.lower() for k in start_keywords)
            match_end = any(k.lower() in route_text.lower() for k in end_keywords)
            
            # ดักจับเงื่อนไข: ถ้าสายรถเมล์นี้วิ่งผ่านร่วมกันทั้งสองสถานที่ ให้เก็บสายนี้ไว้
            if match_start and match_end:
                matched_lines.append(row['สาย'])

        if matched_lines:
            # ลบสายซ้ำ จัดหมวดหมู่ และเชื่อมคำว่า " หรือ " เพื่อแนะนำทุกสายร่วมกันตามสั่ง
            unique_lines_list = sorted(list(set(matched_lines)))
            all_suggested_lines = " หรือ ".join(unique_lines_list)
            first_bus = unique_lines_list[0]
            
            st.success(f"✅ **AI ตรวจพบสายรถเมล์ชานต่ำที่ผ่านร่วมกัน: สาย {all_suggested_lines}**")
            st.markdown(f"""
            **📋 แนะนำขั้นตอนเดินทางสำหรับผู้ใช้วีลแชร์:**
            1. **🚶 จุดขึ้นรถ:** เข็นวีลแชร์ไปยังจุดจอดรถเมล์ ณ **{start_place_name}**
            2. **💳 เลือกขึ้นรถเมล์:** คุณสามารถรอขึ้นรถเมล์สาย **{all_suggested_lines}** สายใดก็ได้ที่มาถึงก่อน (เนื่องจากเป็นรถของ Thai Smile Bus ชานต่ำ มีแรมป์ไฮโดรลิกพับเก็บ และมีเบรก/สายล็อกสำหรับวีลแชร์ปลอดภัย 100%)
            3. **🏁 จุดลงรถ:** นั่งไปลงที่จุดจอดเป้าหมาย ณ **{end_place_name}** พนักงานจะช่วยอำนวยความสะดวกในการพับแรมป์ให้ลงอย่างปลอดภัยครับ
            """)
        else:
            st.warning("ℹ️ ไม่พบสายรถเมล์ตรงในระบบที่เชื่อมระหว่างสองจุดนี้โดยตรง")
            st.write("ระบบแนะนำให้เปลี่ยนไปใช้งานโหมด **รถไฟฟ้า (BTS)** เพื่อความสะดวกสบายและความรวดเร็วในการใช้อารยสถาปัตย์ (ลิฟต์โดยสาร) ครับ")

    # 🎯 OPTION 3: สวัสดิการรถตู้จากรัฐ
    elif "🏥" in travel_mode:
        if is_hospital:
            st.warning("🏥 **ยืนยันสิทธิ์สวัสดิการรับ-ส่งสถานพยาบาลสำเร็จ**")
            st.markdown(f"""
            เนื่องจากปลายทางของคุณคือ **{end_place_name}** คุณสามารถใช้สิทธิ์เรียกรถรับ-ส่งฟรีสำหรับผู้พิการได้:
            * 📞 **บริการรถตู้ กทม. (วีลแชร์):** โทรนัดหมายล่วงหน้าได้ที่ **สายด่วน กทม. โทร. 1555** หรือ **1479**
            * 🚑 **บริการรถรับส่ง สปสช.:** สำหรับไปโรงพยาบาลรัฐตามสิทธิ์ โทรติดต่อที่ **สายด่วน 1330**
            """)
        else:
            st.error("❌ เงื่อนไขไม่ตรงตามเกณฑ์สวัสดิการ")
            st.write(f"สวัสดิการรถตู้รับ-ส่งฟรี จะจำกัดสิทธิ์เฉพาะการเดินทางไป **โรงพยาบาล** เท่านั้น แต่ปัจจุบันคุณเลือกปลายทางเป็น *{end_place_name}*")

with col2:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดและแนวเส้นทางเดินรถ")
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=12)
    
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_place_name}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_place_name}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    if "🚇" in travel_mode:
        folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS ต้นทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6).add_to(m)

    elif "🚌" in travel_mode:
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='purple', weight=5, dash_array='5, 5').add_to(m)

    elif "🏥" in travel_mode and is_hospital:
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='cadetblue', weight=6).add_to(m)

    st_folium(m, width="100%", height=580)
