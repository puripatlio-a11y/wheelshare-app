import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V4", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบวางแผนเลือกโหมดการเดินทางและเจาะลึกสถานีรับ-ส่งอัจฉริยะ")
st.write("---")

# ฟังก์ชันคำนวณระยะทางภูมิศาสตร์
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c * 1000

@st.cache_data
def load_and_prepare_data():
    df_places = pd.read_csv('bangkok_places.csv')
    df_stations = pd.read_csv('bts_station.csv')
    df_accessibility = pd.read_csv('BTS for wheelchair users spreadsheet - BTS green line.csv')
    df_bus_stops = pd.read_csv('bangkok_bus_stops_coordinates.csv')
    
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()
    df_accessibility['clean_name'] = df_accessibility['สถานี'].str.replace('สถานี', '').str.strip()
    
    df_bts_master = pd.merge(
        df_accessibility[['clean_name', 'สถานี', 'มีลิฟต์', 'ทางลาดสำหรับรถเข็น']], 
        df_stations[['clean_name', 'lat', 'lng']], 
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)
    
    return df_places, df_bts_master, df_bus_stops

try:
    df_places, df_bts_master, df_bus_stops = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ ไม่สามารถเปิดไฟล์ข้อมูลได้: {e}")
    st.stop()

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")
place_list = df_places['place_name'].tolist()
start_place_name = st.sidebar.selectbox("📍 เลือกจุดต้นทาง (Origin):", place_list, index=13 if len(place_list) > 13 else 0)
end_place_name = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง (Destination):", place_list, index=0)

st.sidebar.write("---")
st.sidebar.markdown("### 🚏 เลือกโหมดการเดินทางที่ต้องการ")
# 🔥 ออปชันให้ผู้พิการเลือกโหมดการเดินทางหลักตามโจทย์
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
is_hospital = "hospital" in end_place_name.lower() or "โรงพยาบาล" in end_place_name

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### 📊 แผนผังการนำทางโหมด: {travel_mode.split(' - ')[0]}")
    st.write(f"**📍 จาก:** {start_place_name} ➡️ **🏁 ถึง:** {end_place_name}")
    st.write("---")

    # 🎯 OPTION 1: เดินทางด้วย BTS
    if "🚇" in travel_mode:
        df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

        df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
        nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

        transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_start['dist_start'] <= 100 else "🚖 เรียก Taxi / Grab Assistance"
        transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้า" if nearest_bts_end['dist_end'] <= 100 else "🚖 เรียก Taxi / Grab Assistance"

        st.info(f"**🟢 ขั้นที่ 1 (เข้าสู่ระบบ):** {transport_first_leg} จากจุดเริ่มไปยัง **สถานี BTS {nearest_bts_start['clean_name']}** (ระยะทาง {nearest_bts_start['dist_start']:.1f} เมตร)")
        st.write(f"ℹ️ *ตรวจสอบสิ่งอำนวยความสะดวก: มีลิฟต์โดยสาร = {nearest_bts_start['มีลิฟต์']}, มีทางลาด = {nearest_bts_start['ทางลาดสำหรับรถเข็น']}*")
        
        if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
            st.info(f"**🔵 ขั้นที่ 2 (ระบบราง):** ขึ้นรถไฟฟ้าจากสถานี **{nearest_bts_start['clean_name']}** เดินทางมุ่งหน้าสู่สถานีปลายทาง **{nearest_bts_end['clean_name']}**")
        
        st.info(f"**🔴 ขั้นที่ 3 (เข้าสู่จุดหมาย):** {transport_last_leg} จากสถานี BTS {nearest_bts_end['clean_name']} ตรงเข้าสู่เป้าหมาย **{end_place_name}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)")

    # 🎯 OPTION 2: เดินทางด้วยรถเมล์ชานต่ำ (ดึงข้อมูลจาก Bus stop แกะสเต็ปละเอียด)
    elif "🚌" in travel_mode:
        # หาลป้ายรถเมล์ที่ใกล้จุดต้นทางที่สุด
        df_bus_stops['dist_to_origin'] = [haversine_distance(start_info['latitude'], start_info['longitude'], row['latitude'], row['longitude']) for idx, row in df_bus_stops.iterrows()]
        nearest_bus_start = df_bus_stops.sort_values(by='dist_to_origin').iloc[0]
        matched_bus_line = nearest_bus_start['place_name'] # ตัวแปรชื่อสายรถเมล์

        # หาลป้ายรถเมล์ที่ใกล้จุดปลายทางที่สุด (ดึงคำสั่งลงจากไฟล์ตามโจทย์)
        df_bus_stops['dist_to_destination'] = [haversine_distance(end_info['latitude'], end_info['longitude'], row['latitude'], row['longitude']) for idx, row in df_bus_stops.iterrows()]
        nearest_bus_end = df_bus_stops.sort_values(by='dist_to_destination').iloc[0]

        st.success(f"🚌 **แนะนำให้ใช้บริการ: รถเมล์ไทยสมายล์บัส สาย {matched_bus_line}**")
        st.markdown(f"""
        **📋 ขั้นตอนการเดินทางที่คุณต้องทำ:**
        1. **🚶 การขึ้นรถ:** เข็นวีลแชร์เดินเท้าจากจุดเริ่มต้นไปยัง **ป้ายรถเมล์ที่ใกล้ที่สุด** (ระยะทางประมาณ {nearest_bus_start['dist_to_origin']:.1f} เมตร)
        2. **💳 รอขึ้นรถ:** สังเกตรถเมล์สาย **{matched_bus_line}** เมื่อรถจอด แจ้งพนักงานเพื่อขอกางทางลาดไฮโดรลิกเพื่อเข็นรถขึ้นสู่พื้นที่ล็อกวีลแชร์ภายในตัวรถ (ค่าโดยสารคงที่ 20 บาท)
        3. **🛑 การลงรถ:** ให้แจ้งพนักงานขับรถล่วงหน้าว่าจะขอลงที่ **ป้ายจอดรถเมล์เป้าหมายใกล้ปลายทาง**
        4. **🏁 เข้าสู่จุดหมาย:** เมื่อลงจากรถเมล์แล้ว ให้เข็นวีลแชร์ต่อจากป้ายจอดดังกล่าวตรงไปยัง **{end_place_name}** (ระยะทางเดินเท้าช่วงสุดท้ายประมาณ {nearest_bus_match['dist_to_origin']:.1f} เมตร)
        """)

    # 🎯 OPTION 3: สวัสดิการรถตู้จากรัฐ
    elif "🏥" in travel_mode:
        if is_hospital:
            st.warning("🏥 **ยืนยันสิทธิ์สวัสดิการรับ-ส่งสถานพยาบาลสำเร็จ**")
            st.markdown("""
            เนื่องจากปลายทางของคุณคือโรงพยาบาล คุณสามารถใช้บริการขนส่งพิเศษเพื่อสังคมได้โดยไม่ต้องเสียค่าใช้จ่าย:
            
            * 📞 **บริการรถตู้รับ-ส่งวีลแชร์ กทม.:** ออกแบบมาเพื่อผู้ใช้วีลแชร์โดยเฉพาะ ติดต่อและนัดหมายเวลาเดินรถล่วงหน้าได้ที่ **สายด่วน กทม. โทร. 1555** หรือ **สายด่วนคนพิการ โทร. 1479**
            * 🚑 **บริการรถรับส่ง สปสช.:** สำหรับผู้ใช้วีลแชร์ที่ต้องการเดินทางไปโรงพยาบาลรัฐตามสิทธิ์การรักษาหลักประกันสุขภาพ ตรวจสอบคิวรถได้ที่ **สายด่วน สปสช. โทร. 1330**
            
            *หมายเหตุ: โปรดโทรนัดหมายล่วงหน้าอย่างน้อย 24-48 ชั่วโมงก่อนถึงเวลาหมอนัด*
            """)
        else:
            st.error("❌ เงื่อนไขไม่ตรงตามเกณฑ์สวัสดิการ")
            st.write("ขออภัยครับ สวัสดิการรถตู้รับ-ส่งฟรีจากรัฐและ สปสช. จะจำกัดสิทธิ์เฉพาะการเดินทางที่มีปลายทางเป็น **สถานพยาบาลหรือโรงพยาบาล** เท่านั้นครับ โปรดเปลี่ยนจุดปลายทางเป็นโรงพยาบาล หรือเลือกโหมดการเดินทางอื่นแทนด้านซ้ายมือครับ")

with col2:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดและแนวเส้นทางเดินรถ")
    
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=13)
    
    # หมุดหลัก
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_place_name}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_place_name}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    # แสดงเส้นและหมุดย่อยตามโหมดที่เลือกเพื่อไม่ให้แผนที่รก
    if "🚇" in travel_mode:
        folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS ต้นทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        
        folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6).add_to(m)
        color_leg1 = 'red' if nearest_bts_start['dist_start'] > 100 else 'orange'
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [nearest_bts_start['lat'], nearest_bts_start['lng']]], color=color_leg1, weight=4).add_to(m)

    elif "🚌" in travel_mode:
        # ปักหมุดป้ายขึ้นและป้ายลงรถเมล์ที่ AI แอบสแกนหาเจอในไฟล์
        folium.Marker([nearest_bus_start['latitude'], nearest_bus_start['longitude']], popup=f"ป้ายขึ้นรถเมล์สาย {matched_bus_line}", icon=folium.Icon(color='purple', icon='arrow-up', prefix='fa')).add_to(m)
        folium.Marker([nearest_bus_end['latitude'], nearest_bus_end['longitude']], popup=f"ป้ายลงรถเมล์สาย {matched_bus_line}", icon=folium.Icon(color='purple', icon='arrow-down', prefix='fa')).add_to(m)
        
        # ลากเส้นแนววิ่งรถเมล์สีม่วง
        folium.PolyLine([[nearest_bus_start['latitude'], nearest_bus_start['longitude']], [nearest_bus_end['latitude'], nearest_bus_end['longitude']]], color='purple', weight=5, dash_array='10, 10').add_to(m)

    elif "🏥" in travel_mode and is_hospital:
        # ถ้าเป็นรถตู้สวัสดิการ จะลากเป็นเส้นตรงยาวบริการพิเศษ (เส้นทึบสีทอง/เหลือง)
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='cadetblue', weight=6, popup="เส้นทางบริการรถตู้สวัสดิการรัฐ").add_to(m)

    st_folium(m, width="100%", height=580)
