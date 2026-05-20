import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V4.4", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบวางแผนเส้นทางอัจฉริยะ (เวอร์ชันรวมฐานข้อมูลสถานที่ โรงพยาบาล และป้ายรถเมล์)")
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
    # 1. โหลดไฟล์ข้อมูลทั้งหมดที่มีในระบบ
    df_places_raw = pd.read_csv('bangkok_places.csv')
    df_stations = pd.read_csv('bts_station.csv')
    df_accessibility = pd.read_csv('BTS for wheelchair users spreadsheet - BTS green line.csv')
    df_bus_stops = pd.read_csv('bangkok_bus_stops_coordinates.csv')
    
    # 2. จัดการทำความสะอาดข้อมูลสถานี BTS และจับคู่สิทธิ์สิ่งอำนวยความสะดวก
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()
    df_accessibility['clean_name'] = df_accessibility['สถานี'].str.replace('สถานี', '').str.strip()
    
    df_bts_master = pd.merge(
        df_accessibility[['clean_name', 'สถานี', 'มีลิฟต์', 'ทางลาดสำหรับรถเข็น']], 
        df_stations[['clean_name', 'lat', 'lng']], 
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)
    
    # 3. 🎯 [ลอจิกควบรวมสถานที่ตามสั่ง] นำสถานที่และโรงพยาบาลจากไฟล์ Bus Stop มารวมเข้ากับสถานที่หลัก
    df_bus_places = df_bus_stops[['place_name', 'latitude', 'longitude']].copy()
    
    # รวมแถวเข้าด้วยกัน (Concatenate) และทำการลบชื่อสถานที่ที่ซ้ำซ้อนออก (Drop Duplicates)
    df_combined_places = pd.concat([df_places_raw, df_bus_places], ignore_index=True)
    df_combined_places = df_combined_places.drop_duplicates(subset=['place_name']).reset_index(drop=True)
    
    return df_combined_places, df_bts_master, df_bus_stops

try:
    df_places, df_bts_master, df_bus_stops = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ ไม่สามารถเปิดไฟล์ข้อมูลได้: {e}")
    st.stop()

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")

# ดึงรายชื่อสถานที่ที่ควบรวมสำเร็จและเรียงตามตัวอักษรเพื่อความง่ายในการค้นหา
place_list = sorted(df_places['place_name'].tolist())

# ค้นหาดัชนีเริ่มต้นเพื่อให้หน้าเว็บเปิดมาดูสวยงามน่าสนใจ (เช่น เริ่มจาก อนุสาวรีย์ชัยฯ)
default_start = place_list.index("Victory Monument") if "Victory Monument" in place_list else 0

# วนลูปหาชื่อโรงพยาบาลเพื่อตั้งเป็นปลายทางตั้งต้นอัตโนมัติ (ช่วยโชว์ความเทพของเมนูสวัสดิการรัฐทันทีที่เปิดแอป)
hospital_options = [x for x in place_list if any(h in x.lower() for h in ["โรงพยาบาล", "รพ.", "hospital"])]
default_end = place_list.index(hospital_options[0]) if hospital_options else 0

start_place_name = st.sidebar.selectbox("📍 เลือกจุดต้นทาง (Origin):", place_list, index=default_start)
end_place_name = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง (Destination):", place_list, index=default_end)

# 📥 เพิ่มปุ่มดาวน์โหลดไฟล์ฐานข้อมูลที่รวมเสร็จแล้วบน Sidebar
st.sidebar.write("---")
st.sidebar.markdown("### 📥 ดาวน์โหลดฐานข้อมูลระบบ")
csv_buffer = df_places.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📁 ดาวน์โหลดไฟล์รวม bangkok_places.csv",
    data=csv_buffer,
    file_name="bangkok_places.csv",
    mime="text/csv",
    help="คลิกเพื่อดาวน์โหลดฐานข้อมูลที่รวมเอาพิกัดป้ายรถเมล์และโรงพยาบาลทั้งหมดเข้าไว้ด้วยกัน"
)

st.sidebar.write("---")
st.sidebar.markdown("### 🚏 เลือกโหมดการเดินทางที่ต้องการ")
travel_mode = st.sidebar.radio(
    "โปรดเลือกรูปแบบการเดินทางหลักที่สะดวก:",
    [
        "🚇 รถไฟฟ้า (BTS) - เน้นเดินทางเร็ว",
        "🚌 รถเมล์ชานต่ำ (Thai Smile Bus) - เน้นประหยัด",
        "🏥 สวัสดิการรถตู้รัฐ/กทม. ฟรี (สำหรับไปโรงพยาบาล)"
    ]
)

# ดึงข้อมูลละติจูดและลองจิจูดของสถานที่ที่ถูกเลือก
start_info = df_places[df_places['place_name'] == start_place_name].iloc[0]
end_info = df_places[df_places['place_name'] == end_place_name].iloc[0]

# ฟังก์ชันตรวจสอบสิทธิ์ปลายทางสำหรับการใช้งานรถสวัสดิการโรงพยาบาล
is_hospital = any(keyword in end_place_name.lower() for keyword in ["hospital", "โรงพยาบาล", "รพ."])

# ─── การแสดงผลเนื้อหา (Main Content Area) ───
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

    # 🎯 OPTION 2: เดินทางด้วยรถเมล์ชานต่ำ (Thai Smile Bus)
    elif "🚌" in travel_mode:
        df_bus_stops['dist_to_origin'] = [haversine_distance(start_info['latitude'], start_info['longitude'], row['latitude'], row['longitude']) for idx, row in df_bus_stops.iterrows()]
        nearest_bus_start = df_bus_stops.sort_values(by='dist_to_origin').iloc[0]
        matched_bus_line = nearest_bus_start['place_name'] 

        df_bus_stops['dist_to_destination'] = [haversine_distance(end_info['latitude'], end_info['longitude'], row['latitude'], row['longitude']) for idx, row in df_bus_stops.iterrows()]
        nearest_bus_end = df_bus_stops.sort_values(by='dist_to_destination').iloc[0]

        st.success(f"🚌 **แนะนำให้ใช้บริการ: รถเมล์ไทยสมายล์บัส สาย {matched_bus_line}**")
        st.markdown(f"""
        **📋 ขั้นตอนการเดินทางที่คุณต้องทำ:**
        1. **🚶 การขึ้นรถ:** เข็นวีลแชร์เดินเท้าจากจุดเริ่มต้นไปยัง **ป้ายรถเมล์ขาขึ้นที่ใกล้ที่สุด** (ระยะทางประมาณ {nearest_bus_start['dist_to_origin']:.1f} เมตร)
        2. **💳 รอขึ้นรถ:** สังเกตรถเมล์สาย **{matched_bus_line}** เมื่อรถจอด แจ้งพนักงานเพื่อขอกางทางลาดไฮโดรลิกเพื่อเข็นรถขึ้นสู่พื้นที่ล็อกวีลแชร์ภายในตัวรถ (ค่าโดยสารคงที่ 20 บาท)
        3. **🛑 การลงรถ:** นั่งยาวไปลงที่ **ป้ายจอดรถเมล์ขาลงใกล้ปลายทางมากที่สุด**
        4. **🏁 เข้าสู่จุดหมาย:** เมื่อลงจากรถเมล์แล้ว ให้เข็นวีลแชร์ต่อจากป้ายจอดดังกล่าวตรงไปยังเป้าหมาย **{end_place_name}** (ระยะทางเดินเท้าช่วงสุดท้ายประมาณ {nearest_bus_end['dist_to_destination']:.1f} เมตร)
        """)

    # 🎯 OPTION 3: สวัสดิการรถตู้จากรัฐ
    elif "🏥" in travel_mode:
        if is_hospital:
            st.warning("🏥 **ยืนยันสิทธิ์สวัสดิการรับ-ส่งสถานพยาบาลสำเร็จ**")
            st.markdown(f"""
            เนื่องจากระบบตรวจพบว่าปลายทางของคุณคือ **{end_place_name}** คุณสามารถใช้บริการขนส่งพิเศษเพื่อสังคมได้โดยไม่ต้องเสียค่าใช้จ่าย:
            
            * 📞 **บริการรถตู้รับ-ส่งวีลแชร์ กทม.:** ออกแบบมาเพื่อผู้ใช้วีลแชร์โดยเฉพาะ ติดต่อและนัดหมายเวลาเดินรถล่วงหน้าได้ที่ **สายด่วน กทม. โทร. 1555** หรือ **สายด่วนคนพิการ โทร. 1479**
            * 🚑 **บริการรถรับส่ง สปสช.:** สำหรับผู้ใช้วีลแชร์ที่ต้องการเดินทางไปโรงพยาบาลรัฐตามสิทธิ์การรักษาหลักประกันสุขภาพ ตรวจสอบคิวรถได้ที่ **สายด่วน สปสช. โทร. 1330**
            
            *หมายเหตุ: โปรดโทรนัดหมายล่วงหน้าอย่างน้อย 24-48 ชั่วโมงก่อนถึงเวลาหมอนัด*
            """)
        else:
            st.error("❌ เงื่อนไขไม่ตรงตามเกณฑ์สวัสดิการ")
            st.write(f"ขออภัยครับ สวัสดิการรถตู้รับ-ส่งฟรีจากรัฐและ สปสช. จะจำกัดสิทธิ์เฉพาะการเดินทางที่มีปลายทางเป็น **สถานพยาบาลหรือโรงพยาบาล** เท่านั้นครับ แต่ปัจจุบันคุณเลือกจุดหมายเป็น *{end_place_name}* โปรดเปลี่ยนจุดปลายทางเป็นโรงพยาบาล หรือเลือกโหมดการเดินทางอื่นแทนด้านซ้ายมือครับ")

with col2:
    st.markdown("### 🗺️ แผนที่ระบุพิกัดและแนวเส้นทางเดินรถ")
    
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=12)
    
    # ปักหมุดพิกัดหลัก
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_place_name}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_place_name}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    # แสดงเส้นนำทางแบบจำลองพิกัดจริงตามแต่ละโหมด
    if "🚇" in travel_mode:
        folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS ต้นทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
        
        folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6).add_to(m)
        color_leg1 = 'red' if nearest_bts_start['dist_start'] > 100 else 'orange'
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [nearest_bts_start['lat'], nearest_bts_start['lng']]], color=color_leg1, weight=4).add_to(m)

    elif "🚌" in travel_mode:
        folium.Marker([nearest_bus_start['latitude'], nearest_bus_start['longitude']], popup=f"ป้ายขึ้นรถเมล์สาย {matched_bus_line}", icon=folium.Icon(color='purple', icon='arrow-up', prefix='fa')).add_to(m)
        folium.Marker([nearest_bus_end['latitude'], nearest_bus_end['longitude']], popup=f"ป้ายลงรถเมล์ที่ใกล้จุดหมายที่สุด", icon=folium.Icon(color='purple', icon='arrow-down', prefix='fa')).add_to(m)
        folium.PolyLine([[nearest_bus_start['latitude'], nearest_bus_start['longitude']], [nearest_bus_end['latitude'], nearest_bus_end['longitude']]], color='purple', weight=5, dash_array='10, 10').add_to(m)

    elif "🏥" in travel_mode and is_hospital:
        folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='cadetblue', weight=6, popup="เส้นทางบริการรถตู้สวัสดิการรัฐ").add_to(m)

    st_folium(m, width="100%", height=580)
