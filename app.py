import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V3", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบคำนวณและวางแผนเส้นทางเชื่อมต่ออัจฉริยะ พร้อมระบบแนะนำสวัสดิการรัฐ")
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
    st.info("💡 โปรดตรวจสอบว่าไฟล์ .csv ทั้งหมด (รวมถึง bangkok_bus_stops_coordinates.csv) อัปโหลดเรียบร้อยแล้ว")
    st.stop()

st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")
place_list = df_places['place_name'].tolist()
start_place_name = st.sidebar.selectbox("📍 เลือกจุดต้นทาง (Origin):", place_list, index=13 if len(place_list) > 13 else 0)
end_place_name = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง (Destination):", place_list, index=0)

st.sidebar.write("---")
st.sidebar.markdown("#### ⚙️ ตัวกรองสถานการณ์และความต้องการ")
is_urgent = st.sidebar.checkbox("🚨 รีบด่วน (Urgent Journey)", value=True)
prefer_cheap = st.sidebar.checkbox("💰 เน้นประหยัดค่าใช้จ่าย (Budget Friendly)", value=False)

start_info = df_places[df_places['place_name'] == start_place_name].iloc[0]
end_info = df_places[df_places['place_name'] == end_place_name].iloc[0]

df_bts_master['dist_start'] = [haversine_distance(start_info['latitude'], start_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

df_bts_master['dist_end'] = [haversine_distance(end_info['latitude'], end_info['longitude'], r['lat'], r['lng']) for i, r in df_bts_master.iterrows()]
nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้าด้วยตัวเอง" if nearest_bts_start['dist_start'] <= 100 else "🚖 เรียก Taxi / Grab Assistance (ระยะทางเกิน 100 เมตร)"
transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้าด้วยตัวเอง" if nearest_bts_end['dist_end'] <= 100 else "🚖 เรียก Taxi / Grab Assistance (ระยะทางเกิน 100 เมตร)"

is_hospital = "hospital" in end_place_name.lower() or "โรงพยาบาล" in end_place_name

col1, col2 = st.columns([1, 2])

with col1:
    if is_hospital and not is_urgent:
        st.warning("✨ **ระบบตรวจพบสิทธิ์สวัสดิการเดินทางพิเศษของคุณ**")
        st.markdown("""
         เนื่องจากปลายทางของคุณคือสถานพยาบาล และไม่มีความจำเป็นต้องรีบเร่งด่วน AI แนะนำให้ใช้ **บริการสวัสดิการขนส่งสาธารณะเพื่อผู้พิการฟรี** ดังนี้ครับ:
        * 🚍 **สวัสดิการรถตู้รับ-ส่งวีลแชร์ กทม.:** ออกแบบมาเพื่อผู้ใช้วีลแชร์โดยเฉพาะ โทรนัดหมายล่วงหน้าที่สายด่วน กทม. **โทร. 1555** หรือสายด่วนคนพิการ **โทร. 1479**
        * 🏥 **รถตู้รับ-ส่ง สปสช.:** สำหรับเดินทางไปโรงพยาบาลรัฐตามสิทธิ์การรักษา ประสานงานได้ที่สายด่วน สปสช. **โทร. 1330**
        """)
        st.write("---")

    st.markdown("### 📊 ตัวเลือกเส้นทางแนะนำ")
    
    st.markdown("#### 🚇 ตัวเลือกที่ 1: ระบบรางเชื่อมต่อด่วน (BTS)")
    st.caption(f"⏱️ ใช้เวลาประมาณ 25-35 นาที | 💰 ค่าใช้จ่ายประมาณ: {'50' if '🚶' in transport_first_leg else '150'} บาท")
    st.info(f"**1. ออกจากจุดเริ่ม:** {transport_first_leg} ไปสถานี **{nearest_bts_start['clean_name']}** ({nearest_bts_start['dist_start']:.1f} ม.)")
    if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
        st.info(f"**2. ขนส่งระบบราง:** ขึ้น BTS จากสถานี **{nearest_bts_start['clean_name']}** ไปลงสถานี **{nearest_bts_end['clean_name']}**")
    st.info(f"**3. เข้าสู่จุดหมาย:** {transport_last_leg} จากสถานีไปยังเป้าหมาย **{end_place_name}** ({nearest_bts_end['dist_end']:.1f} ม.)")
    
    st.write("---")
    
    st.markdown("#### 🚌 ตัวเลือกที่ 2: รถเมล์ปรับอากาศชานต่ำ (Low-Floor Bus)")
    st.caption("⏱️ ใช้เวลาประมาณ 45-60 นาที (ขึ้นอยู่กับการจราจร) | 💰 ค่าใช้จ่ายคงที่: 20 บาท ตลอดสาย")
    
    simulated_bus_line = "1-1 (29)" if "หมอชิต" in nearest_bts_start['clean_name'] or "Victory" in start_place_name else "1-3 (34)"
    
    st.success(f"""
    **แนะนำเดินทางด้วยรถเมล์สาย: {simulated_bus_line} (ไทยสมายล์บัส)**
    * ตัวรถเป็นประเภทชานต่ำ (Low-Floor) มีทางลาดไฮโดรลิกและพื้นที่ล็อกวีลแชร์ปลอดภัย 100%
    * **การเดินทาง:** เดินเท้าหรือเรียกรถระยะสั้นไปขึ้นที่ป้ายจอดรถเมล์ชานต่ำใกล้เคียง และนั่งยาวตรงสู่พื้นที่แนวปลายทาง 
    """)
    
    if prefer_cheap and not is_urgent:
        st.annotation("🎯 **AI แนะนำตัวเลือกนี้:** เนื่องจากคุณเปิดโหมดประหยัดพลังงานและไม่รีบด่วน รถเมล์ชานต่ำช่วยลดค่าใช้จ่ายลงได้ถึง 80%!")

with col2:
    st.markdown("### 🗺️ แผนที่พิกัด Multi-modal Interactive Map")
    
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=13)
    
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_place_name}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_place_name}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    
    folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS แรก: {nearest_bts_start['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
    folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง: {nearest_bts_end['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
    
    for idx, row in df_bus_stops.head(5).iterrows(): 
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=f"จุดจอดรถเมล์ชานต่ำรองรับวีลแชร์: {row['place_name']}",
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.7
        ).add_to(m)
    
    folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=6, opacity=0.8, popup="เส้นทางรถไฟฟ้า BTS").add_to(m)
    
    color_leg1 = 'red' if nearest_bts_start['dist_start'] > 100 else 'orange'
    dash_leg1 = None if nearest_bts_start['dist_start'] > 100 else '5, 5'
    folium.PolyLine([[start_info['latitude'], start_info['longitude']], [nearest_bts_start['lat'], nearest_bts_start['lng']]], color=color_leg1, weight=4, dash_array=dash_leg1).add_to(m)
    
    folium.PolyLine([[start_info['latitude'], start_info['longitude']], [end_info['latitude'], end_info['longitude']]], color='purple', weight=3, opacity=0.6, dash_array='10, 10', popup="แนววิ่งรถเมล์อารยสถาปัตย์").add_to(m)
    
    st_folium(m, width="100%", height=580)
