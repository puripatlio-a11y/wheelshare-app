import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V2", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบคำนวณและวางแผนเส้นทางเชื่อมต่ออัจฉริยะเพื่อผู้ใช้วีลแชร์")
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
    
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()
    df_accessibility['clean_name'] = df_accessibility['สถานี'].str.replace('สถานี', '').str.strip()
    
    df_bts_master = pd.merge(
        df_accessibility[['clean_name', 'สถานี', 'มีลิฟต์', 'ทางลาดสำหรับรถเข็น']], 
        df_stations[['clean_name', 'lat', 'lng']], 
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)
    return df_places, df_bts_master

try:
    df_places, df_bts_master = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ ไม่สามารถเปิดไฟล์ข้อมูลได้: {e}")
    st.stop()

st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")
place_list = df_places['place_name'].tolist()
start_place_name = st.sidebar.selectbox("📍 เลือกจุดต้นทาง (Origin):", place_list, index=13 if len(place_list) > 13 else 0)
end_place_name = st.sidebar.selectbox("🏁 เลือกจุดปลายทาง (Destination):", place_list, index=0)

start_info = df_places[df_places['place_name'] == start_place_name].iloc[0]
end_info = df_places[df_places['place_name'] == end_place_name].iloc[0]

dist_start_to_bts = [haversine_distance(start_info['latitude'], start_info['longitude'], row['lat'], row['lng']) for idx, row in df_bts_master.iterrows()]
df_bts_master['dist_start'] = dist_start_to_bts
nearest_bts_start = df_bts_master.sort_values(by='dist_start').iloc[0]

dist_end_to_bts = [haversine_distance(end_info['latitude'], end_info['longitude'], row['lat'], row['lng']) for idx, row in df_bts_master.iterrows()]
df_bts_master['dist_end'] = dist_end_to_bts
nearest_bts_end = df_bts_master.sort_values(by='dist_end').iloc[0]

transport_first_leg = "🚶 เข็นวีลแชร์เดินเท้าด้วยตัวเอง" if nearest_bts_start['dist_start'] <= 100 else "🚖 เรียก Taxi / Grab Assistance (เนื่องจากห่างไกลเกิน 100 เมตร)"
transport_last_leg = "🚶 เข็นวีลแชร์เดินเท้าด้วยตัวเอง" if nearest_bts_end['dist_end'] <= 100 else "🚖 เรียก Taxi / Grab Assistance (เนื่องจากห่างไกลเกิน 100 เมตร)"

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 📊การเดินทางแนะนำ")
    st.write(f"**📍 ต้นทาง:** {start_place_name}")
    st.write(f"**🏁 ปลายทาง:** {end_place_name}")
    st.write("---")
    st.info(f"**ขั้นแรกออกจากจุดเริ่ม:** {transport_first_leg} ไปที่ **สถานี {nearest_bts_start['clean_name']}** (ระยะทาง {nearest_bts_start['dist_start']:.1f} เมตร)")
    if nearest_bts_start['clean_name'] != nearest_bts_end['clean_name']:
        st.info(f"**ขั้นเชื่อมต่อระบบราง:** ขึ้น BTS จากสถานี **{nearest_bts_start['clean_name']}** ไปลงสถานีเป้าหมาย **{nearest_bts_end['clean_name']}**")
    st.info(f"**ขั้นเข้าสู่จุดหมาย:** {transport_last_leg} จากสถานี {nearest_bts_end['clean_name']} ไปยังเป้าหมาย **{end_place_name}** (ระยะทาง {nearest_bts_end['dist_end']:.1f} เมตร)")

with col2:
    m = folium.Map(location=[(start_info['latitude'] + end_info['latitude'])/2, (start_info['longitude'] + end_info['longitude'])/2], zoom_start=13)
    folium.Marker([start_info['latitude'], start_info['longitude']], popup=f"ต้นทาง: {start_place_name}", icon=folium.Icon(color='orange', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_info['latitude'], end_info['longitude']], popup=f"ปลายทาง: {end_place_name}", icon=folium.Icon(color='green', icon='flag', prefix='fa')).add_to(m)
    folium.Marker([nearest_bts_start['lat'], nearest_bts_start['lng']], popup=f"BTS แรก: {nearest_bts_start['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
    folium.Marker([nearest_bts_end['lat'], nearest_bts_end['lng']], popup=f"BTS ปลายทาง: {nearest_bts_end['clean_name']}", icon=folium.Icon(color='blue', icon='train', prefix='fa')).add_to(m)
    
    color_leg1 = 'red' if nearest_bts_start['dist_start'] > 100 else 'orange'
    dash_leg1 = None if nearest_bts_start['dist_start'] > 100 else '5, 5'
    folium.PolyLine([[start_info['latitude'], start_info['longitude']], [nearest_bts_start['lat'], nearest_bts_start['lng']]], color=color_leg1, weight=5, dash_array=dash_leg1).add_to(m)
    folium.PolyLine([[nearest_bts_start['lat'], nearest_bts_start['lng']], [nearest_bts_end['lat'], nearest_bts_end['lng']]], color='blue', weight=7, opacity=0.8).add_to(m)
    
    color_leg3 = 'red' if nearest_bts_end['dist_end'] > 100 else 'darkgreen'
    dash_leg3 = None if nearest_bts_end['dist_end'] > 100 else '5, 5'
    folium.PolyLine([[nearest_bts_end['lat'], nearest_bts_end['lng']], [end_info['latitude'], end_info['longitude']]], color=color_leg3, weight=5, dash_array=dash_leg3).add_to(m)
    st_folium(m, width="100%", height=550)
