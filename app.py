import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Accessibility Route Planner V5.3", layout="wide")
st.title("♿ AI Accessibility Route Planner (Wheelshare)")
st.subheader("ระบบวางแผนเส้นทางอัจฉริยะ (เวอร์ชันเน้นสถานที่หลัก และแนะนำทุกสายรถเมล์ที่ผ่านร่วมกัน)")
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
    # โหลดไฟล์ข้อมูล
    df_places = pd.read_csv('bangkok_places_bus_spot.csv')
    df_stations = pd.read_csv('bts_station.csv')
    df_accessibility = pd.read_csv('BTS for wheelchair users spreadsheet - BTS green line.csv')
    df_bus_routes = pd.read_csv('ThaiSmalieBus  - Sheet1.csv')
    
    # ทำความสะอาดข้อมูล BTS
    df_stations['clean_name'] = df_stations['name'].str.replace('สถานี', '').str.strip()
    df_accessibility['clean_name'] = df_accessibility['สถานี'].str.replace('สถานี', '').str.strip()
    df_bts_master = pd.merge(
        df_accessibility[['clean_name', 'สถานี', 'มีลิฟต์', 'ทางลาดสำหรับรถเข็น']], 
        df_stations[['clean_name', 'lat', 'lng']], 
        on='clean_name', how='inner'
    ).drop_duplicates(subset=['clean_name']).reset_index(drop=True)
    
    # 🎯 [ลอจิกคัดกรอง] กรองเอาป้ายรถเมล์ย่อยๆ ออกจาก Dropdown (คงไว้เฉพาะสถานที่หลักและโรงพยาบาล)
    # โดยคัดกรองแถวที่มีรหัสสายรถเมล์ชานต่ำ เช่น 1-1, 1-13 หรือคำว่า สถานีรถไฟ ออกไป เพื่อความสะอาดตา
    df_clean_places = df_places[
        ~df_places['place_name'].str.contains(r'\d-\d', regex=True) & 
        ~df_places['place_name'].str.contains('สถานีรถไฟ|หมู่บ้าน|ตลาดยิ่งเจริญ', na=False)
    ].reset_index(drop=True)
    
    return df_clean_places, df_bts_master, df_bus_routes

try:
    df_places, df_bts_master, df_bus_routes = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ ไม่สามารถเปิดไฟล์ข้อมูลได้: {e}")
    st.stop()

# ─── แถบเมนูด้านซ้าย (Sidebar) ───
st.sidebar.header("🕹️ เมนูเลือกการเดินทาง")
place_list = sorted(df_places['place_name'].tolist())

# กำหนดจุดเริ่มต้นและปลายทางเริ่มต้นให้ใช้งานง่าย
default_start = place_list.index("Victory Monument") if "Victory Monument" in place_list else 0
hospital_options = [x for x in place_list if any(h in x.lower() for h in ["โรงพยาบาล", "รพ.", "hospital"])]
default_end = place_list.index(hospital_options[0]) if hospital_options else 0

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

    # 🎯 OPTION 1: BTS
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

    # 🎯 OPTION 2: รถเมล์ชานต่ำ (แนะนำทุกสายที่ผ่านร่วมกันอัตโนมัติ)
    elif "🚌" in travel_mode:
        st.markdown("#### 🚏 สายรถเมล์ไทยสมายล์บัสที่ผ่านเส้นทางของคุณ")
        
        # Dictionary แปลงคีย์เวิร์ดเพื่อค้นหากลุ่มคำภาษาไทยในไฟล์เส้นทางวิ่ง
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
        
        # ลอจิกค้นหาสายรถเมล์ที่ผ่านร่วมกัน
        matched_lines = []
        for idx, row in df_bus_routes.iterrows():
            route_text = str(row['ต้นทาง']) + str(row['ปลาย']) + str(row['ผ่าน'])
            
            match_start = any(k.lower() in route_text.lower() for k in start_keywords)
            match_end = any(k.lower() in route_text.lower() for k in end_keywords)
            
            # ⚡ ตรงตามเงื่อนไขที่คุณต้องการ: ถ้าผ่านร่วมกันทั้งสองจุด ให้เก็บสายนี้ไว้
            if match_start and match_end:
                matched_lines.append(row['สาย'])
                
        # ถ้าหาเส้นทางตรงแบบสายเดียวไม่เจอ ให้ค้นหาสายที่ผ่านใกล้เคียงมาแสดง
        if not matched_lines:
            for idx, row in df_bus_routes.iterrows():
                route_text = str(row['ต้นทาง']) + str(row['ปลาย']) + str(row['ผ่าน'])
                if any(k.lower() in route_text.lower() for k in start_keywords) or any(k.lower() in route_text.lower() for k in end_keywords):
                    matched_lines.append(row['สาย'])

        if matched_lines:
            # นำสายรถเมล์ที่ได้มาลบตัวซ้ำ และรวมเป็นข้อความเดียวกันเพื่อแสดงผลพร้อมกัน
            unique_lines_list = sorted(list(set(matched_lines)))
            all_suggested_lines = " หรือ ".join(unique_lines_list)
            
            st.success(f"✅ **AI ตรวจพบสายรถเมล์ชานต่ำที่ผ่านร่วมกัน: สาย {all_suggested_lines}**")
            st.markdown(f"""
