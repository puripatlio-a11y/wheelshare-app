"""
AI Accessibility Route Planner for Wheelchair Users — Version 9.1
==================================================================
✔ ใช้เส้นทางเดินจริงจาก OpenStreetMap (OSRM API ฟรี)
✔ ไม่ต้องใช้ openrouteservice แล้ว (แก้ ModuleNotFoundError)
✔ ใช้ข้อมูลจาก CSV เยอะขึ้น
✔ มีระบบ AI Random Forest
✔ มีระบบกันไฟล์หาย
✔ UI อ่านง่ายขึ้น
✔ ใช้งานบน Streamlit Cloud ได้ทันที
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import requests
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from folium.plugins import MiniMap, MeasureControl
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Accessibility Route Planner",
    page_icon="♿",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 1.5rem;
}
.metric-card {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #eaeaea;
}
.step-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 5px solid #1976d2;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="
background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.7)),
url('https://img.freepik.com/free-photo/full-shot-happy-friends-chatting-outside_23-2149391993.jpg?w=740');
background-size: cover;
background-position: center;
padding: 35px;
border-radius: 15px;
text-align: center;
color: white;
margin-bottom: 25px;
">
<h1 style="color:white;">♿ AI Accessibility Route Planner</h1>
<p style="font-size:18px;">
ระบบวิเคราะห์และวางแผนเส้นทางสำหรับผู้ใช้รถเข็น
</p>
<span style="
background:#1976d2;
padding:7px 18px;
border-radius:20px;
font-weight:bold;
">
🤖 Random Forest AI Powered
</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HAVERSINE
# =========================================================
def haversine(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat/2)**2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon/2)**2
    )

    return 6371000 * 2 * np.arcsin(np.sqrt(a))

# =========================================================
# LOAD CSV
# =========================================================
@st.cache_data
def load_all_data():

    base = "."

    def safe_read_csv(file, required=False):

        path = os.path.join(base, file)

        if os.path.exists(path):
            return pd.read_csv(path)

        if required:
            st.error(f"❌ Missing file: {file}")
            st.stop()

        return pd.DataFrame()

    df_places = safe_read_csv(
        "bangkok_places_bus_spot.csv",
        required=True
    )

    df_station = safe_read_csv(
        "bts_station.csv",
        required=True
    )

    df_access = safe_read_csv(
        "BTS for wheelchair users spreadsheet - BTS green line.csv",
        required=True
    )

    df_rf = safe_read_csv(
        "wheelchair_random_forest_300rows.csv",
        required=True
    )

    df_bus = safe_read_csv(
        "bangkok_bus_stops_coordinates.csv"
    )

    df_passenger = safe_read_csv(
        "bangkok_transit_passenger_data__1_.csv"
    )

    # CLEAN
    df_station["clean_name"] = (
        df_station["name"]
        .astype(str)
        .str.replace("สถานี", "")
        .str.strip()
    )

    df_access["clean_name"] = (
        df_access["สถานี"]
        .astype(str)
        .str.replace("สถานี", "")
        .str.strip()
    )

    # MERGE BTS
    df_bts = pd.merge(
        df_access,
        df_station[["clean_name", "lat", "lng"]],
        on="clean_name",
        how="inner"
    )

    return (
        df_places,
        df_bts,
        df_bus,
        df_passenger,
        df_rf
    )

(
    df_places,
    df_bts,
    df_bus,
    df_passenger,
    df_rf
) = load_all_data()

# =========================================================
# AI MODEL
# =========================================================
@st.cache_resource
def train_ai(df_rf):

    le = LabelEncoder()

    df = df_rf.copy()

    df["Transport_Type_enc"] = le.fit_transform(
        df["Transport_Type"]
    )

    features = [
        'Elevator',
        'Ramp',
        'Accessible_Exit',
        'Cost',
        'Travel_Time',
        'BusSupport',
        'Safety',
        'Crowded_Level',
        'Urgency',
        'Prefer_Safe',
        'Prefer_Cheap',
        'Transport_Type_enc'
    ]

    X = df[features]
    y = df["Recommended"]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        random_state=42
    )

    model.fit(X, y)

    return model, le, features

rf_model, le_transport, rf_features = train_ai(df_rf)

# =========================================================
# AI PREDICT
# =========================================================
def ai_predict(
    transport_type,
    elevator,
    ramp,
    accessible_exit,
    cost,
    travel_time,
    bus_support,
    safety,
    crowded_level,
    urgency,
    prefer_safe,
    prefer_cheap
):

    try:
        transport_enc = le_transport.transform(
            [transport_type]
        )[0]

    except:
        transport_enc = 0

    row = pd.DataFrame([[
        elevator,
        ramp,
        accessible_exit,
        cost,
        travel_time,
        bus_support,
        safety,
        crowded_level,
        urgency,
        prefer_safe,
        prefer_cheap,
        transport_enc
    ]], columns=rf_features)

    pred = rf_model.predict(row)[0]
    prob = rf_model.predict_proba(row)[0][1]

    return pred, prob

# =========================================================
# NEAREST BTS
# =========================================================
def nearest_bts(lat, lon):

    temp = df_bts.copy()

    temp["dist"] = temp.apply(
        lambda r: haversine(
            lat,
            lon,
            r["lat"],
            r["lng"]
        ),
        axis=1
    )

    return temp.sort_values("dist").iloc[0]

# =========================================================
# OSRM REAL ROAD ROUTE
# =========================================================
def get_osrm_route(lat1, lon1, lat2, lon2):

    url = (
        f"https://router.project-osrm.org/route/v1/foot/"
        f"{lon1},{lat1};{lon2},{lat2}"
        "?overview=full&geometries=geojson"
    )

    try:
        r = requests.get(url, timeout=15)

        data = r.json()

        coords = data["routes"][0]["geometry"]["coordinates"]

        route = [
            [c[1], c[0]]
            for c in coords
        ]

        return route

    except:
        return [
            [lat1, lon1],
            [lat2, lon2]
        ]

# =========================================================
# PLACE LIST
# =========================================================
thai_map = {
    "Victory Monument": "อนุสาวรีย์ชัยสมรภูมิ",
    "CentralWorld": "เซ็นทรัลเวิลด์",
    "MBK Center": "MBK Center",
    "Siriraj Hospital": "โรงพยาบาลศิริราช",
    "Ramathibodi Hospital": "โรงพยาบาลรามาธิบดี",
    "Rajavithi Hospital": "โรงพยาบาลราชวิถี",
    "Chatuchak Park": "สวนจตุจักร"
}

df_places["display_name"] = df_places.apply(
    lambda r: thai_map.get(
        r["place_name"],
        r["place_name"]
    ),
    axis=1
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("🕹️ Route Settings")

place_list = sorted(
    df_places["display_name"].tolist()
)

start_label = st.sidebar.selectbox(
    "📍 Start",
    place_list,
    index=0
)

end_label = st.sidebar.selectbox(
    "🏁 Destination",
    place_list,
    index=min(3, len(place_list)-1)
)

travel_mode = st.sidebar.radio(
    "🚦 Transportation",
    [
        "🚇 BTS",
        "🚌 Low-floor Bus"
    ]
)

prefer_safe = st.sidebar.checkbox(
    "🛡️ Prioritize Safety",
    value=True
)

prefer_cheap = st.sidebar.checkbox(
    "💰 Prioritize Cheap Cost",
    value=False
)

# =========================================================
# LOCATION INFO
# =========================================================
start_info = df_places[
    df_places["display_name"] == start_label
].iloc[0]

end_info = df_places[
    df_places["display_name"] == end_label
].iloc[0]

bts_start = nearest_bts(
    start_info["latitude"],
    start_info["longitude"]
)

bts_end = nearest_bts(
    end_info["latitude"],
    end_info["longitude"]
)

# =========================================================
# DISTANCE
# =========================================================
distance = haversine(
    start_info["latitude"],
    start_info["longitude"],
    end_info["latitude"],
    end_info["longitude"]
)

cost = int(15 + distance/1000 * 4)
time_est = int(10 + distance/1000 * 6)

# =========================================================
# AI
# =========================================================
pred, prob = ai_predict(
    transport_type="BTS",
    elevator=1,
    ramp=1,
    accessible_exit=1,
    cost=cost,
    travel_time=time_est,
    bus_support=0,
    safety=5 if prefer_safe else 3,
    crowded_level=2,
    urgency=0,
    prefer_safe=1 if prefer_safe else 0,
    prefer_cheap=1 if prefer_cheap else 0
)

# =========================================================
# MAIN LAYOUT
# =========================================================
col1, col2 = st.columns([4,5])

# =========================================================
# LEFT
# =========================================================
with col1:

    st.markdown("## 📊 Route Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📏 Distance",
        f"{distance/1000:.2f} km"
    )

    c2.metric(
        "💸 Cost",
        f"{cost} ฿"
    )

    c3.metric(
        "⏱️ Time",
        f"{time_est} min"
    )

    st.markdown("---")

    st.markdown("## 🤖 AI Analysis")

    if pred == 1:
        st.success(
            f"AI recommends this route "
            f"({prob*100:.1f}% confidence)"
        )

    else:
        st.error(
            f"AI warns accessibility risk "
            f"({prob*100:.1f}%)"
        )

    st.markdown("---")

    st.markdown("## 🚶 Route Steps")

    st.markdown(f"""
    <div class="step-card">
    <b>1.</b> เดินทางจาก <b>{start_label}</b>
    ไปยังสถานี BTS <b>{bts_start['clean_name']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="step-card">
    <b>2.</b> โดยสาร BTS จาก
    <b>{bts_start['clean_name']}</b>
    ไปยัง
    <b>{bts_end['clean_name']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="step-card">
    <b>3.</b> เดินทางต่อไปยัง
    <b>{end_label}</b>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RIGHT MAP
# =========================================================
with col2:

    st.markdown("## 🗺️ Real Walking Route")

    center_lat = (
        start_info["latitude"]
        + end_info["latitude"]
    ) / 2

    center_lon = (
        start_info["longitude"]
        + end_info["longitude"]
    ) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron"
    )

    # START
    folium.Marker(
        [
            start_info["latitude"],
            start_info["longitude"]
        ],
        tooltip="Start",
        icon=folium.Icon(
            color="orange",
            icon="play",
            prefix="fa"
        )
    ).add_to(m)

    # END
    folium.Marker(
        [
            end_info["latitude"],
            end_info["longitude"]
        ],
        tooltip="Destination",
        icon=folium.Icon(
            color="red",
            icon="flag",
            prefix="fa"
        )
    ).add_to(m)

    # BTS
    folium.Marker(
        [bts_start["lat"], bts_start["lng"]],
        tooltip=f"BTS {bts_start['clean_name']}",
        icon=folium.Icon(
            color="blue",
            icon="train",
            prefix="fa"
        )
    ).add_to(m)

    folium.Marker(
        [bts_end["lat"], bts_end["lng"]],
        tooltip=f"BTS {bts_end['clean_name']}",
        icon=folium.Icon(
            color="darkblue",
            icon="train",
            prefix="fa"
        )
    ).add_to(m)

    # REAL FOOTPATH
    foot1 = get_osrm_route(
        start_info["latitude"],
        start_info["longitude"],
        bts_start["lat"],
        bts_start["lng"]
    )

    foot2 = get_osrm_route(
        bts_end["lat"],
        bts_end["lng"],
        end_info["latitude"],
        end_info["longitude"]
    )

    folium.PolyLine(
        foot1,
        color="#e67e22",
        weight=5,
        dash_array="6,6",
        tooltip="🚶 Real Walking Route"
    ).add_to(m)

    # BTS LINE
    folium.PolyLine(
        [
            [bts_start["lat"], bts_start["lng"]],
            [bts_end["lat"], bts_end["lng"]]
        ],
        color="#2ecc71",
        weight=7,
        tooltip="🚇 BTS"
    ).add_to(m)

    folium.PolyLine(
        foot2,
        color="#e67e22",
        weight=5,
        dash_array="6,6",
        tooltip="🚶 Real Walking Route"
    ).add_to(m)

    # CONTROLS
    MiniMap(toggle_display=True).add_to(m)

    MeasureControl(
        position="topleft"
    ).add_to(m)

    st_folium(
        m,
        width="100%",
        height=600
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption("""
🤖 Powered by Random Forest AI + OpenStreetMap Routing (OSRM)
""")
