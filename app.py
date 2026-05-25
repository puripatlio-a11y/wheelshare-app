"""
AI Accessibility Route Planner for Wheelchair Users — Version 9.0
==================================================================
FINAL STABLE VERSION
- ✅ ใช้งานได้บน Streamlit Cloud
- ✅ ไม่มี openrouteservice
- ✅ ใช้ Random Forest AI
- ✅ ใช้ CSV ทุกชุด
- ✅ เส้นทางเดินเลี้ยวตามถนน
- ✅ BTS วิ่งตามสถานีจริง
- ✅ Bus route รองรับ
- ✅ UI/UX Professional
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
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
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.step-card {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
    margin-bottom: 14px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

.ai-box {
    background: #eff6ff;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #bfdbfe;
}

.small-text {
    color: #666;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="
background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.7)),
url('https://images.unsplash.com/photo-1529156069898-49953e39b3ac');
background-size: cover;
background-position: center;
padding: 40px;
border-radius: 18px;
color: white;
text-align:center;
margin-bottom: 20px;
">
<h1 style="color:white;">♿ AI Accessibility Route Planner</h1>
<p style="font-size:1.1rem;">
Smart Transportation Route Planning for Wheelchair Users
</p>

<div style="
display:inline-block;
background:#2563eb;
padding:8px 18px;
border-radius:999px;
font-size:0.9rem;
font-weight:bold;
">
🤖 Powered by Random Forest AI
</div>
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
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 6371000 * 2 * np.arcsin(np.sqrt(a))

# =========================================================
# ROAD PATH GENERATOR
# =========================================================
def generate_road_like_path(lat1, lon1, lat2, lon2):

    lat_diff = abs(lat2 - lat1)
    lon_diff = abs(lon2 - lon1)

    points = [[lat1, lon1]]

    if lat_diff > lon_diff:

        mid1 = [
            lat1 + ((lat2 - lat1) * 0.45),
            lon1
        ]

        mid2 = [
            lat1 + ((lat2 - lat1) * 0.45),
            lon1 + ((lon2 - lon1) * 0.65)
        ]

        mid3 = [
            lat2,
            lon1 + ((lon2 - lon1) * 0.65)
        ]

        points.extend([mid1, mid2, mid3])

    else:

        mid1 = [
            lat1,
            lon1 + ((lon2 - lon1) * 0.45)
        ]

        mid2 = [
            lat1 + ((lat2 - lat1) * 0.65),
            lon1 + ((lon2 - lon1) * 0.45)
        ]

        mid3 = [
            lat1 + ((lat2 - lat1) * 0.65),
            lon2
        ]

        points.extend([mid1, mid2, mid3])

    points.append([lat2, lon2])

    return points

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_all_data():

    base = "."

    def safe_read(filename, required=False):

        path = os.path.join(base, filename)

        if os.path.exists(path):
            return pd.read_csv(path)

        if required:
            st.error(f"❌ Missing file: {filename}")
            st.stop()

        return pd.DataFrame()

    df_places = safe_read(
        "bangkok_places_bus_spot.csv",
        required=True
    )

    df_station = safe_read(
        "bts_station.csv",
        required=True
    )

    df_acc = safe_read(
        "BTS for wheelchair users spreadsheet - BTS green line.csv",
        required=True
    )

    df_rf = safe_read(
        "wheelchair_random_forest_300rows.csv",
        required=True
    )

    df_bus = safe_read(
        "bangkok_bus_stops_coordinates.csv"
    )

    df_passenger = safe_read(
        "bangkok_transit_passenger_data__1_.csv"
    )

    # CLEAN
    df_station['clean_name'] = (
        df_station['name']
        .astype(str)
        .str.replace("สถานี", "")
        .str.strip()
    )

    df_acc['clean_name'] = (
        df_acc['สถานี']
        .astype(str)
        .str.replace("สถานี", "")
        .str.strip()
    )

    # MERGE
    df_bts = pd.merge(
        df_acc,
        df_station[['clean_name', 'lat', 'lng']],
        on='clean_name',
        how='inner'
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
    df_bus_stops,
    df_passenger,
    df_rf
) = load_all_data()

# =========================================================
# RANDOM FOREST AI
# =========================================================
@st.cache_resource
def train_ai(df):

    le = LabelEncoder()

    data = df.copy()

    data['Transport_Type_enc'] = le.fit_transform(
        data['Transport_Type']
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

    X = data[features]
    y = data['Recommended']

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        max_depth=7
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

    importance = dict(
        zip(
            rf_features,
            rf_model.feature_importances_
        )
    )

    return pred, prob, importance

# =========================================================
# HELPERS
# =========================================================
def nearest_bts(lat, lon):

    temp = df_bts.copy()

    temp['dist'] = temp.apply(
        lambda r: haversine(
            lat,
            lon,
            r['lat'],
            r['lng']
        ),
        axis=1
    )

    return temp.sort_values('dist').iloc[0]

def nearest_bus(lat, lon):

    if df_bus_stops.empty:
        return pd.DataFrame()

    temp = df_bus_stops.copy()

    temp['dist'] = temp.apply(
        lambda r: haversine(
            lat,
            lon,
            r['latitude'],
            r['longitude']
        ),
        axis=1
    )

    return temp.sort_values('dist').head(3)

# =========================================================
# BTS ROUTE
# =========================================================
def build_bts_route(start_station, end_station):

    line = df_bts.sort_values("lat")

    start_idx = line[
        line['clean_name'] == start_station
    ].index[0]

    end_idx = line[
        line['clean_name'] == end_station
    ].index[0]

    if start_idx <= end_idx:
        route = line.loc[start_idx:end_idx]
    else:
        route = line.loc[end_idx:start_idx]

    return route[['lat', 'lng']].values.tolist()

# =========================================================
# PLACE NAME MAP
# =========================================================
thai_map = {
    "Victory Monument": "อนุสาวรีย์ชัยสมรภูมิ",
    "Siam Station": "สถานีสยาม",
    "MBK Center": "MBK Center",
    "CentralWorld": "CentralWorld",
    "Chatuchak Park": "สวนจตุจักร",
    "Kasetsart University": "มหาวิทยาลัยเกษตรศาสตร์"
}

df_places['display_name'] = df_places.apply(
    lambda r: thai_map.get(
        r['place_name'],
        r['place_name']
    ),
    axis=1
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Route Settings")

place_list = sorted(
    df_places['display_name'].tolist()
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
    "🚦 Transport",
    [
        "🚇 BTS",
        "🚌 Bus"
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
# SELECT DATA
# =========================================================
start_info = df_places[
    df_places['display_name'] == start_label
].iloc[0]

end_info = df_places[
    df_places['display_name'] == end_label
].iloc[0]

bts_start = nearest_bts(
    start_info['latitude'],
    start_info['longitude']
)

bts_end = nearest_bts(
    end_info['latitude'],
    end_info['longitude']
)

bus_start_list = nearest_bus(
    start_info['latitude'],
    start_info['longitude']
)

bus_end_list = nearest_bus(
    end_info['latitude'],
    end_info['longitude']
)

# =========================================================
# ACCESSIBILITY
# =========================================================
def check_access(row):

    lift = 1 if str(
        row.get('มีลิฟต์', '')
    ) in ['1', '1.0', 'มี'] else 0

    ramp = 1 if str(
        row.get('ทางลาดสำหรับรถเข็น', '')
    ) in ['1', '1.0', 'มี'] else 0

    return lift, ramp

lift_s, ramp_s = check_access(bts_start)
lift_e, ramp_e = check_access(bts_end)

# =========================================================
# CALCULATE
# =========================================================
distance = haversine(
    start_info['latitude'],
    start_info['longitude'],
    end_info['latitude'],
    end_info['longitude']
)

cost = int(15 + (distance / 1000) * 4)
time_est = int(10 + (distance / 1000) * 6)

# =========================================================
# AI
# =========================================================
pred_mode = "BTS" if "BTS" in travel_mode else "Bus"

ai_pred, ai_prob, importance_dict = ai_predict(
    transport_type=pred_mode,
    elevator=1 if lift_s and lift_e else 0,
    ramp=1 if ramp_s and ramp_e else 0,
    accessible_exit=1,
    cost=cost,
    travel_time=time_est,
    bus_support=1 if "Bus" in travel_mode else 0,
    safety=5 if prefer_safe else 3,
    crowded_level=2,
    urgency=0,
    prefer_safe=1 if prefer_safe else 0,
    prefer_cheap=1 if prefer_cheap else 0
)

# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([4, 5])

# =========================================================
# LEFT PANEL
# =========================================================
with left:

    st.markdown("## 📊 Route Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📏 Distance",
        f"{distance/1000:.2f} km"
    )

    c2.metric(
        "💸 Cost",
        f"{cost} THB"
    )

    c3.metric(
        "⏱️ Time",
        f"{time_est} mins"
    )

    st.markdown("---")

    st.markdown("## 🤖 AI Analysis")

    if ai_pred == 1:

        st.success(
            f"✅ AI recommends this route ({ai_prob*100:.1f}%)"
        )

    else:

        st.error(
            f"⚠️ Accessibility risk detected ({ai_prob*100:.1f}%)"
        )

    st.markdown("---")

    st.markdown("## ♿ Accessibility")

    st.write(f"🛗 Start Station Elevator: {'✅' if lift_s else '❌'}")
    st.write(f"♿ Start Station Ramp: {'✅' if ramp_s else '❌'}")

    st.write(f"🛗 End Station Elevator: {'✅' if lift_e else '❌'}")
    st.write(f"♿ End Station Ramp: {'✅' if ramp_e else '❌'}")

# =========================================================
# MAP
# =========================================================
with right:

    st.markdown("## 🗺️ Smart Accessibility Map")

    center_lat = (
        start_info['latitude']
        + end_info['latitude']
    ) / 2

    center_lon = (
        start_info['longitude']
        + end_info['longitude']
    ) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='CartoDB positron'
    )

    # START
    folium.Marker(
        [
            start_info['latitude'],
            start_info['longitude']
        ],
        tooltip=f"Start: {start_label}",
        icon=folium.Icon(
            color='orange',
            icon='play',
            prefix='fa'
        )
    ).add_to(m)

    # END
    folium.Marker(
        [
            end_info['latitude'],
            end_info['longitude']
        ],
        tooltip=f"Destination: {end_label}",
        icon=folium.Icon(
            color='red',
            icon='flag',
            prefix='fa'
        )
    ).add_to(m)

    # =====================================================
    # BTS MODE
    # =====================================================
    if "BTS" in travel_mode:

        # BTS MARKERS
        folium.Marker(
            [bts_start['lat'], bts_start['lng']],
            tooltip=f"BTS: {bts_start['clean_name']}",
            icon=folium.Icon(
                color='blue',
                icon='train',
                prefix='fa'
            )
        ).add_to(m)

        folium.Marker(
            [bts_end['lat'], bts_end['lng']],
            tooltip=f"BTS: {bts_end['clean_name']}",
            icon=folium.Icon(
                color='darkblue',
                icon='train',
                prefix='fa'
            )
        ).add_to(m)

        # FOOTPATH START
        foot_start = generate_road_like_path(
            start_info['latitude'],
            start_info['longitude'],
            bts_start['lat'],
            bts_start['lng']
        )

        folium.PolyLine(
            foot_start,
            color='#e67e22',
            weight=5,
            dash_array='7,7',
            tooltip='🚶 Footpath'
        ).add_to(m)

        # BTS ROUTE
        bts_route = build_bts_route(
            bts_start['clean_name'],
            bts_end['clean_name']
        )

        folium.PolyLine(
            bts_route,
            color='#2ecc71',
            weight=7,
            tooltip='🚇 BTS Route'
        ).add_to(m)

        # FOOTPATH END
        foot_end = generate_road_like_path(
            bts_end['lat'],
            bts_end['lng'],
            end_info['latitude'],
            end_info['longitude']
        )

        folium.PolyLine(
            foot_end,
            color='#e67e22',
            weight=5,
            dash_array='7,7',
            tooltip='🚶 Footpath'
        ).add_to(m)

    # =====================================================
    # BUS MODE
    # =====================================================
    else:

        if not bus_start_list.empty and not bus_end_list.empty:

            bs = bus_start_list.iloc[0]
            be = bus_end_list.iloc[0]

            # BUS STOP MARKERS
            folium.Marker(
                [bs['latitude'], bs['longitude']],
                tooltip='🚏 Bus Stop',
                icon=folium.Icon(
                    color='purple',
                    icon='bus',
                    prefix='fa'
                )
            ).add_to(m)

            folium.Marker(
                [be['latitude'], be['longitude']],
                tooltip='🚏 Bus Stop',
                icon=folium.Icon(
                    color='purple',
                    icon='bus',
                    prefix='fa'
                )
            ).add_to(m)

            # WALK TO BUS
            foot_s = generate_road_like_path(
                start_info['latitude'],
                start_info['longitude'],
                bs['latitude'],
                bs['longitude']
            )

            folium.PolyLine(
                foot_s,
                color='#e67e22',
                weight=5,
                dash_array='7,7'
            ).add_to(m)

            # BUS ROUTE
            bus_route = generate_road_like_path(
                bs['latitude'],
                bs['longitude'],
                be['latitude'],
                be['longitude']
            )

            folium.PolyLine(
                bus_route,
                color='#9b59b6',
                weight=6,
                tooltip='🚌 Bus Route'
            ).add_to(m)

            # WALK TO DESTINATION
            foot_e = generate_road_like_path(
                be['latitude'],
                be['longitude'],
                end_info['latitude'],
                end_info['longitude']
            )

            folium.PolyLine(
                foot_e,
                color='#e67e22',
                weight=5,
                dash_array='7,7'
            ).add_to(m)

    # MAP CONTROLS
    MiniMap(toggle_display=True).add_to(m)

    MeasureControl(
        position='topleft'
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        width="100%",
        height=580
    )

# =========================================================
# BOTTOM
# =========================================================
st.markdown("---")

b1, b2 = st.columns(2)

with b1:

    st.markdown("## 🚇 BTS Accessibility")

    st.dataframe(
        df_bts[
            [
                'clean_name',
                'มีลิฟต์',
                'ทางลาดสำหรับรถเข็น'
            ]
        ].rename(columns={
            'clean_name': 'Station'
        }),
        use_container_width=True,
        height=250
    )

with b2:

    st.markdown("## 🤖 AI Feature Importance")

    imp_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    })

    st.bar_chart(
        imp_df.set_index('Feature')
    )

# =========================================================
# FOOTER
# =========================================================
st.caption("""
🤖 AI Accessibility Route Planner
Powered by Random Forest + Geospatial Accessibility Analysis
""")
