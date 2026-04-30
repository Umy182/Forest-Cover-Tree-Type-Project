import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import gdown

# ── Download model from Google Drive if not already present ──────────────────
MODEL_PATH    = "best_model.joblib"
GDRIVE_FILE_ID = "1659iKQ4aMca2jfEuZ-laZ9MYBePniyjr"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... (first time only, please wait ⏳)"):
        gdown.download(
            url=f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&confirm=t",
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Cover Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(160deg, #0d1f12 0%, #122a1a 40%, #1a3d25 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit header ── */
header[data-testid="stHeader"] { background: transparent; }

/* ── Hero section ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #d4edda;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.4);
}
.hero p {
    color: #8fbc8f;
    font-size: 1.1rem;
    font-weight: 300;
    margin-top: 0.5rem;
    letter-spacing: 0.5px;
}

/* ── Divider ── */
.green-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #4caf50, transparent);
    margin: 1rem auto;
    width: 60%;
    border: none;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 28, 15, 0.85) !important;
    border-right: 1px solid rgba(76, 175, 80, 0.2);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5d6a7 !important;
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    border-bottom: 1px solid rgba(76,175,80,0.25);
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}
[data-testid="stSidebar"] label {
    color: #c8e6c9 !important;
    font-size: 0.88rem;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #b2dfdb !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(27, 67, 35, 0.9), rgba(15, 40, 20, 0.95));
    border: 1px solid rgba(76, 175, 80, 0.35);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}
.result-card .tree-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-card .tree-name {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #a5d6a7;
    margin: 0;
}
.result-card .tree-class {
    color: #4caf50;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.result-card .tree-desc {
    color: #81c784;
    font-size: 0.95rem;
    font-weight: 300;
    margin-top: 1rem;
    line-height: 1.6;
    max-width: 420px;
    margin-left: auto;
    margin-right: auto;
}

/* ── Info cards ── */
.info-card {
    background: rgba(18, 45, 24, 0.7);
    border: 1px solid rgba(76,175,80,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #c8e6c9;
    font-size: 0.9rem;
}
.info-card strong { color: #a5d6a7; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #388e3c);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2rem;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(46,125,50,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #43a047);
    box-shadow: 0 6px 20px rgba(46,125,50,0.5);
    transform: translateY(-1px);
}

/* ── Metric boxes ── */
.metric-row {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.metric-box {
    background: rgba(46,125,50,0.15);
    border: 1px solid rgba(76,175,80,0.25);
    border-radius: 10px;
    padding: 0.8rem 1.4rem;
    text-align: center;
    min-width: 130px;
}
.metric-box .metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #69f0ae;
}
.metric-box .metric-label {
    font-size: 0.75rem;
    color: #81c784;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.1rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: rgba(139,195,74,0.4);
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)

# ── Tree type reference data ──────────────────────────────────────────────────
TREE_TYPES = {
    1: {
        "name": "Spruce / Fir",
        "emoji": "🌲",
        "color": "#1b5e20",
        "desc": "Dense coniferous forests found at high elevations. Spruce and Fir dominate cold, moist habitats in the Rocky Mountain subalpine zone."
    },
    2: {
        "name": "Lodgepole Pine",
        "emoji": "🌳",
        "color": "#2e7d32",
        "desc": "One of the most widespread pines in North America. Thrives after disturbances like fire, forming dense even-aged stands across a wide elevation range."
    },
    3: {
        "name": "Ponderosa Pine",
        "emoji": "🌴",
        "color": "#33691e",
        "desc": "Distinctive orange-barked pine found at lower elevations. Adapted to drier, sunnier slopes and well-drained rocky soils."
    },
    4: {
        "name": "Cottonwood / Willow",
        "emoji": "🍃",
        "color": "#558b2f",
        "desc": "Riparian specialists growing along rivers and streams. Require consistent moisture and are often found at the lowest elevations near water."
    },
    5: {
        "name": "Aspen",
        "emoji": "🍂",
        "color": "#827717",
        "desc": "Deciduous trees famous for their shimmering golden fall color. Often grow in clonal colonies connected by a shared root system."
    },
    6: {
        "name": "Douglas Fir",
        "emoji": "🌿",
        "color": "#1a5276",
        "desc": "A towering conifer native to the western U.S. Prefers moderate elevations with good moisture, often mixed with Ponderosa Pine."
    },
    7: {
        "name": "Krummholz",
        "emoji": "🏔️",
        "color": "#4a235a",
        "desc": "Stunted, wind-sculpted trees at the treeline. Extreme conditions force them into low, contorted shapes — the German word means 'crooked wood'."
    },
}

WILDERNESS_AREAS = {
    "Rawah (Area 1)": 1,
    "Neota (Area 2)": 2,
    "Comanche Peak (Area 3)": 3,
    "Cache la Poudre (Area 4)": 4,
}

SOIL_TYPES = {f"Soil Type {i}": i for i in range(1, 41)}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        model = joblib.load("best_model.joblib")
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌲 Forest Cover Predictor</h1>
    <p>Predict which tree species dominates a wilderness area based on terrain characteristics</p>
</div>
<hr class="green-line">
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Forest Features")
    st.markdown("Adjust the terrain parameters below, then click **Predict**.")

    st.markdown("### 📍 Location")
    wilderness = st.selectbox("Wilderness Area", list(WILDERNESS_AREAS.keys()))
    soil_label = st.selectbox("Soil Type", list(SOIL_TYPES.keys()))

    st.markdown("### ⛰️ Elevation & Terrain")
    elevation   = st.slider("Elevation (meters)",      1800, 3900, 2800)
    aspect      = st.slider("Aspect (degrees)",           0,  360,  180)
    slope       = st.slider("Slope (degrees)",            0,   66,   14)

    st.markdown("### 💧 Distances")
    horiz_hydro = st.slider("Horizontal Distance to Hydrology (m)",  0, 1400,  200)
    vert_hydro  = st.slider("Vertical Distance to Hydrology (m)",  -170, 600,   30)
    horiz_road  = st.slider("Horizontal Distance to Roadways (m)",   0, 7000, 1500)
    horiz_fire  = st.slider("Horizontal Distance to Fire Points (m)",0, 7200, 1500)

    st.markdown("### ☀️ Hillshade Index")
    shade_9am   = st.slider("Hillshade 9am",    0, 254, 212)
    shade_noon  = st.slider("Hillshade Noon",   0, 254, 223)
    shade_3pm   = st.slider("Hillshade 3pm",    0, 254, 142)

    st.markdown("")
    predict_btn = st.button("🌲 Predict Cover Type", use_container_width=True)

# ── Main content ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown("### 📋 Input Summary")

    wilderness_num = WILDERNESS_AREAS[wilderness]
    soil_num       = SOIL_TYPES[soil_label]

    info_items = [
        ("📍 Wilderness Area", wilderness),
        ("🌱 Soil Type",       soil_label),
        ("⛰️ Elevation",       f"{elevation} m"),
        ("🧭 Aspect",          f"{aspect}°"),
        ("📐 Slope",           f"{slope}°"),
        ("💧 Horiz. Distance to Hydrology", f"{horiz_hydro} m"),
        ("💧 Vert. Distance to Hydrology",  f"{vert_hydro} m"),
        ("🛤️ Distance to Roadways",          f"{horiz_road} m"),
        ("🔥 Distance to Fire Points",       f"{horiz_fire} m"),
        ("☀️ Hillshade 9am / Noon / 3pm",   f"{shade_9am} / {shade_noon} / {shade_3pm}"),
    ]

    for label, value in info_items:
        st.markdown(
            f'<div class="info-card"><strong>{label}:</strong> {value}</div>',
            unsafe_allow_html=True
        )

with col2:
    st.markdown("### 🔍 Prediction Result")

    if not predict_btn:
        st.markdown("""
        <div class="result-card">
            <div class="tree-emoji">🌲</div>
            <p class="tree-name" style="color:#4a7c59;">Awaiting Input</p>
            <p class="tree-desc">Set your terrain parameters in the sidebar and click <strong>Predict Cover Type</strong> to see which tree species is most likely to dominate.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None:
            st.error(
                "⚠️ **Model file not found!**\n\n"
                "Make sure `best_model.joblib` is in the same folder as `app.py`.\n\n"
                "Run the **save_model.py** cell at the end of your notebook first!"
            )
        else:
            # Build input feature vector
            wilderness_flags = [1 if wilderness_num == i else 0 for i in range(1, 5)]
            soil_flags       = [1 if soil_num == i else 0 for i in range(1, 41)]

            feature_names = (
                ["Elevation", "Aspect", "Slope",
                 "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways",
                 "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
                 "Horizontal_Distance_To_Fire_Points"] +
                [f"Wilderness_Area{i}" for i in range(1, 5)] +
                [f"Soil_Type{i}" for i in range(1, 41)]
            )

            values = (
                [elevation, aspect, slope,
                 horiz_hydro, vert_hydro, horiz_road,
                 shade_9am, shade_noon, shade_3pm, horiz_fire]
                + wilderness_flags
                + soil_flags
            )

            X_input = pd.DataFrame([values], columns=feature_names)

            prediction = model.predict(X_input)[0]
            proba      = model.predict_proba(X_input)[0]

            # Some models return 0-indexed, adjust if needed
            if prediction == 0:
                prediction = 1
                proba_labels = list(range(1, 8))
            else:
                proba_labels = list(range(1, 8))

            tree  = TREE_TYPES[prediction]
            conf  = proba.max() * 100

            st.markdown(f"""
            <div class="result-card">
                <div class="tree-emoji">{tree['emoji']}</div>
                <p class="tree-name">{tree['name']}</p>
                <p class="tree-class">Cover Type {prediction}</p>
                <p class="tree-desc">{tree['desc']}</p>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-val">{conf:.1f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{elevation}m</div>
                        <div class="metric-label">Elevation</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">WA{wilderness_num}</div>
                        <div class="metric-label">Wilderness</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Probability per Cover Type**")
            prob_df = pd.DataFrame({
                "Cover Type": [f"{TREE_TYPES[i]['emoji']} {TREE_TYPES[i]['name']}" for i in range(1, 8)],
                "Probability": proba[:7] if len(proba) >= 7 else list(proba) + [0] * (7 - len(proba))
            }).set_index("Cover Type")
            st.bar_chart(prob_df, color="#4caf50", height=220)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Forest Cover Type Dataset · UCI Machine Learning Repository ·
    Research Question: Which tree type would cover best in a specific wilderness?
</div>
""", unsafe_allow_html=True)
