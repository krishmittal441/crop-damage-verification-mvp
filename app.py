import ee
import streamlit as st
import json
import datetime

# ============================================================
# EARTH ENGINE INITIALIZATION (SERVICE ACCOUNT)
# ============================================================
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)

ee.Initialize(credentials)

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Crop Damage Verification",
    layout="centered"
)

st.title("🛰️ Satellite-Based Crop Damage Verification")
st.caption("Before–After satellite analysis for government & insurance use")
st.divider()

# ============================================================
# INPUTS
# ============================================================
st.subheader("1️⃣ Location")

lat = st.number_input("Latitude", value=26.2, format="%.6f")
lon = st.number_input("Longitude", value=91.7, format="%.6f")

radius_km = st.selectbox(
    "AOI Radius (km)",
    options=[0.5, 1, 2],
    index=1,
    help="Smaller radius = more field-level accuracy"
)

st.divider()

st.subheader("2️⃣ Dates")

baseline_start = st.date_input(
    "Baseline Start (Before Event)",
    value=datetime.date(2023, 6, 1)
)
baseline_end = st.date_input(
    "Baseline End (Before Event)",
    value=datetime.date(2023, 6, 20)
)

damage_start = st.date_input(
    "Damage Start (After Event)",
    value=datetime.date(2023, 7, 5)
)
damage_end = st.date_input(
    "Damage End (After Event)",
    value=datetime.date(2023, 7, 25)
)

# ============================================================
# VALIDATION
# ============================================================
if baseline_start >= baseline_end:
    st.error("Baseline start must be before baseline end")
    st.stop()

if damage_start >= damage_end:
    st.error("Damage start must be before damage end")
    st.stop()

if damage_start <= baseline_end:
    st.error("Damage period must start AFTER baseline period")
    st.stop()

# ============================================================
# ANALYSIS FUNCTION
# ============================================================
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    # ---------- SENTINEL-2 (OPTICAL) ----------
    s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterBounds(aoi) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))

    def ndvi(img):
        return img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    def ndwi(img):
        return img.normalizedDifference(["B3", "B8"]).rename("NDWI")

    s2_before = s2.filterDate(str(b_start), str(b_end)).map(ndvi).map(ndwi)
    s2_after  = s2.filterDate(str(d_start), str(d_end)).map(ndvi).map(ndwi)

    if s2_before.size().getInfo() == 0 or s2_after.size().getInfo() == 0:
        return None

    ndvi_before = s2_before.select("NDVI").mean()
    ndvi_after  = s2_after.select("NDVI").mean()

    ndwi_before = s2_before.select("NDWI").mean()
    ndwi_after  = s2_after.select("NDWI").mean()

    ndvi_change = ndvi_after.subtract(ndvi_before)
    ndwi_change = ndwi_after.subtract(ndwi_before)

    # ---------- SENTINEL-1 (SAR VV) ----------
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(aoi) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")) \
        .select("VV")

    s1_before = s1.filterDate(str(b_start), str(b_end))
    s1_after  = s1.filterDate(str(d_start), str(d_end))

    if s1_before.size().getInfo() == 0 or s1_after.size().getInfo() == 0:
        sar_change = None
    else:
        sar_before = s1_before.mean()
        sar_after  = s1_after.mean()
        sar_change = sar_after.subtract(sar_before)

    # ---------- REDUCTION ----------
    def reduce(img):
        return img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e13
        ).getInfo()

    return {
        "ndvi_change": reduce(ndvi_change).get("NDVI"),
        "ndwi_change": reduce(ndwi_change).get("NDWI"),
        "sar_vv_change": reduce(sar_change).get("VV") if sar_change else None
    }

# ============================================================
# RUN ANALYSIS
# ============================================================
st.divider()
st.subheader("3️⃣ Run Analysis")

if st.button("🔍 Run Satellite Damage Assessment", use_container_width=True):

    with st.spinner("Analyzing satellite data..."):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if results is None:
        st.error("No usable satellite data for selected area & dates.")
        st.stop()

    ndvi = results["ndvi_change"]
    ndwi = results["ndwi_change"]
    sar  = results["sar_vv_change"]

    st.success("Analysis complete")
    st.divider()

    # ========================================================
    # RAW VALUES (NEVER HIDDEN)
    # ========================================================
    st.subheader("📊 Raw Satellite Values")

    col1, col2, col3 = st.columns(3)
    col1.metric("NDVI Change", round(ndvi, 3) if ndvi is not None else "N/A")
    col2.metric("NDWI Change", round(ndwi, 3) if ndwi is not None else "N/A")
    col3.metric("SAR VV Change (dB)", round(sar, 2) if sar is not None else "N/A")

    st.divider()

    # ========================================================
    # INTERPRETATION LOGIC
    # ========================================================
    st.subheader("🧠 Interpretation")

    interpretation = []
    confidence = "LOW"

    if ndwi is not None and sar is not None:
        if ndwi > 0.15 and sar < -1.5:
            interpretation.append("Flood / Excess water detected")
            confidence = "HIGH"

    if ndvi < -0.15:
        interpretation.append("Vegetation stress or crop damage")

    if not interpretation:
        interpretation.append("No strong disaster signal detected")

    st.write("*Assessment:*")
    for line in interpretation:
        st.write(f"• {line}")

    st.write(f"*Confidence:* {confidence}")

    st.caption("Raw values shown for transparency. Interpretation is rule-based (Phase-1).")
