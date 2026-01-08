import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------------------------------------
# EARTH ENGINE INIT (SERVICE ACCOUNT)
# -------------------------------------------------
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)
ee.Initialize(credentials)

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Crop Damage Verification", layout="centered")

st.title("🛰️ Satellite-Based Crop Damage Verification")
st.subheader("Government & Insurance Decision Support")
st.divider()

# -------------------------------------------------
# LOCATION INPUT
# -------------------------------------------------
lat = st.number_input("Latitude", value=26.2, format="%.6f")
lon = st.number_input("Longitude", value=93.8, format="%.6f")
radius_km = st.selectbox("AOI Radius (km)", [0.5, 1, 2, 5], index=1)

# -------------------------------------------------
# DATE INPUT
# -------------------------------------------------
st.info(
    "📌 Use WIDE windows during disasters (especially monsoon regions).\n\n"
    "Baseline: at least 2–3 weeks before event\n"
    "Damage: at least 2–3 weeks after event"
)

baseline_start = st.date_input("Baseline Start", datetime.date(2023, 6, 1))
baseline_end   = st.date_input("Baseline End", datetime.date(2023, 6, 20))

damage_start = st.date_input("Damage Start", datetime.date(2023, 7, 5))
damage_end   = st.date_input("Damage End", datetime.date(2023, 7, 30))

if baseline_start >= baseline_end or damage_start >= damage_end:
    st.error("Invalid date range")
    st.stop()

# -------------------------------------------------
# CORE ANALYSIS FUNCTION
# -------------------------------------------------
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    # -------------------------
    # OPTICAL (SENTINEL-2)
    # -------------------------
    def get_optical(start, end):
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
        )
        count = col.size()

        image = ee.Algorithms.If(
            count.gt(0),
            col.median()
              .normalizedDifference(["B8", "B4"]).rename("NDVI")
              .addBands(col.median().normalizedDifference(["B3", "B8"]).rename("NDWI")),
            None
        )

        return image, count

    nd_before, cnt_b = get_optical(b_start, b_end)
    nd_after, cnt_a = get_optical(d_start, d_end)

    # -------------------------
    # SAR (SENTINEL-1)
    # -------------------------
    def get_sar(start, end):
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV")
        )
        count = col.size()
        image = ee.Algorithms.If(count.gt(0), col.median().rename("SAR"), None)
        return image, count

    sar_before, sar_b = get_sar(b_start, b_end)
    sar_after, sar_a = get_sar(d_start, d_end)

    # -------------------------
    # CHANGE COMPUTATION
    # -------------------------
    results = {}

    if nd_before and nd_after:
        diff = ee.Image(nd_after).subtract(ee.Image(nd_before))
        stats = diff.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )
        results["NDVI Change"] = stats.get("NDVI").getInfo()
        results["NDWI Change"] = stats.get("NDWI").getInfo()
    else:
        results["NDVI Change"] = None
        results["NDWI Change"] = None

    if sar_before and sar_after:
        sar_diff = ee.Image(sar_after).subtract(ee.Image(sar_before))
        sar_stats = sar_diff.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )
        results["SAR Change"] = sar_stats.get("SAR").getInfo()
    else:
        results["SAR Change"] = None

    results["Optical Images (Before)"] = cnt_b.getInfo()
    results["Optical Images (After)"] = cnt_a.getInfo()
    results["SAR Images (Before)"] = sar_b.getInfo()
    results["SAR Images (After)"] = sar_a.getInfo()

    return results

# -------------------------------------------------
# RUN
# -------------------------------------------------
if st.button("🔍 Analyze Damage", use_container_width=True):

    with st.spinner("Processing satellite data..."):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    st.subheader("📊 Results")

    st.json(results)

    # -------------------------
    # INTERPRETATION
    # -------------------------
    if results["NDWI Change"] is not None and results["NDWI Change"] < -0.05:
        st.error("🌊 Flood / Surface Water Detected (NDWI)")
    elif results["NDVI Change"] is not None and results["NDVI Change"] < -0.15:
        st.warning("🌾 Vegetation Stress Detected (NDVI)")
    elif results["SAR Change"] is not None and abs(results["SAR Change"]) > 1:
        st.warning("📡 Surface Change Detected (SAR)")
    else:
        st.success("No strong damage signal detected")

    st.caption(
        "⚠️ Optical may fail during heavy clouds. SAR is authoritative during floods."
    )
