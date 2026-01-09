import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ======================================================
# EARTH ENGINE INITIALIZATION (SERVICE ACCOUNT)
# ======================================================
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)

ee.Initialize(credentials)

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="Crop Damage Verification",
    layout="centered"
)

st.title("🛰️ Satellite-Based Crop Damage Verification")
st.subheader("Event-based, auditable damage assessment for government & insurance")
st.divider()

# ======================================================
# LOCATION INPUT
# ======================================================
st.markdown("### Location")

lat = st.number_input("Latitude", value=26.2, format="%.6f")
lon = st.number_input("Longitude", value=93.8, format="%.6f")

radius_km = st.selectbox(
    "AOI Radius (km)",
    options=[0.5, 1, 2],
    index=2,
    help="Larger AOI improves satellite availability"
)

# ======================================================
# DATE INPUT
# ======================================================
st.markdown("### Dates")

baseline_start = st.date_input("Baseline Start", datetime.date(2023, 6, 1))
baseline_end   = st.date_input("Baseline End",   datetime.date(2023, 6, 20))

damage_start = st.date_input("Damage Start", datetime.date(2023, 7, 5))
damage_end   = st.date_input("Damage End",   datetime.date(2023, 7, 30))

if baseline_start >= baseline_end:
    st.error("Baseline dates invalid")
    st.stop()

if damage_start >= damage_end:
    st.error("Damage dates invalid")
    st.stop()

if damage_start <= baseline_end:
    st.error("Damage period must start after baseline")
    st.stop()

# ======================================================
# ANALYSIS FUNCTION
# ======================================================
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    # ---------- OPTICAL (Sentinel-2) ----------
    def optical_composite(start, end):
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(aoi)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
        )

        return ee.Image(
            ee.Algorithms.If(
                col.size().gt(0),
                col.median(),
                None
            )
        )

    before_opt = optical_composite(b_start, b_end)
    after_opt  = optical_composite(d_start, d_end)

    ndvi_change = None
    ndwi_change = None

    if before_opt and after_opt:
        ndvi_before = before_opt.normalizedDifference(["B8", "B4"])
        ndvi_after  = after_opt.normalizedDifference(["B8", "B4"])

        ndwi_before = before_opt.normalizedDifference(["B3", "B8"])
        ndwi_after  = after_opt.normalizedDifference(["B3", "B8"])

        diff = (
            ndvi_after.subtract(ndvi_before)
            .rename("NDVI")
            .addBands(
                ndwi_after.subtract(ndwi_before).rename("NDWI")
            )
        )

        stats = diff.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )

        ndvi_val = stats.get("NDVI")
        ndwi_val = stats.get("NDWI")

        ndvi_change = ndvi_val.getInfo() if ndvi_val else None
        ndwi_change = ndwi_val.getInfo() if ndwi_val else None

    # ---------- SAR (Sentinel-1) ----------
    sar_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(d_start), str(d_end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    sar_change = None
    if sar_col.size().gt(0):
        sar = sar_col.mean()
        sar_stats = sar.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )
        sar_val = sar_stats.get("VV")
        sar_change = sar_val.getInfo() if sar_val else None

    return {
        "NDVI Change": ndvi_change,
        "NDWI Change": ndwi_change,
        "SAR VV": sar_change
    }

# ======================================================
# RUN ANALYSIS
# ======================================================
st.divider()

if st.button("🔍 Analyze Damage", use_container_width=True):

    with st.spinner("Analyzing satellite data…"):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if not any(results.values()):
        st.error(
            "No usable satellite data found.\n\n"
            "This usually happens due to extreme cloud cover.\n"
            "SAR will be relied on in production."
        )
        st.stop()

    st.success("Analysis complete")

    col1, col2, col3 = st.columns(3)

    col1.metric("NDVI Change", results["NDVI Change"])
    col2.metric("NDWI Change", results["NDWI Change"])
    col3.metric("SAR VV", results["SAR VV"])

    # ---------- INTERPRETATION ----------
    if results["NDWI Change"] is not None and results["NDWI Change"] < -0.05:
        verdict = "🌊 Flood / Excess Water Detected"
    elif results["NDVI Change"] is not None and results["NDVI Change"] < -0.15:
        verdict = "🌾 Vegetation Damage Detected"
    elif results["SAR VV"] is not None:
        verdict = "📡 SAR signal available (cloud-independent)"
    else:
        verdict = "ℹ️ No strong damage signal"

    st.subheader("Assessment")
    st.write(verdict)
