import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------------------
# EARTH ENGINE INIT (SERVICE ACCOUNT)
# -------------------------------
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])
credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)
ee.Initialize(credentials)

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="Crop Damage Verification", layout="centered")

st.title("🛰️ Satellite-Based Crop Damage Verification")
st.caption("Event-based, index-level assessment for government & insurance use")
st.divider()

# -------------------------------
# EVENT TYPE
# -------------------------------
event_type = st.selectbox(
    "Select Event Type",
    ["Flood", "Drought", "Cyclone"],
    help="Event type affects interpretation, not raw values"
)

# -------------------------------
# LOCATION + AOI
# -------------------------------
st.subheader("Location")
lat = st.number_input("Latitude", value=26.2006, format="%.6f")
lon = st.number_input("Longitude", value=92.9376, format="%.6f")

radius_km = st.selectbox(
    "AOI Radius (km)",
    [0.5, 1, 2],
    index=1
)

st.divider()

# -------------------------------
# DATE INPUTS
# -------------------------------
st.subheader("Dates")

baseline_start = st.date_input("Baseline Start (Before Event)", datetime.date(2023, 6, 1))
baseline_end   = st.date_input("Baseline End (Before Event)", datetime.date(2023, 6, 20))

damage_start = st.date_input("Event / Damage Start", datetime.date(2023, 7, 1))
damage_end   = st.date_input("Damage End (After Event)", datetime.date(2023, 7, 20))

# -------------------------------
# ANALYSIS FUNCTION
# -------------------------------
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):
    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    # ---------- NDVI ----------
    def get_ndvi(start, end):
        col = ee.ImageCollection("COPERNICUS/S2") \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)) \
            .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        return col.median()

    # ---------- NDWI ----------
    def get_ndwi(start, end):
        col = ee.ImageCollection("COPERNICUS/S2") \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)) \
            .map(lambda img: img.normalizedDifference(["B3", "B8"]).rename("NDWI"))
        return col.median()

    # ---------- SAR VV ----------
    def get_sar(start, end):
        col = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
            .select("VV")
        return col.median()

    ndvi_before = get_ndvi(b_start, b_end)
    ndvi_after  = get_ndvi(d_start, d_end)

    ndwi_before = get_ndwi(b_start, b_end)
    ndwi_after  = get_ndwi(d_start, d_end)

    sar_before = get_sar(b_start, b_end)
    sar_after  = get_sar(d_start, d_end)

    stats = ee.Image.cat([
        ndvi_after.subtract(ndvi_before).rename("NDVI_CHANGE"),
        ndwi_after.subtract(ndwi_before).rename("NDWI_CHANGE"),
        sar_after.subtract(sar_before).rename("SAR_VV_CHANGE")
    ]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )

    return stats.getInfo()

# -------------------------------
# RUN ANALYSIS
# -------------------------------
if st.button("🔍 Run Satellite Damage Assessment", use_container_width=True):

    with st.spinner("Processing satellite data..."):
        results = analyze_damage(
            lat, lon, radius_km,
            str(baseline_start), str(baseline_end),
            str(damage_start), str(damage_end)
        )

    if not results:
        st.error("No usable satellite data found.")
        st.stop()

    ndvi = results.get("NDVI_CHANGE")
    ndwi = results.get("NDWI_CHANGE")
    sar  = results.get("SAR_VV_CHANGE")

    # -------------------------------
    # RAW VALUES (NEVER HIDDEN)
    # -------------------------------
    st.subheader("Raw Index Values")

    c1, c2, c3 = st.columns(3)
    c1.metric("NDVI Change", round(ndvi, 3))
    c2.metric("NDWI Change", round(ndwi, 3))
    c3.metric("SAR VV Change (dB)", round(sar, 2))

    st.divider()

    # -------------------------------
    # EVENT-SPECIFIC INTERPRETATION
    # -------------------------------
    st.subheader("Assessment")

    assessment = []

    if event_type == "Flood":
        if ndwi > 0.15 and sar < -1.5:
            assessment.append("🌊 Flood / excess water signal detected")
        else:
            assessment.append("No strong flood signal detected")

    if event_type == "Drought":
        if ndvi < -0.15:
            assessment.append("🌾 Vegetation stress consistent with drought")
        else:
            assessment.append("No drought signal detected")

    if event_type == "Cyclone":
        if ndvi < -0.2 and sar < -1:
            assessment.append("🌀 Structural + vegetation damage detected")
        else:
            assessment.append("No strong cyclone damage signal")

    for a in assessment:
        st.info(a)

    # -------------------------------
    # PDF EXPORT
    # -------------------------------
    def generate_pdf(data):
        file_name = "Crop_Damage_Report.pdf"
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Crop Damage Verification Report</b>", styles["Title"]))
        story.append(Spacer(1, 12))

        for k, v in data.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
            story.append(Spacer(1, 6))

        doc.build(story)
        return file_name

    pdf_data = {
        "Event Type": event_type,
        "Latitude": lat,
        "Longitude": lon,
        "AOI Radius (km)": radius_km,
        "NDVI Change": ndvi,
        "NDWI Change": ndwi,
        "SAR VV Change (dB)": sar
    }

    pdf_file = generate_pdf(pdf_data)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Report (PDF)",
            f,
            file_name=pdf_file,
            mime="application/pdf"
        )
