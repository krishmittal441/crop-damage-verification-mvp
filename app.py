import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ======================================================
# EARTH ENGINE INITIALIZATION
# ======================================================
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])
credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)
ee.Initialize(credentials)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="CropVerify – Phase 1", layout="centered")

st.title("CropVerify – Phase 1")
st.caption(
    "Satellite-based, event-specific crop damage verification "
    "for government and agricultural insurance use"
)
st.markdown("---")

# ======================================================
# SECTION 1: EVENT & LOCATION
# ======================================================
st.subheader("1. Event & Location")

col1, col2 = st.columns(2)
with col1:
    event_type = st.selectbox(
        "Disaster Type",
        ["Flood", "Drought", "Cyclone"]
    )

with col2:
    radius_km = st.selectbox(
        "Analysis Radius (km)",
        [1, 2, 5],
        index=1,
        help="Larger radius improves satellite reliability"
    )

col3, col4 = st.columns(2)
with col3:
    lat = st.number_input("Latitude", value=26.2, format="%.6f")
with col4:
    lon = st.number_input("Longitude", value=93.8, format="%.6f")

# ======================================================
# SECTION 2: TIME PERIODS
# ======================================================
st.subheader("2. Time Periods")

st.info(
    "Baseline = normal conditions before the event.\n"
    "Event period = disaster impact window.\n"
    "For floods and cyclones, wider windows improve accuracy."
)

col1, col2 = st.columns(2)
with col1:
    baseline_start = st.date_input("Baseline Start", datetime.date(2023, 6, 1))
    baseline_end   = st.date_input("Baseline End", datetime.date(2023, 6, 20))
with col2:
    event_start = st.date_input("Event Start", datetime.date(2023, 7, 5))
    event_end   = st.date_input("Event End", datetime.date(2023, 7, 30))

if baseline_start >= baseline_end or event_start >= event_end:
    st.error("Invalid date ranges selected.")
    st.stop()

# ======================================================
# COMMON HELPERS
# ======================================================
def get_aoi(lat, lon, radius_km):
    return ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

def optical_composite(aoi, start, end):
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
    )
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), None))

def sar_composite(aoi, start, end):
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.mean(), None))

def safe_mean(image, band, aoi):
    if image is None:
        return None
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    val = stats.get(band)
    return val.getInfo() if val else None

# ======================================================
# EVENT ANALYSIS FUNCTIONS
# ======================================================
def analyze_flood(aoi):
    sar_b = sar_composite(aoi, baseline_start, baseline_end)
    sar_a = sar_composite(aoi, event_start, event_end)
    opt_b = optical_composite(aoi, baseline_start, baseline_end)
    opt_a = optical_composite(aoi, event_start, event_end)

    sar_before = safe_mean(sar_b, "VV", aoi)
    sar_after  = safe_mean(sar_a, "VV", aoi)
    sar_change = sar_after - sar_before if sar_before is not None and sar_after is not None else None

    ndwi_before = safe_mean(
        opt_b.normalizedDifference(["B3", "B8"]).rename("NDWI"), "NDWI", aoi
    ) if opt_b else None

    ndwi_after = safe_mean(
        opt_a.normalizedDifference(["B3", "B8"]).rename("NDWI"), "NDWI", aoi
    ) if opt_a else None

    ndwi_change = ndwi_after - ndwi_before if ndwi_before is not None and ndwi_after is not None else None

    if sar_change is not None and sar_change <= -3:
        assessment = "Open water flooding detected"
        confidence = "High"
    elif ndwi_change is not None and ndwi_change >= 0.15:
        assessment = "Surface water / waterlogging detected"
        confidence = "Medium"
    else:
        assessment = "No strong flood signal detected"
        confidence = "Low"

    return {
        "Event": "Flood",
        "SAR Before (dB)": sar_before,
        "SAR After (dB)": sar_after,
        "SAR Change (dB)": sar_change,
        "NDWI Before": ndwi_before,
        "NDWI After": ndwi_after,
        "NDWI Change": ndwi_change,
        "Assessment": assessment,
        "Confidence": confidence,
        "Explanation": "Flood assessment based on SAR backscatter and NDWI moisture signal."
    }

def analyze_drought(aoi):
    opt_b = optical_composite(aoi, baseline_start, baseline_end)
    opt_a = optical_composite(aoi, event_start, event_end)

    ndvi_before = safe_mean(
        opt_b.normalizedDifference(["B8", "B4"]).rename("NDVI"), "NDVI", aoi
    )
    ndvi_after = safe_mean(
        opt_a.normalizedDifference(["B8", "B4"]).rename("NDVI"), "NDVI", aoi
    )

    ndvi_change = ndvi_after - ndvi_before if ndvi_before is not None and ndvi_after is not None else None

    if ndvi_change is not None and ndvi_change <= -0.2:
        assessment = "Drought stress detected"
        confidence = "High"
    elif ndvi_change is not None and ndvi_change <= -0.1:
        assessment = "Early vegetation stress"
        confidence = "Medium"
    else:
        assessment = "No drought signal detected"
        confidence = "Low"

    return {
        "Event": "Drought",
        "NDVI Before": ndvi_before,
        "NDVI After": ndvi_after,
        "NDVI Change": ndvi_change,
        "Assessment": assessment,
        "Confidence": confidence,
        "Explanation": "Vegetation stress assessed using NDVI trend over time."
    }

def analyze_cyclone(aoi):
    sar_b = sar_composite(aoi, baseline_start, baseline_end)
    sar_a = sar_composite(aoi, event_start, event_end)
    opt_b = optical_composite(aoi, baseline_start, baseline_end)
    opt_a = optical_composite(aoi, event_start, event_end)

    sar_change = (
        safe_mean(sar_a, "VV", aoi) - safe_mean(sar_b, "VV", aoi)
        if sar_a and sar_b else None
    )

    ndvi_change = (
        safe_mean(opt_a.normalizedDifference(["B8", "B4"]).rename("NDVI"), "NDVI", aoi)
        - safe_mean(opt_b.normalizedDifference(["B8", "B4"]).rename("NDVI"), "NDVI", aoi)
    )

    if sar_change is not None and abs(sar_change) >= 2 and ndvi_change <= -0.2:
        assessment = "Cyclone-related crop damage likely"
        confidence = "High"
    elif ndvi_change <= -0.15:
        assessment = "Vegetation damage possible"
        confidence = "Medium"
    else:
        assessment = "No strong cyclone damage detected"
        confidence = "Low"

    return {
        "Event": "Cyclone",
        "SAR Change (dB)": sar_change,
        "NDVI Change": ndvi_change,
        "Assessment": assessment,
        "Confidence": confidence,
        "Explanation": "Structural and vegetation damage assessed using SAR and NDVI."
    }

# ======================================================
# RUN ANALYSIS
# ======================================================
st.markdown("---")
run = st.button("Run Satellite Damage Assessment", use_container_width=True)

if run:
    aoi = get_aoi(lat, lon, radius_km)

    with st.spinner("Processing satellite data..."):
        if event_type == "Flood":
            results = analyze_flood(aoi)
        elif event_type == "Drought":
            results = analyze_drought(aoi)
        else:
            results = analyze_cyclone(aoi)

    st.subheader("3. Summary Results")

    if event_type == "Flood":
        c1, c2, c3 = st.columns(3)
        c1.metric("SAR VV Change (dB)", round(results["SAR Change (dB)"], 2) if results["SAR Change (dB)"] is not None else "NA")
        c2.metric("NDWI Change", round(results["NDWI Change"], 2) if results["NDWI Change"] is not None else "NA")
        c3.metric("Confidence", results["Confidence"])

    elif event_type == "Drought":
        c1, c2, c3 = st.columns(3)
        c1.metric("NDVI Before", round(results["NDVI Before"], 2) if results["NDVI Before"] is not None else "NA")
        c2.metric("NDVI After", round(results["NDVI After"], 2) if results["NDVI After"] is not None else "NA")
        c3.metric("NDVI Change", round(results["NDVI Change"], 2) if results["NDVI Change"] is not None else "NA")
        st.caption(f"Confidence: {results['Confidence']}")

    else:  # Cyclone
        c1, c2, c3 = st.columns(3)
        c1.metric("SAR VV Change (dB)", round(results["SAR Change (dB)"], 2) if results["SAR Change (dB)"] is not None else "NA")
        c2.metric("NDVI Change", round(results["NDVI Change"], 2) if results["NDVI Change"] is not None else "NA")
        c3.metric("Confidence", results["Confidence"])

    st.subheader("4. Assessment")
    st.success(results["Assessment"])
    st.write(results["Explanation"])

    with st.expander("View Technical Details"):
        st.json(results)

    st.subheader("5. Official Report")

    def generate_pdf(results):
        file_name = "CropVerify_Phase1_Report.pdf"
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Crop Damage Verification Report</b>", styles["Title"]))
        story.append(Spacer(1, 12))

        for k, v in results.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
            story.append(Spacer(1, 6))

        doc.build(story)
        return file_name

    pdf = generate_pdf(results)
    with open(pdf, "rb") as f:
        st.download_button(
            "Download Damage Verification Report (PDF)",
            f,
            file_name=pdf,
            use_container_width=True
        )
