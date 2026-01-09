import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ======================================================
# EARTH ENGINE INIT
# ======================================================
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])
credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)
ee.Initialize(credentials)

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="CropVerify – Phase 1", layout="centered")

st.title("🛰️ CropVerify – Phase 1")
st.subheader("Event-based Satellite Crop Damage Verification")
st.divider()

# ======================================================
# USER INPUTS
# ======================================================
event_type = st.selectbox(
    "Event Type",
    ["Flood", "Drought", "Cyclone"]
)

lat = st.number_input("Latitude", value=26.2, format="%.6f")
lon = st.number_input("Longitude", value=93.8, format="%.6f")

radius_km = st.selectbox("AOI Radius (km)", [1, 2, 5], index=1)

baseline_start = st.date_input("Baseline Start", datetime.date(2023, 6, 1))
baseline_end   = st.date_input("Baseline End", datetime.date(2023, 6, 20))

event_start = st.date_input("Event Start", datetime.date(2023, 7, 5))
event_end   = st.date_input("Event End", datetime.date(2023, 7, 30))

if baseline_start >= baseline_end or event_start >= event_end:
    st.error("Invalid date range")
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
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    val = stats.get(band)
    return val.getInfo() if val else None

# ======================================================
# EVENT-SPECIFIC ANALYSIS
# ======================================================
def analyze_flood(aoi):
    sar_b = sar_composite(aoi, baseline_start, baseline_end)
    sar_a = sar_composite(aoi, event_start, event_end)

    opt_b = optical_composite(aoi, baseline_start, baseline_end)
    opt_a = optical_composite(aoi, event_start, event_end)

    sar_before = safe_mean(sar_b, "VV", aoi) if sar_b else None
    sar_after  = safe_mean(sar_a, "VV", aoi) if sar_a else None
    sar_change = sar_after - sar_before if sar_before and sar_after else None

    ndwi_before = safe_mean(
        opt_b.normalizedDifference(["B3", "B8"]).rename("NDWI"), "NDWI", aoi
    ) if opt_b else None

    ndwi_after = safe_mean(
        opt_a.normalizedDifference(["B3", "B8"]).rename("NDWI"), "NDWI", aoi
    ) if opt_a else None

    ndwi_change = ndwi_after - ndwi_before if ndwi_before and ndwi_after else None

    if sar_change is not None and sar_change <= -3:
        assessment = "Open water flooding detected"
        confidence = "High"
    elif sar_change is not None and sar_change >= 1:
        assessment = "Flooded standing crops detected"
        confidence = "High"
    elif ndwi_change is not None and ndwi_change >= 0.15:
        assessment = "Surface water / waterlogging detected"
        confidence = "Medium"
    else:
        assessment = "No strong flood signal"
        confidence = "Low"

    explanation = (
        f"SAR VV changed by {sar_change:.2f} dB and NDWI changed by "
        f"{ndwi_change:.2f} indicating flood-related surface moisture."
        if sar_change is not None else "Insufficient SAR signal."
    )

    return {
        "SAR Before (dB)": sar_before,
        "SAR After (dB)": sar_after,
        "SAR Change (dB)": sar_change,
        "NDWI Before": ndwi_before,
        "NDWI After": ndwi_after,
        "NDWI Change": ndwi_change,
        "Assessment": assessment,
        "Confidence": confidence,
        "Explanation": explanation
    }

def analyze_drought(aoi):
    opt_b = optical_composite(aoi, baseline_start, baseline_end)
    opt_a = optical_composite(aoi, event_start, event_end)

    ndvi_b = safe_mean(opt_b.normalizedDifference(["B8", "B4"]).rename("NDVI"), "NDVI", aoi)
    ndvi_a = safe_mean(opt_a.normalizedDifference(["B8", "B4"]).rename("NDVI"), "NDVI", aoi)

    evi_b = safe_mean(
        opt_b.expression(
            "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
            {"NIR": opt_b.select("B8"), "RED": opt_b.select("B4"), "BLUE": opt_b.select("B2")}
        ).rename("EVI"),
        "EVI",
        aoi
    )

    evi_a = safe_mean(
        opt_a.expression(
            "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
            {"NIR": opt_a.select("B8"), "RED": opt_a.select("B4"), "BLUE": opt_a.select("B2")}
        ).rename("EVI"),
        "EVI",
        aoi
    )

    ndvi_change = ndvi_a - ndvi_b
    evi_change = evi_a - evi_b

    if ndvi_change <= -0.2 and evi_change <= -0.15:
        assessment = "Drought stress detected"
        confidence = "High"
    elif ndvi_change <= -0.1:
        assessment = "Early vegetation stress"
        confidence = "Medium"
    else:
        assessment = "No drought signal"
        confidence = "Low"

    return {
        "NDVI Before": ndvi_b,
        "NDVI After": ndvi_a,
        "NDVI Change": ndvi_change,
        "EVI Before": evi_b,
        "EVI After": evi_a,
        "EVI Change": evi_change,
        "Assessment": assessment,
        "Confidence": confidence,
        "Explanation": "Vegetation stress evaluated using NDVI and EVI trends."
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

    if sar_change and abs(sar_change) >= 2 and ndvi_change <= -0.2:
        assessment = "Cyclone damage likely"
        confidence = "High"
    elif ndvi_change <= -0.15:
        assessment = "Vegetation damage possible"
        confidence = "Medium"
    else:
        assessment = "No strong cyclone damage"
        confidence = "Low"

    return {
        "SAR Change (dB)": sar_change,
        "NDVI Change": ndvi_change,
        "Assessment": assessment,
        "Confidence": confidence,
        "Explanation": "Structural and vegetation damage assessed using SAR and NDVI."
    }

# ======================================================
# PDF GENERATION
# ======================================================
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

# ======================================================
# RUN ANALYSIS
# ======================================================
st.divider()

if st.button("🔍 Run Phase-1 Analysis", use_container_width=True):

    aoi = get_aoi(lat, lon, radius_km)

    with st.spinner("Running satellite analysis..."):
        if event_type == "Flood":
            results = analyze_flood(aoi)
        elif event_type == "Drought":
            results = analyze_drought(aoi)
        else:
            results = analyze_cyclone(aoi)

    st.success("Analysis complete")
    st.json(results)

    pdf = generate_pdf(results)
    with open(pdf, "rb") as f:
        st.download_button("Download PDF Report", f, pdf)
