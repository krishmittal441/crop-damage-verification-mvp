import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ----------------------------------
# INITIALIZE EARTH ENGINE
# ----------------------------------
service_account_info = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=st.secrets["EE_SERVICE_ACCOUNT"]
)

ee.Initialize(credentials)

# ----------------------------------
# STREAMLIT CONFIG
# ----------------------------------
st.set_page_config(page_title="Crop Damage Verification", layout="centered")

# ----------------------------------
# HEADER
# ----------------------------------
st.title("🛰️ Satellite-Based Crop Damage Verification")
st.subheader("Optical + Radar (SAR) based assessment")
st.divider()

# ----------------------------------
# LOCATION INPUT
# ----------------------------------
lat = st.number_input("Latitude", value=27.25, format="%.6f")
lon = st.number_input("Longitude", value=94.10, format="%.6f")

radius_km = st.selectbox(
    "Analysis Radius (km)",
    options=[0.5, 1, 2],
    index=0
)

st.divider()

# ----------------------------------
# DATE INPUTS (EVENT-CENTRIC)
# ----------------------------------
baseline_start = st.date_input("Baseline Start Date", datetime.date(2023, 6, 1))
baseline_end = st.date_input("Baseline End Date", datetime.date(2023, 6, 20))

damage_start = st.date_input("Event Window Start", datetime.date(2023, 7, 10))
damage_end = st.date_input("Event Window End", datetime.date(2023, 7, 18))

# ----------------------------------
# VALIDATION
# ----------------------------------
if baseline_start >= baseline_end:
    st.error("Invalid baseline dates")
    st.stop()

if damage_start >= damage_end:
    st.error("Invalid event dates")
    st.stop()

if damage_start <= baseline_end:
    st.error("Event must be after baseline")
    st.stop()

# ----------------------------------
# ANALYSIS FUNCTION
# ----------------------------------
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    # ---------- OPTICAL (Sentinel-2) ----------
    def optical_index(start, end, bands, name):
        col = (
            ee.ImageCollection("COPERNICUS/S2")
            .filterBounds(aoi)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .select(bands)
        )

        size = col.size()

        return ee.Algorithms.If(
            size.gt(0),
            ee.Image(col.sort("CLOUDY_PIXEL_PERCENTAGE").first())
            .normalizedDifference(bands)
            .rename(name),
            None
        )

    # ---------- SAR (Sentinel-1) ----------
    def sar_vv(start, end):
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .select("VV")
        )

        size = col.size()

        return ee.Algorithms.If(size.gt(0), col.mean(), None)

    try:
        # Optical indices
        ndvi_before = optical_index(b_start, b_end, ["B8", "B4"], "NDVI")
        ndvi_after = optical_index(d_start, d_end, ["B8", "B4"], "NDVI")

        ndwi_before = optical_index(b_start, b_end, ["B3", "B8"], "NDWI")
        ndwi_after = optical_index(d_start, d_end, ["B3", "B8"], "NDWI")

        # SAR windows (WIDER by design)
        sar_before = sar_vv(
            b_start - datetime.timedelta(days=10),
            b_end + datetime.timedelta(days=10)
        )

        sar_after = sar_vv(
            d_start - datetime.timedelta(days=10),
            d_end + datetime.timedelta(days=10)
        )

        images = []

        if ndvi_before and ndvi_after:
            images.append(ndvi_after.subtract(ndvi_before))

        if ndwi_before and ndwi_after:
            images.append(ndwi_after.subtract(ndwi_before))

        if sar_before and sar_after:
            images.append(sar_after.subtract(sar_before).rename("SAR"))

        if len(images) == 0:
            return None

        stats = ee.Image.cat(images).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )

        return {
            "ndvi": ee.Number(stats.get("NDVI", 0)).getInfo(),
            "ndwi": ee.Number(stats.get("NDWI", 0)).getInfo(),
            "sar": ee.Number(stats.get("SAR", 0)).getInfo()
        }

    except Exception:
        return None

# ----------------------------------
# PDF FUNCTION
# ----------------------------------
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

# ----------------------------------
# RUN ANALYSIS
# ----------------------------------
if st.button("🔍 Analyze Damage", use_container_width=True):

    with st.spinner("Analyzing satellite data (optical + SAR)..."):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if results is None:
        st.warning(
            "No usable optical or SAR data found for this area and period. "
            "This can happen due to orbit gaps or extreme cloud cover."
        )
        st.stop()

    ndvi = results["ndvi"]
    ndwi = results["ndwi"]
    sar = results["sar"]

    # ---------- DECISION LOGIC ----------
    if sar < -1.5:
        verdict = "🔵 Flood Detected (SAR-confirmed)"
        explanation = "Radar backscatter dropped significantly, indicating inundation."
    elif ndwi > 0.15:
        verdict = "🔵 Possible Waterlogging (Optical)"
        explanation = "NDWI indicates excess surface water."
    elif ndvi < -0.15:
        verdict = "🟠 Vegetation Stress Detected"
        explanation = "NDVI decline detected."
    else:
        verdict = "🟢 No Significant Satellite-Visible Damage"
        explanation = "No strong optical or radar damage signal."

    st.success("Analysis Complete")

    c1, c2, c3 = st.columns(3)
    c1.metric("NDVI Change", round(ndvi, 3))
    c2.metric("NDWI Change", round(ndwi, 3))
    c3.metric("SAR VV Change (dB)", round(sar, 2))

    st.info(explanation)

    pdf = generate_pdf({
        "Latitude": lat,
        "Longitude": lon,
        "Radius (km)": radius_km,
        "Baseline Period": f"{baseline_start} to {baseline_end}",
        "Event Period": f"{damage_start} to {damage_end}",
        "NDVI Change": round(ndvi, 3),
        "NDWI Change": round(ndwi, 3),
        "SAR VV Change (dB)": round(sar, 2),
        "Assessment": verdict,
        "Satellites Used": "Sentinel-2 (Optical), Sentinel-1 (SAR)"
    })

    with open(pdf, "rb") as f:
        st.download_button("Download Report (PDF)", f, pdf)
