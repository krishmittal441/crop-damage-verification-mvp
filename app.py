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
st.subheader("Objective: satellite-backed assessment for insurance & government use")
st.divider()

# ----------------------------------
# LOCATION INPUT
# ----------------------------------
st.markdown("### Location Details")

lat = st.number_input("Latitude", value=28.7041, format="%.6f")
lon = st.number_input("Longitude", value=77.1025, format="%.6f")

radius_km = st.selectbox(
    "Analysis Area Radius (km)",
    options=[0.5, 1, 2],
    index=1
)

st.divider()

# ----------------------------------
# DATE TIP
# ----------------------------------
st.info(
    "📌 Date Selection Tip\n\n"
    "• Baseline ≥ 7 days BEFORE event\n"
    "• Damage window ≥ 7 days AFTER event\n"
)

# ----------------------------------
# DATE INPUTS
# ----------------------------------
baseline_start = st.date_input("Baseline Start Date", datetime.date(2023, 6, 1))
baseline_end = st.date_input("Baseline End Date", datetime.date(2023, 6, 20))

damage_start = st.date_input("Damage Start Date", datetime.date(2023, 7, 10))
damage_end = st.date_input("Damage End Date", datetime.date(2023, 7, 18))

# ----------------------------------
# VALIDATIONS
# ----------------------------------
if baseline_start >= baseline_end:
    st.error("Baseline dates invalid")
    st.stop()

if damage_start >= damage_end:
    st.error("Damage dates invalid")
    st.stop()

if damage_start <= baseline_end:
    st.error("Damage period must be after baseline")
    st.stop()

# ----------------------------------
# ANALYSIS FUNCTION (NULL-SAFE)
# ----------------------------------
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    def compute_index(start, end, bands, name):
        col = (
            ee.ImageCollection("COPERNICUS/S2")
            .filterBounds(aoi)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .select(bands)
        )

        image = ee.Image(col.sort("CLOUDY_PIXEL_PERCENTAGE").first())
        return image.normalizedDifference(bands).rename(name)

    try:
        ndvi_before = compute_index(b_start, b_end, ["B8", "B4"], "NDVI")
        ndvi_after = compute_index(d_start, d_end, ["B8", "B4"], "NDVI")

        ndwi_before = compute_index(b_start, b_end, ["B3", "B8"], "NDWI")
        ndwi_after = compute_index(d_start, d_end, ["B3", "B8"], "NDWI")

        ndvi_change = ndvi_after.subtract(ndvi_before)
        ndwi_change = ndwi_after.subtract(ndwi_before)

        stats = ee.Image.cat([ndvi_change, ndwi_change]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )

        ndvi = ee.Number(stats.get("NDVI", 0))
        ndwi = ee.Number(stats.get("NDWI", 0))

        return {
            "ndvi_change": ndvi.getInfo(),
            "ndwi_change": ndwi.getInfo()
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
# RUN
# ----------------------------------
st.markdown("### Run Analysis")

if st.button("🔍 Analyze Damage", use_container_width=True):

    with st.spinner("Processing satellite data..."):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if results is None:
        st.error("Satellite data unavailable for selected dates/area.")
        st.stop()

    ndvi = results["ndvi_change"]
    ndwi = results["ndwi_change"]

    if ndwi > 0.15:
        severity = "🔵 Flood / Waterlogging Detected"
        explanation = "NDWI indicates surface water presence."
    elif ndvi < -0.15:
        severity = "🟠 Vegetation Stress Detected"
        explanation = "NDVI decline detected after event."
    else:
        severity = "🟢 No Significant Satellite-Visible Damage"
        explanation = "No strong vegetation or water signal."

    st.success("Analysis Complete")

    c1, c2 = st.columns(2)
    c1.metric("NDVI Change", round(ndvi, 3))
    c2.metric("NDWI Change", round(ndwi, 3))

    st.info(explanation)

    pdf = generate_pdf({
        "Latitude": lat,
        "Longitude": lon,
        "Radius (km)": radius_km,
        "Baseline Period": f"{baseline_start} to {baseline_end}",
        "Damage Period": f"{damage_start} to {damage_end}",
        "NDVI Change": round(ndvi, 3),
        "NDWI Change": round(ndwi, 3),
        "Assessment": severity,
        "Note": "Optical indices may miss floods; SAR recommended for confirmation."
    })

    with open(pdf, "rb") as f:
        st.download_button("Download PDF", f, pdf)
