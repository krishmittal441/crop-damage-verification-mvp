import ee
import streamlit as st
import json
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ----------------------------------
# INITIALIZE EARTH ENGINE (SERVICE ACCOUNT)
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
st.set_page_config(
    page_title="Crop Damage Verification",
    layout="centered"
)

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
    index=1,
    help="Smaller radius = more field-level accuracy"
)

st.divider()

# ----------------------------------
# DATE TIP
# ----------------------------------
st.info(
    "📌 *Date Selection Tip*\n\n"
    "• Keep *Baseline dates at least 7 days BEFORE the event*\n"
    "• Keep *Damage Assessment dates at least 7 days AFTER the event*\n\n"
    "This improves satellite accuracy and avoids cloud/noise issues."
)

# ----------------------------------
# BASELINE PERIOD
# ----------------------------------
st.markdown("### Step 1: Baseline Crop Health (Before Event)")

baseline_start = st.date_input("Baseline Start Date", value=datetime.date(2024, 6, 1))
baseline_end = st.date_input("Baseline End Date (Just Before Event)", value=datetime.date(2024, 6, 20))

st.divider()

# ----------------------------------
# DAMAGE PERIOD
# ----------------------------------
st.markdown("### Step 2: Damage Assessment Period (After Event)")

damage_start = st.date_input("Damage Assessment Start Date (After Event)", value=datetime.date(2024, 7, 1))
damage_end = st.date_input("Damage Assessment End Date", value=datetime.date(2024, 7, 20))

st.divider()

# ----------------------------------
# DATE VALIDATIONS
# ----------------------------------
if baseline_start >= baseline_end:
    st.error("❌ Baseline start date must be earlier than baseline end date.")
    st.stop()

if damage_start >= damage_end:
    st.error("❌ Damage assessment start date must be earlier than its end date.")
    st.stop()

if damage_start <= baseline_end:
    st.error("❌ Damage assessment must start AFTER baseline period.")
    st.stop()

# ----------------------------------
# DAMAGE ANALYSIS FUNCTION (NDVI + NDWI)
# ----------------------------------
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    def safe_index(start, end, bands, name):
        collection = (
            ee.ImageCollection("COPERNICUS/S2")
            .filterBounds(aoi)
            .filterDate(ee.Date(str(start)), ee.Date(str(end)))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .select(bands)
        )

        size = collection.size()

        def compute():
            image = collection.median()
            return image.normalizedDifference(bands).rename(name)

        return ee.Algorithms.If(size.gt(0), compute(), None)

    # NDVI
    ndvi_before = safe_index(b_start, b_end, ["B8", "B4"], "NDVI")
    ndvi_after = safe_index(d_start, d_end, ["B8", "B4"], "NDVI")

    # NDWI
    ndwi_before = safe_index(b_start, b_end, ["B3", "B8"], "NDWI")
    ndwi_after = safe_index(d_start, d_end, ["B3", "B8"], "NDWI")

    if any(x is None for x in [ndvi_before, ndvi_after, ndwi_before, ndwi_after]):
        return None

    ndvi_change = ee.Image(ndvi_after).subtract(ee.Image(ndvi_before))
    ndwi_change = ee.Image(ndwi_after).subtract(ee.Image(ndwi_before))

    stats = ee.Image.cat([ndvi_change, ndwi_change]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )

    return {
        "ndvi_change": stats.get("NDVI").getInfo(),
        "ndwi_change": stats.get("NDWI").getInfo()
    }

# ----------------------------------
# PDF GENERATION FUNCTION
# ----------------------------------
def generate_pdf(data):
    file_name = "Crop_Damage_Verification_Report.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Satellite-Based Crop Damage Verification Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for key, value in data.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<i>This report uses NDVI (vegetation) and NDWI (water) indices from Sentinel-2 satellite data "
        "and is intended to support insurance and government decision-making.</i>",
        styles["Italic"]
    ))

    doc.build(story)
    return file_name

# ----------------------------------
# RUN ANALYSIS
# ----------------------------------
st.markdown("### Run Damage Analysis")

if st.button("🔍 Analyze Damage", use_container_width=True):

    with st.spinner("Analyzing satellite data..."):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if results is None:
        st.error("❌ No usable satellite data found for this location/date range.")
        st.stop()

    ndvi_change = results["ndvi_change"]
    ndwi_change = results["ndwi_change"]

    # RULE-BASED INTERPRETATION
    if ndwi_change > 0.15:
        severity = "🔵 Flood / Waterlogging Detected"
        explanation = "Water-sensitive index shows presence of surface water or inundation."
    elif ndvi_change < -0.15:
        severity = "🟠 Vegetation Stress Detected"
        explanation = "Vegetation index indicates crop stress or damage after the event."
    else:
        severity = "🟢 No Significant Satellite-Visible Damage"
        explanation = "No strong vegetation loss or water signal detected."

    st.success("✅ Analysis Complete")

    col1, col2 = st.columns(2)
    col1.metric("NDVI Change", round(ndvi_change, 3))
    col2.metric("NDWI Change", round(ndwi_change, 3))

    st.info(f"Interpretation: {explanation}")

    # ----------------------------------
    # PDF DOWNLOAD
    # ----------------------------------
    report_data = {
        "Latitude": lat,
        "Longitude": lon,
        "Analysis Radius (km)": radius_km,
        "Baseline Period": f"{baseline_start} to {baseline_end}",
        "Damage Assessment Period": f"{damage_start} to {damage_end}",
        "NDVI Change": round(ndvi_change, 3),
        "NDWI Change": round(ndwi_change, 3),
        "Assessment Result": severity,
        "Interpretation": explanation,
        "Satellite Data": "Sentinel-2 (Optical)"
    }

    pdf_file = generate_pdf(report_data)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Damage Verification Report (PDF)",
            data=f,
            file_name=pdf_file,
            mime="application/pdf"
        )
