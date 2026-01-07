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
    "Analysis Area Radius",
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
st.markdown("### Step 1: Baseline Crop Health (Before Damage)")

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
    st.error(
        "❌ *Timeline Error*\n\n"
        "Damage Assessment period must start AFTER the Baseline period ends."
    )
    st.stop()

# ----------------------------------
# DAMAGE ANALYSIS FUNCTION
# ----------------------------------
def analyze_damage(lat, lon,radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    def safe_ndvi(start, end):
        collection = (
            ee.ImageCollection("COPERNICUS/S2")
            .filterBounds(aoi)
            .filterDate(ee.Date(str(start)), ee.Date(str(end)))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .select(["B8", "B4"])
        )

        size = collection.size()

        def compute():
            image = collection.first()
            return image.normalizedDifference(["B8", "B4"])

        return ee.Algorithms.If(size.gt(0), compute(), None)

    ndvi_before = safe_ndvi(b_start, b_end)
    ndvi_after = safe_ndvi(d_start, d_end)

    if ndvi_before is None or ndvi_after is None:
        return None

    ndvi_change = ee.Image(ndvi_after).subtract(ee.Image(ndvi_before))

    mean_change = ndvi_change.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )

    return mean_change.get("nd").getInfo()

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
        "<i>This report is generated using satellite-derived NDVI analysis "
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
        ndvi_change = analyze_damage(
            lat, lon,radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if ndvi_change is None:
        st.error("❌ No usable satellite data found for this location/date range.")
        st.stop()

    # DAMAGE CLASSIFICATION
    if ndvi_change < -0.30:
        severity = "🔴 High Damage"
    elif ndvi_change < -0.15:
        severity = "🟠 Medium Damage"
    elif ndvi_change < -0.05:
        severity = "🟡 Low Damage"
    else:
        severity = "🟢 No Significant Damage"

    st.success("✅ Analysis Complete")

    col1, col2 = st.columns(2)
    col1.metric("Average NDVI Change", round(ndvi_change, 3))
    col2.metric("Damage Severity", severity)

    # ----------------------------------
    # PDF DOWNLOAD
    # ----------------------------------
    report_data = {
        "Latitude": lat,
        "Longitude": lon,
        "Baseline Period": f"{baseline_start} to {baseline_end}",
        "Damage Assessment Period": f"{damage_start} to {damage_end}",
        "Average NDVI Change": round(ndvi_change, 3),
        "Damage Severity": severity,
        "Satellite": "Sentinel-2"
    }

    pdf_file = generate_pdf(report_data)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Damage Verification Report (PDF)",
            data=f,
            file_name=pdf_file,
            mime="application/pdf"
        )
