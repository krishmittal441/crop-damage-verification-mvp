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
st.subheader("Optical + Radar-based assessment for government & insurance use")
st.divider()

# ----------------------------------
# LOCATION INPUT
# ----------------------------------
st.markdown("### Location Details")

lat = st.number_input("Latitude", value=27.25, format="%.6f")
lon = st.number_input("Longitude", value=94.10, format="%.6f")

radius_km = st.selectbox(
    "Analysis Area Radius (km)",
    options=[0.5, 1, 2],
    index=0
)

st.divider()

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
# ANALYSIS FUNCTION (OPTICAL + SAR)
# ----------------------------------
def analyze_damage(lat, lon, radius_km, b_start, b_end, d_start, d_end):

    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    # ---------- NDVI / NDWI (Sentinel-2) ----------
    def optical_index(start, end, bands, name):
        col = (
            ee.ImageCollection("COPERNICUS/S2")
            .filterBounds(aoi)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
            .select(bands)
        )

        image = col.sort("CLOUDY_PIXEL_PERCENTAGE").first()
        return ee.Image(image).normalizedDifference(bands).rename(name)

    # ---------- SAR FLOOD (Sentinel-1) ----------
    def sar_vv(start, end):
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV")
        )

        return col.mean()

    try:
        # Optical
        ndvi_before = optical_index(b_start, b_end, ["B8", "B4"], "NDVI")
        ndvi_after = optical_index(d_start, d_end, ["B8", "B4"], "NDVI")

        ndwi_before = optical_index(b_start, b_end, ["B3", "B8"], "NDWI")
        ndwi_after = optical_index(d_start, d_end, ["B3", "B8"], "NDWI")

        ndvi_change = ndvi_after.subtract(ndvi_before)
        ndwi_change = ndwi_after.subtract(ndwi_before)

        # SAR
        sar_before = sar_vv(b_start, b_end)
        sar_after = sar_vv(d_start, d_end)

        sar_change = sar_after.subtract(sar_before)

        stats = ee.Image.cat([
            ndvi_change,
            ndwi_change,
            sar_change.rename("SAR")
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13
        )

        ndvi = ee.Number(stats.get("NDVI", 0)).getInfo()
        ndwi = ee.Number(stats.get("NDWI", 0)).getInfo()
        sar = ee.Number(stats.get("SAR", 0)).getInfo()

        return {
            "ndvi": ndvi,
            "ndwi": ndwi,
            "sar": sar
        }

    except Exception:
        return None

# ----------------------------------
# PDF FUNCTION
# ----------------------------------
def generate_pdf(data):
    file_name = "Crop_Damage_Verification_Report.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Satellite-Based Crop Damage Verification Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for k, v in data.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<i>This report uses optical (Sentinel-2) and radar (Sentinel-1 SAR) satellite data. "
        "SAR is used to detect flooding during cloud-covered periods.</i>",
        styles["Italic"]
    ))

    doc.build(story)
    return file_name

# ----------------------------------
# RUN ANALYSIS
# ----------------------------------
st.markdown("### Run Damage Analysis")

if st.button("🔍 Analyze Damage", use_container_width=True):

    with st.spinner("Analyzing optical + radar satellite data..."):
        results = analyze_damage(
            lat, lon, radius_km,
            baseline_start, baseline_end,
            damage_start, damage_end
        )

    if results is None:
        st.error("Satellite data unavailable.")
        st.stop()

    ndvi = results["ndvi"]
    ndwi = results["ndwi"]
    sar = results["sar"]

    # ---------- RULE-BASED DECISION ----------
    if sar < -1.5:
        severity = "🔵 Flood Detected (SAR-confirmed)"
        explanation = "Radar backscatter dropped significantly, indicating surface water or inundation."
    elif ndwi > 0.15:
        severity = "🔵 Possible Waterlogging (Optical)"
        explanation = "Water-sensitive index indicates excess surface water."
    elif ndvi < -0.15:
        severity = "🟠 Vegetation Stress Detected"
        explanation = "Vegetation index shows decline after event."
    else:
        severity = "🟢 No Significant Satellite-Visible Damage"
        explanation = "No strong optical or radar damage signal detected."

    st.success("Analysis Complete")

    c1, c2, c3 = st.columns(3)
    c1.metric("NDVI Change", round(ndvi, 3))
    c2.metric("NDWI Change", round(ndwi, 3))
    c3.metric("SAR VV Change (dB)", round(sar, 2))

    st.info(explanation)

    pdf = generate_pdf({
        "Latitude": lat,
        "Longitude": lon,
        "Analysis Radius (km)": radius_km,
        "Baseline Period": f"{baseline_start} to {baseline_end}",
        "Damage Period": f"{damage_start} to {damage_end}",
        "NDVI Change": round(ndvi, 3),
        "NDWI Change": round(ndwi, 3),
        "SAR VV Change (dB)": round(sar, 2),
        "Assessment Result": severity,
        "Interpretation": explanation,
        "Satellites Used": "Sentinel-2 (Optical), Sentinel-1 (SAR)"
    })

    with open(pdf, "rb") as f:
        st.download_button("Download Report (PDF)", f, pdf)
