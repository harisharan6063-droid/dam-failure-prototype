from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from transformer_model import FEATURE_COLUMNS, load_historical_records, predict_peak_outflow, train_historical_model

st.set_page_config(page_title="Dam Break Inundation Model", page_icon="🌊", layout="wide")

DEFAULT_WORKBOOK = Path(r"C:\Users\Welcome\OneDrive\Desktop\M.B.-G._M.T.-F.(2020)-DAM_FAILURES_DATABASE(V1).xlsx")


def illustrative_hydrograph(peak: float, formation_minutes: float) -> pd.DataFrame:
    """A visual aid, not a hydrodynamic simulation."""
    minutes = np.linspace(0, max(formation_minutes * 4, 60), 121)
    rise = np.clip(minutes / max(formation_minutes, 1), 0, 1)
    return pd.DataFrame({"Minutes after breach begins": minutes, "Illustrative discharge (m³/s)": peak * rise * np.exp(1 - rise)})


st.title("DAM FAILURE — HISTORICAL ML PROTOTYPE")
st.caption("One historical-data Transformer estimate per manually entered condition")
st.warning("No synthetic scenarios are generated. This workbook has no terrain grids or flood-depth time series, so this app does not claim to simulate inundation. The hydrograph is illustrative only, not HEC-RAS, SPH, or Delft3D output.")
uploaded = st.sidebar.file_uploader("Historical dam-failure workbook (.xlsx)", type="xlsx")
source = uploaded if uploaded is not None else (DEFAULT_WORKBOOK if DEFAULT_WORKBOOK.exists() else None)
if source is None:
    st.error("Upload the supplied .xlsx workbook in the sidebar to train the model.")
    st.stop()
try:
    records = load_historical_records(source)
except Exception as error:
    st.error(f"Could not read the DATABASE sheet: {error}")
    st.stop()
if "model" not in st.session_state or st.session_state.get("record_count") != len(records):
    with st.spinner("Training Transformer on complete historical records..."):
        st.session_state.model, st.session_state.x_scaler, st.session_state.y_scaler = train_historical_model(records)
        st.session_state.record_count = len(records)
st.success(f"Training source: {len(records)} complete historical records from the supplied workbook.")
with st.expander("Training fields and limitation"):
    st.write("Inputs: dam height, water volume stored, water height, breach average width. Target: observed peak outflow.")
    st.write("This is a small-data prototype—not a validated safety or engineering model.")
    st.dataframe(records.describe().T.round(2), use_container_width=True)
st.subheader("Enter one dam-break condition")
c1, c2, c3, c4, c5 = st.columns(5)
dam_height = c1.number_input("Dam Height (m)", min_value=0.1, value=50.0)
water_volume = c2.number_input("Water Volume Stored (MCM)", min_value=0.001, value=120.0)
water_height = c3.number_input("Water Height (m)", min_value=0.01, value=45.0)
breach_width = c4.number_input("Breach Average Width (m)", min_value=0.1, value=80.0)
formation_minutes = c5.number_input("Assumed Breach Formation (min)", min_value=1.0, value=30.0, help="Used only for the illustrative chart; it is not a model feature.")
if st.button("PREDICT PEAK OUTFLOW", type="primary"):
    values = dict(zip(FEATURE_COLUMNS, [dam_height, water_volume, water_height, breach_width]))
    st.session_state.peak = predict_peak_outflow(st.session_state.model, st.session_state.x_scaler, st.session_state.y_scaler, values)
    st.session_state.formation_minutes = formation_minutes
if "peak" in st.session_state:
    st.subheader("Historical Transformer prediction")
    st.metric("Predicted Peak Outflow", f"{st.session_state.peak:,.0f} m³/s")
    st.subheader("Illustrative breach hydrograph")
    st.line_chart(illustrative_hydrograph(st.session_state.peak, st.session_state.formation_minutes), x="Minutes after breach begins", y="Illustrative discharge (m³/s)")
st.subheader("For a real breach and inundation simulation")
st.markdown("Use this estimate only as an initial condition, then build a model from a site DEM, dam/channel geometry, reservoir conditions, breach progression, and downstream boundary conditions in [HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/) or [Delft3D](https://www.deltares.nl/en/software-and-data/products/delft3d). The spreadsheet alone cannot support a credible flood map.")
st.stop()


@st.cache_resource(show_spinner="Preparing prototype Transformer model…")
def get_model():
    return train_demo_model()


def flood_map(row: pd.Series, key: str):
    center = [20.5937, 78.9629]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
    area = float(row["flood_area_km2"])
    depth = float(row["max_depth_m"])
    radius = max(3000, np.sqrt(area / np.pi) * 1000)
    for fraction, color, opacity in [(1.0, "#2b83ba", 0.16), (0.68, "#00a6ca", 0.24), (0.38, "#fdae61", 0.34)]:
        folium.Circle(location=center, radius=radius * fraction, color=color, weight=1,
                      fill=True, fill_color=color, fill_opacity=opacity,
                      tooltip=f"Estimated depth zone · up to {depth * fraction:.1f} m").add_to(fmap)
    folium.Marker(center, tooltip="Dam location", icon=folium.Icon(color="red", icon="warning-sign")).add_to(fmap)
    st_folium(fmap, height=430, use_container_width=True, key=key, returned_objects=[])


st.title("DAM BREAK INUNDATION MODEL")
st.caption("Rapid scenario screening • prototype ML prediction layer")
st.info("Prototype transparency: the current Transformer is trained on synthetic, formula-derived labels—not SPH/Delft3D outputs. Hydraulic simulation integration is the next stage.")

with st.sidebar:
    st.header("Dam Parameters")
    dam_height = st.number_input("Dam Height (m)", 5.0, 300.0, 50.0)
    volume = st.number_input("Reservoir Volume (MCM)", 1.0, 10000.0, 120.0)
    water_level = st.number_input("Water Level (m)", 1.0, float(dam_height), min(45.0, float(dam_height)))
    breach_width = st.number_input("Breach Width (m)", 2.0, 500.0, 80.0)
    breach_time = st.number_input("Breach Time (min)", 2.0, 360.0, 30.0)
    generate = st.button("GENERATE SCENARIOS", type="primary", use_container_width=True)

base = {"dam_height_m": dam_height, "reservoir_volume_mcm": volume, "water_level_m": water_level, "breach_width_m": breach_width, "breach_time_min": breach_time}
if generate or "predictions" not in st.session_state:
    scenarios = generate_scenarios(base)
    DATA_PATH.parent.mkdir(exist_ok=True)
    scenarios.to_csv(DATA_PATH, index=False)
    model, x_scaler, y_scaler = get_model()
    st.session_state.predictions = predict(model, x_scaler, y_scaler, scenarios)

predictions = st.session_state.predictions
st.subheader(f"Generated Scenarios: {len(predictions)}")
table = predictions[["scenario", "water_level_m", "breach_width_m", "breach_time_min"]].copy()
table.columns = ["Scenario", "Water Level (m)", "Breach Width (m)", "Breach Time (min)"]
st.dataframe(table, use_container_width=True, hide_index=True, height=245)

st.subheader("Run Transformer Prediction")
chosen_id = st.selectbox("Select scenario", predictions["scenario"].tolist(), index=11)
chosen = predictions.loc[predictions["scenario"] == chosen_id].iloc[0]
metrics = st.columns(4)
metrics[0].metric("Maximum Depth", f"{chosen['max_depth_m']:.1f} m")
metrics[1].metric("Maximum Velocity", f"{chosen['max_velocity_mps']:.1f} m/s")
metrics[2].metric("Arrival Time", f"{chosen['arrival_time_min']:.0f} min")
metrics[3].metric("Estimated Flood Area", f"{chosen['flood_area_km2']:.1f} km²")

st.subheader("INUNDATION MAP")
flood_map(chosen, "main_map")

st.subheader("Scenario comparison")
defaults = [x for x in [12, 27, 41] if x <= len(predictions)]
compare_ids = st.multiselect("Compare scenarios", predictions["scenario"].tolist(), default=defaults, max_selections=3)
if compare_ids:
    comparison = predictions[predictions["scenario"].isin(compare_ids)]
    st.dataframe(comparison[["scenario", "max_depth_m", "max_velocity_mps", "arrival_time_min", "flood_area_km2"]].round(2), use_container_width=True, hide_index=True)
    cols = st.columns(len(compare_ids))
    for col, scenario_id in zip(cols, compare_ids):
        row = predictions.loc[predictions["scenario"] == scenario_id].iloc[0]
        with col:
            st.caption(f"Scenario {scenario_id}")
            flood_map(row, f"comparison_{scenario_id}")

st.caption("For decision support only; site-specific results require surveyed terrain, breach modeling, and validated hydrodynamic simulations.")
