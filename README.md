# Dam Break Inundation Model — Prototype

An interactive Streamlit demonstration that trains on the supplied historical dam-failure workbook.

## What this prototype does

- Uses no generated scenarios or formula-derived labels.
- Trains a small Transformer on complete records from the workbook's `DATABASE` sheet.
- Accepts one manually entered dam condition and estimates historical peak outflow.
- Shows an explicitly labelled illustrative breach hydrograph.

## Important data note

The model is trained only on observed historical records: dam height, stored water volume, water height and breach average width as inputs, with observed peak outflow as the target. The workbook has only 45 complete records for this mapping and has no terrain grids or flood-depth time series. It is not an SPH, Delft3D or HEC-RAS simulation, and is not suitable for engineering or operational decisions.

## Setup and run

```bash
pip install streamlit numpy pandas scikit-learn torch folium streamlit-folium
python -m streamlit run app.py
```

Run the commands from this folder (`dam_prototype`). The app reads the specified workbook from your Desktop or lets you upload it in the sidebar.
