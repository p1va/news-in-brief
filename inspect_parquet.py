"""Parquet inspector for manual article review."""

import streamlit as st
import pyarrow.parquet as pq
from pathlib import Path

ARTIFACTS_DIR = Path("italy-today/artifacts")

st.set_page_config(page_title="Parquet Inspector", layout="wide")
st.title("Article Inspector")

# --- Sidebar: pick date and parquet file ---
dates = sorted([d.name for d in ARTIFACTS_DIR.iterdir() if d.is_dir()], reverse=True)
selected_date = st.sidebar.selectbox("Date", dates)

parquets = sorted((ARTIFACTS_DIR / selected_date).glob("*.parquet"))
labels = [p.name for p in parquets]
selected_file = st.sidebar.selectbox("File", labels)

path = ARTIFACTS_DIR / selected_date / selected_file
df = pq.read_table(path).to_pandas()

# Drop heavy columns that aren't useful for inspection
drop_cols = [c for c in df.columns if "embedding" in c.lower()]
df_display = df.drop(columns=drop_cols, errors="ignore")

# --- Sidebar: filters ---
st.sidebar.markdown("---")
sources = sorted(df_display["source"].unique())
selected_sources = st.sidebar.multiselect("Filter sources", sources)
if selected_sources:
    df_display = df_display[df_display["source"].isin(selected_sources)]

search = st.sidebar.text_input("Search titles")
if search:
    df_display = df_display[df_display["title"].str.contains(search, case=False, na=False)]

# --- Main area ---
st.caption(f"{len(df_display)} articles — {selected_date} / {selected_file}")
st.dataframe(
    df_display,
    use_container_width=True,
    height=700,
    column_config={
        "link": st.column_config.LinkColumn("link", display_text="open"),
    },
)
