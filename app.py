from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# GREENGUARD AI — DNCC ENVIRONMENTAL INTELLIGENCE DASHBOARD
# ============================================================
st.set_page_config(
    page_title="GreenGuard AI | DNCC",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "outputs" / "greenguard_dashboard_data.gpkg"
CLUSTER_PATH = ROOT / "data" / "processed" / "heat_pattern_clusters_2014_2025.csv"
TREND_PATH = ROOT / "data" / "processed" / "grid_temperature_trends_2014_2025.csv"
YEARLY_PATH = ROOT / "outputs" / "temperature_trend_summary.csv"
FORECAST_PATH = ROOT / "outputs" / "grid_lst_forecasts.csv"
FORECAST_METRICS_PATH = ROOT / "outputs" / "lst_forecast_backtest_metrics.csv"

DARK_BG = "#071D16"
PANEL_BG = "#102B21"
CARD_BG = "#15372A"
BORDER = "#285441"
TEXT = "#F1F7F2"
MUTED = "#B7CBBE"
EMERALD = "#42D392"

HEAT_COLORS = {"Persistent hotspot": "#E05252", "Warm area": "#F2A541", "Relatively cooler": "#45B97C", "Not assessed": "#8A9690"}
VEGETATION_COLORS = {"Low vegetation": "#E05252", "Moderate vegetation": "#F2A541", "High vegetation": "#45B97C", "Not assessed": "#8A9690"}
BUILTUP_COLORS = {"Lower built-up signal": "#45B97C", "Moderate built-up signal": "#F2A541", "Higher built-up signal": "#E05252", "Not assessed": "#8A9690"}
PRIORITY_COLORS = {"Very Low": "#287D50", "Low": "#66B77A", "Moderate": "#E6C84F", "High": "#F39B45", "Very High": "#D94B4B", "Not assessed": "#8A9690"}
CLUSTER_COLORS = {"Lower-heat pattern": "#45B97C", "Higher-heat pattern": "#E05252", "Not assessed": "#8A9690"}
TREND_COLORS = {"Increasing": "#E05252", "Relatively stable": "#F2A541", "Decreasing": "#45B97C", "Insufficient observations": "#8A9690", "Not assessed": "#8A9690"}
FORECAST_RISK_COLORS = {"Lower": "#45B97C", "Moderate": "#F2A541", "High": "#F39B45", "Very High": "#D94B4B", "Not assessed": "#8A9690"}
LAYER_OPTIONS = ["Persistent historical heat", "Green-planning priority", "Vegetation Intervention Priority", "AI Heat Pattern Clusters", "Vegetation (NDVI)", "Built-up indicator (NDBI)", "Assessment coverage", "Historical temperature trend", "1-Year Forecast Heat Risk", "2-Year Forecast Heat Risk"]

# ------------------------------ Styling ------------------------------
st.markdown(f"""
<style>
.stApp {{ background: {DARK_BG}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: rgba(7,29,22,.96); }}
[data-testid="stSidebar"] {{ background:#0B251B; border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color:{TEXT}; }}
h1,h2,h3,h4 {{ color:{TEXT}!important; }}
p,label,.stMarkdown,[data-testid="stCaptionContainer"] {{ color:{MUTED}; }}
.hero {{ background:linear-gradient(120deg,#123D2A 0%,#0D2B20 65%,#174B32 100%); border:1px solid #2C6949; border-radius:20px; padding:28px 30px; margin:4px 0 22px; box-shadow:0 8px 26px rgba(0,0,0,.18); }}
.hero-title {{ color:#F3FFF6; font-size:32px; font-weight:750; letter-spacing:-.6px; margin-bottom:8px; }}
.hero-subtitle {{ color:#C8E5D2; font-size:15px; line-height:1.7; margin:0; }}
.section-card {{ background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:16px; padding:18px 20px; margin-bottom:12px; }}
.metric-card {{ background:linear-gradient(145deg,{CARD_BG},#10281E); border:1px solid #32654A; border-radius:16px; padding:20px 18px; min-height:190px; box-shadow:0 5px 16px rgba(0,0,0,.12); }}
.metric-label {{ color:#C4DCCB; font-size:14px; font-weight:650; letter-spacing:.4px; text-transform:uppercase; margin-bottom:12px; }}
.metric-value {{ color:#6BE0A1; font-size:30px; font-weight:750; line-height:1.2; margin-bottom:12px; overflow-wrap:anywhere; }}
.metric-description {{ color:#D4E5D9; font-size:13px; line-height:1.65; }}
.small-note {{ color:#B7CBBE; font-size:13px; line-height:1.65; }}
div[data-testid="stMetric"] {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; }}
div[data-testid="stMetric"] label {{ color:#C4DCCB!important; }}
div[data-testid="stMetricValue"] {{ color:#6BE0A1!important; }}
div[data-testid="stSelectbox"] label,div[data-testid="stRadio"] label {{ color:{TEXT}!important; }}
div[data-baseweb="select"]>div {{ background:#15372A; border-color:#32654A; }}
div[data-testid="stAlert"] {{ border-radius:12px; }}
hr {{ border-color:{BORDER}; }}
#MainMenu,footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------ Helpers ------------------------------
def first_existing_column(data, candidates):
    return next((c for c in candidates if c in data.columns), None)


def to_numeric(data, column):
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def format_number(value, digits=3):
    try:
        return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def get_assessed_mask(data):
    if "assessment_status" in data.columns:
        status = data["assessment_status"].astype(str).str.strip().str.lower()
        unassessed = {"not assessed", "unassessed", "unassessed - low coverage", "low coverage", "false", "0", "nan", "none"}
        return ~status.isin(unassessed)
    if "persistent_heat_score" in data.columns:
        return pd.to_numeric(data["persistent_heat_score"], errors="coerce").notna()
    return pd.Series(True, index=data.index)


def add_relative_categories(data):
    data = data.copy()
    for col in ["persistent_heat_score", "median_ndvi", "median_ndbi"]:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = to_numeric(data, col)
    assessed = get_assessed_mask(data)

    data["heat_category"] = "Not assessed"
    valid = assessed & data["persistent_heat_score"].notna()
    if valid.any():
        rank = data.loc[valid, "persistent_heat_score"].rank(method="average", pct=True)
        data.loc[rank.index[rank <= 1/3], "heat_category"] = "Relatively cooler"
        data.loc[rank.index[(rank > 1/3) & (rank <= 2/3)], "heat_category"] = "Warm area"
        data.loc[rank.index[rank > 2/3], "heat_category"] = "Persistent hotspot"

    data["vegetation_category"] = "Not assessed"
    valid = assessed & data["median_ndvi"].notna()
    if valid.any():
        rank = data.loc[valid, "median_ndvi"].rank(method="average", pct=True)
        data.loc[rank.index[rank <= 1/3], "vegetation_category"] = "Low vegetation"
        data.loc[rank.index[(rank > 1/3) & (rank <= 2/3)], "vegetation_category"] = "Moderate vegetation"
        data.loc[rank.index[rank > 2/3], "vegetation_category"] = "High vegetation"

    data["builtup_category"] = "Not assessed"
    valid = assessed & data["median_ndbi"].notna()
    if valid.any():
        rank = data.loc[valid, "median_ndbi"].rank(method="average", pct=True)
        data.loc[rank.index[rank <= 1/3], "builtup_category"] = "Lower built-up signal"
        data.loc[rank.index[(rank > 1/3) & (rank <= 2/3)], "builtup_category"] = "Moderate built-up signal"
        data.loc[rank.index[rank > 2/3], "builtup_category"] = "Higher built-up signal"

    if "priority_category" not in data.columns:
        data["priority_category"] = "Not assessed"
    data["priority_category"] = data["priority_category"].fillna("Not assessed").astype(str)
    data.loc[~assessed, "priority_category"] = "Not assessed"
    data.loc[~data["priority_category"].isin(PRIORITY_COLORS), "priority_category"] = "Not assessed"
    return data


def add_intervention_priority(data):
    """Calculate a relative vegetation-intervention shortlist score."""
    data = data.copy()
    for col in ["persistent_heat_score", "median_ndvi", "median_ndbi"]:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    assessed = get_assessed_mask(data)
    heat_pct = data["persistent_heat_score"].where(assessed).rank(method="average", pct=True)
    ndvi_pct = data["median_ndvi"].where(assessed).rank(method="average", pct=True)
    ndbi_pct = data["median_ndbi"].where(assessed).rank(method="average", pct=True)
    components = pd.DataFrame({"heat": heat_pct, "vegetation_deficit": 1.0 - ndvi_pct, "builtup": ndbi_pct}, index=data.index)
    weights = pd.Series({"heat": 0.50, "vegetation_deficit": 0.30, "builtup": 0.20})
    available_weights = components.notna().mul(weights, axis=1).sum(axis=1)
    score = components.mul(weights, axis=1).sum(axis=1, min_count=1).div(available_weights.where(available_weights > 0)) * 100
    eligible = assessed & data["persistent_heat_score"].notna() & data["median_ndvi"].notna()
    data["vegetation_intervention_score"] = score.where(eligible)
    data["vegetation_intervention_category"] = "Not assessed"
    valid = data.loc[eligible, "vegetation_intervention_score"].dropna()
    if not valid.empty:
        pct = valid.rank(method="average", pct=True)
        data.loc[pct.index[pct <= .20], "vegetation_intervention_category"] = "Very Low"
        data.loc[pct.index[(pct > .20) & (pct <= .40)], "vegetation_intervention_category"] = "Low"
        data.loc[pct.index[(pct > .40) & (pct <= .60)], "vegetation_intervention_category"] = "Moderate"
        data.loc[pct.index[(pct > .60) & (pct <= .80)], "vegetation_intervention_category"] = "High"
        data.loc[pct.index[pct > .80], "vegetation_intervention_category"] = "Very High"
    return data


def intervention_reason(row):
    reasons = []
    if row.get("heat_category") == "Persistent hotspot": reasons.append("persistent heat")
    elif row.get("heat_category") == "Warm area": reasons.append("moderate-to-high heat")
    if row.get("vegetation_category") == "Low vegetation": reasons.append("low vegetation signal")
    if row.get("builtup_category") == "Higher built-up signal": reasons.append("higher built-up signal")
    return ", ".join(reasons).capitalize() if reasons else "Combined relative indicators"


def get_cluster_category(data):
    col = first_existing_column(data, ["cluster", "cluster_label", "cluster_id", "kmeans_cluster", "heat_cluster"])
    if col is None:
        return pd.Series("Not assessed", index=data.index)
    def label(value):
        if pd.isna(value): return "Not assessed"
        s = str(value).strip().lower()
        if s in {"0", "cluster 0", "0.0"} or "lower" in s or "cool" in s: return "Lower-heat pattern"
        if s in {"1", "cluster 1", "1.0"} or "higher" in s or "hot" in s: return "Higher-heat pattern"
        return f"Cluster {value}"
    return data[col].apply(label)


def metric_card(title, value, description):
    st.markdown(f'''<div class="metric-card"><div class="metric-label">{title}</div><div class="metric-value">{value}</div><div class="metric-description">{description}</div></div>''', unsafe_allow_html=True)


def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dashboard data file not found: {DATA_PATH}")
        st.stop()
    try:
        gdf = gpd.read_file(DATA_PATH)
    except Exception as exc:
        st.error(f"Could not read dashboard GeoPackage: {exc}")
        st.stop()
    if gdf.empty:
        st.error("Dashboard dataset is empty."); st.stop()
    if gdf.crs is None:
        st.error("GeoPackage has no CRS. Check the source dataset."); st.stop()
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # Preserve the original grid polygons for the map. Calculate centroids separately
    # only for the latitude/longitude shown in the grid details panel.
    centroid_gdf = gdf.to_crs(epsg=3857).copy()
    centroid_gdf["geometry"] = centroid_gdf.geometry.centroid
    centroid_gdf = centroid_gdf.to_crs(epsg=4326)
    gdf["map_lon"] = centroid_gdf.geometry.x.to_numpy()
    gdf["map_lat"] = centroid_gdf.geometry.y.to_numpy()

    main_key = first_existing_column(gdf, ["grid_cell_id", "grid_id", "cell_id"])
    if main_key is None:
        st.error("No grid ID column found in GeoPackage."); st.stop()
    gdf[main_key] = gdf[main_key].astype(str)

    # Join cluster labels, keeping one row per grid ID.
    if CLUSTER_PATH.exists():
        try:
            clusters = pd.read_csv(CLUSTER_PATH)
            cluster_key = first_existing_column(clusters, ["grid_cell_id", "grid_id", "cell_id"])
            if cluster_key:
                clusters[cluster_key] = clusters[cluster_key].astype(str)
                clusters = clusters.drop_duplicates(subset=[cluster_key])
                columns = [c for c in clusters.columns if c == cluster_key or c not in gdf.columns]
                gdf = gdf.merge(clusters[columns], left_on=main_key, right_on=cluster_key, how="left", suffixes=("", "_cluster"))
        except Exception as exc:
            st.warning(f"Cluster data could not be joined: {exc}")

    # Join grid-wise historical trends. Do not overwrite original indicators.
    if TREND_PATH.exists():
        try:
            trends = pd.read_csv(TREND_PATH)
            trend_key = first_existing_column(trends, ["grid_cell_id", "grid_id", "cell_id"])
            if trend_key:
                trends[trend_key] = trends[trend_key].astype(str)
                trends = trends.drop_duplicates(subset=[trend_key])
                cols = [c for c in trends.columns if c == trend_key or c not in gdf.columns]
                gdf = gdf.merge(trends[cols], left_on=main_key, right_on=trend_key, how="left", suffixes=("", "_trend"))
        except Exception as exc:
            st.warning(f"Historical trend data could not be joined: {exc}")

    # Join validated forecast outputs when the forecast builder has been run.
    if FORECAST_PATH.exists():
        try:
            forecasts = pd.read_csv(FORECAST_PATH)
            forecast_key = first_existing_column(forecasts, ["grid_cell_id", "grid_id", "cell_id"])
            if forecast_key:
                forecasts[forecast_key] = forecasts[forecast_key].astype(str)
                forecasts = forecasts.drop_duplicates(subset=[forecast_key])
                cols = [c for c in forecasts.columns if c == forecast_key or c not in gdf.columns]
                gdf = gdf.merge(forecasts[cols], left_on=main_key, right_on=forecast_key, how="left", suffixes=("", "_forecast"))
        except Exception as exc:
            st.warning(f"Forecast data could not be joined: {exc}")
    return add_relative_categories(gdf)


def create_map(data, layer):
    data = data.copy()
    if layer == "Persistent historical heat":
        col, cmap, order, title, desc = "heat_category", HEAT_COLORS, list(HEAT_COLORS), "Persistent Historical Heat", "Relative historical heat patterns across assessed grids. The score is not temperature in °C."
    elif layer == "Green-planning priority":
        col, cmap, order, title, desc = "priority_category", PRIORITY_COLORS, list(PRIORITY_COLORS), "Green-Planning Priority", "Exploratory planning priority based on persistent heat, vegetation deficit and built-up signal."
    elif layer == "Vegetation Intervention Priority":
        if "vegetation_intervention_category" not in data.columns:
            data = add_intervention_priority(data)
        col, cmap, order, title, desc = "vegetation_intervention_category", PRIORITY_COLORS, list(PRIORITY_COLORS), "Vegetation Intervention Priority", "Relative shortlist for vegetation-planning review. Combines persistent heat (50%), vegetation deficit (30%), and built-up signal (20%); exploratory weights, not a measured planting impact."
    elif layer == "Vegetation (NDVI)":
        col, cmap, order, title, desc = "vegetation_category", VEGETATION_COLORS, list(VEGETATION_COLORS), "Vegetation (NDVI)", "Relative vegetation signal; NDVI is not a direct tree count."
    elif layer == "Built-up indicator (NDBI)":
        col, cmap, order, title, desc = "builtup_category", BUILTUP_COLORS, list(BUILTUP_COLORS), "Built-up Indicator (NDBI)", "Relative built-up spectral signal, not a direct building-density measurement."
    elif layer == "AI Heat Pattern Clusters":
        data["cluster_category"] = get_cluster_category(data)
        col, cmap, title, desc = "cluster_category", dict(CLUSTER_COLORS), "AI Heat Pattern Clusters", "K-Means groups similar observed patterns; clusters are not a future forecast or proof of causation."
        for val in data[col].dropna().unique():
            cmap.setdefault(val, "#8A9690")
        order = list(cmap)
    elif layer == "Historical temperature trend":
        if "trend_category" not in data.columns:
            data["trend_category"] = "Not assessed"
        data["trend_category"] = data["trend_category"].fillna("Not assessed")
        col, cmap, order, title, desc = "trend_category", TREND_COLORS, list(TREND_COLORS), "Historical LST Trend (2014–2025)", "Grid-wise linear trend fitted to annual hot-season Landsat surface-temperature observations. Historical trend, not a forecast."
    elif layer == "1-Year Forecast Heat Risk":
        col, cmap, order = "heat_risk_1y", FORECAST_RISK_COLORS, list(FORECAST_RISK_COLORS)
        title, desc = "Experimental 1-Year-Ahead Heat Risk", "Risk categories are based on the forecast LST relative to historical LST quantiles. Experimental range forecasts, not a climate projection or guarantee."
    elif layer == "2-Year Forecast Heat Risk":
        col, cmap, order = "heat_risk_2y", FORECAST_RISK_COLORS, list(FORECAST_RISK_COLORS)
        title, desc = "Experimental 2-Year-Ahead Heat Risk", "Longer-horizon forecasts are more uncertain. Risk categories are relative to the historical LST distribution, not official health-risk thresholds."
    else:
        assessed = get_assessed_mask(data)
        data["coverage_category"] = np.where(assessed, "Assessed", "Not assessed")
        col, cmap, order, title, desc = "coverage_category", {"Assessed":"#45B97C", "Not assessed":"#8A9690"}, ["Assessed", "Not assessed"], "Assessment Coverage", "Unassessed grids are excluded from scored categories, not assigned zero."

    if col not in data.columns:
        data[col] = "Not assessed"
    data[col] = data[col].fillna("Not assessed").astype(str)
    data = data.dropna(subset=["geometry"])
    if data.crs is None:
        st.error("Grid polygons have no coordinate reference system.")
        st.stop()
    if data.crs.to_epsg() != 4326:
        data = data.to_crs(epsg=4326)

    grid_key = "grid_cell_id" if "grid_cell_id" in data.columns else first_existing_column(data, ["grid_id", "cell_id"])
    if grid_key is None:
        st.error("No grid ID column found for polygon rendering.")
        st.stop()
    data[grid_key] = data[grid_key].astype(str)
    # Each GeoJSON feature retains its grid ID so Plotly can color the actual polygon.
    geojson = json.loads(data[[grid_key, col, "geometry"]].to_json(drop_id=True))
    hover_fields = [
        ("persistent_heat_score", "Heat score"),
        ("green_priority_score", "Green-planning priority"),
        ("median_ndvi", "Median NDVI"),
        ("median_ndbi", "Median NDBI"),
        ("slope_c_per_year", "Historical trend (°C/year)"),
        ("net_change_c", "Net LST change (°C)"),
        ("r_squared", "Trend R²"),
        ("lst_point_1y", "1-year LST estimate (°C)"),
        ("lst_lower_1y", "1-year range lower (°C)"),
        ("lst_upper_1y", "1-year range upper (°C)"),
        ("lst_point_2y", "2-year LST estimate (°C)"),
        ("lst_lower_2y", "2-year range lower (°C)"),
        ("lst_upper_2y", "2-year range upper (°C)"),
    ]
    available_hover = [(field, label) for field, label in hover_fields if field in data.columns]

    fig = go.Figure()
    for category in order:
        subset = data[data[col] == category]
        if subset.empty:
            continue
        custom_fields = [grid_key] + [field for field, _ in available_hover]
        customdata = subset[custom_fields].astype(object).where(pd.notna(subset[custom_fields]), None).values.tolist()
        hover_lines = ["<b>Grid %{customdata[0]}</b>", f"Category: {category}"]
        for idx, (_, label) in enumerate(available_hover, start=1):
            hover_lines.append(f"{label}: %{{customdata[{idx}]}}")
        hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"
        fig.add_trace(go.Choroplethmap(
            geojson=geojson,
            featureidkey=f"properties.{grid_key}",
            locations=subset[grid_key].tolist(),
            z=[1] * len(subset),
            zmin=0,
            zmax=1,
            colorscale=[[0, cmap.get(category, "#8A9690")], [1, cmap.get(category, "#8A9690")]],
            showscale=False,
            name=category,
            marker={"opacity":0.82, "line":{"color":"#071D16", "width":0.7}},
            customdata=customdata,
            hovertemplate=hovertemplate,
        ))

    # Label every grid polygon with its grid ID at the polygon centroid.
    # Keep labels as a separate map trace so category colors and hover details remain intact.
    label_data = data.dropna(subset=["map_lat", "map_lon"]).copy()
    if not label_data.empty:
        fig.add_trace(
            go.Scattermap(
                lat=label_data["map_lat"],
                lon=label_data["map_lon"],
                mode="text",
                text=label_data[grid_key].astype(str),
                textposition="middle center",
                textfont={"size": 9, "color": "#FFFFFF"},
                hoverinfo="skip",
                showlegend=False,
                name="Grid IDs",
            )
        )

    fig.update_layout(
        map={"style":"open-street-map", "center":{"lat":23.82,"lon":90.41}, "zoom":10.3},
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font={"family":"Arial, sans-serif", "color":TEXT, "size":14},
        title={"text":title, "font":{"size":23,"color":TEXT}, "x":.02,"xanchor":"left"},
        legend={"title":{"text":"GRID CATEGORY","font":{"size":17,"color":TEXT}}, "font":{"size":16,"color":TEXT}, "bgcolor":"rgba(11,37,27,.94)", "bordercolor":BORDER, "borderwidth":1, "itemsizing":"constant", "orientation":"v", "y":.97, "x":.02},
        margin={"l":0,"r":0,"t":65,"b":0},
        height=650,
    )
    return fig, desc

# ------------------------------ Load & validate ------------------------------
gdf = load_data()
gdf = add_intervention_priority(gdf)
required = ["grid_cell_id", "persistent_heat_score", "green_priority_score", "median_ndvi"]
missing = [c for c in required if c not in gdf.columns]
if missing:
    st.error("Dashboard dataset is missing required columns: " + ", ".join(missing)); st.stop()
for col in ["persistent_heat_score", "green_priority_score", "median_ndvi", "median_ndbi", "slope_c_per_year", "net_change_c", "r_squared", "start_lst_c", "end_lst_c"]:
    if col in gdf.columns: gdf[col] = to_numeric(gdf, col)
if "median_ndbi" not in gdf.columns: gdf["median_ndbi"] = np.nan
assessed_mask = get_assessed_mask(gdf)

# ------------------------------ Hero ------------------------------
st.markdown('''<div class="hero"><div class="hero-title">🌿 GreenGuard AI</div><p class="hero-subtitle">AI-Powered Urban Heat Intelligence & Green Planning for DNCC<br>Explore historical heat patterns, vegetation signals, built-up indicators, historical temperature trends, and grid-level green-planning priorities.</p></div>''', unsafe_allow_html=True)

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("## 🌱 GreenGuard AI")
    st.caption("DNCC Environmental Intelligence")
    st.markdown("---")
    st.markdown("### Map Controls")
    selected_layer = st.radio("Select map layer", LAYER_OPTIONS, index=0)
    st.markdown("---")
    st.markdown("### Dataset Overview")
    st.metric("Total grid cells", f"{len(gdf):,}")
    st.metric("Assessed grids", f"{int(assessed_mask.sum()):,}")
    st.metric("Not assessed", f"{int((~assessed_mask).sum()):,}")
    st.markdown("---")
    st.markdown("<div class='small-note'><b>Study period:</b> 2014–2025<br><b>Satellite:</b> Landsat 8/9<br><b>Grid size:</b> 500 m<br><b>Focus:</b> DNCC</div>", unsafe_allow_html=True)

# ------------------------------ Overview ------------------------------
st.markdown("## Dashboard Overview")
cols = st.columns(3)
cols[0].metric("Total grid cells", f"{len(gdf):,}", help="All grid cells in the DNCC study grid.")
cols[1].metric("Assessed grids", f"{int(assessed_mask.sum()):,}", help="Grids with sufficient data for assessment.")
cols[2].metric("Coverage excluded", f"{int((~assessed_mask).sum()):,}", help="Unassessed grids are not assigned zero scores.")
st.markdown("---")

# ------------------------------ Main map ------------------------------
st.markdown("## 🗺️ Environmental Map")
map_col, info_col = st.columns([3.2,1])
with map_col:
    fig, layer_description = create_map(gdf, selected_layer)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"scrollZoom":True})
with info_col:
    st.markdown(f"<div class='section-card'><h3 style='margin-top:0'>Layer Guide</h3><p class='small-note'>Choose a map layer from the sidebar. Colors represent categories for the selected indicator.</p></div><div class='section-card'><h4 style='margin-top:0'>What this map shows</h4><p class='small-note'>{layer_description}</p></div>", unsafe_allow_html=True)
    if selected_layer == "Persistent historical heat":
        items = [("#E05252","Persistent hotspot","Upper third of relative heat scores."),("#F2A541","Warm area","Middle third of relative heat scores."),("#45B97C","Relatively cooler","Lower third of relative heat scores."),("#8A9690","Not assessed","Insufficient assessment data; not a low-heat label.")]
    elif selected_layer == "Green-planning priority":
        items = [(PRIORITY_COLORS[k],k,"Relative exploratory priority category.") for k in ["Very High","High","Moderate","Low","Very Low","Not assessed"]]
    elif selected_layer == "Vegetation Intervention Priority":
        items = [(PRIORITY_COLORS[k], k, "Relative rank for vegetation-planning review.") for k in ["Very High", "High", "Moderate", "Low", "Very Low", "Not assessed"]]
    elif selected_layer == "Vegetation (NDVI)":
        items = [(VEGETATION_COLORS[k],k,"Relative NDVI category.") for k in VEGETATION_COLORS]
    elif selected_layer == "Built-up indicator (NDBI)":
        items = [(BUILTUP_COLORS[k],k,"Relative NDBI category.") for k in BUILTUP_COLORS]
    elif selected_layer == "AI Heat Pattern Clusters":
        items = [(CLUSTER_COLORS[k],k,"Cluster grouping indicates similarity, not causation.") for k in CLUSTER_COLORS]
    elif selected_layer == "Historical temperature trend":
        items = [(TREND_COLORS[k],k,"Linear trend label using exploratory ±0.10°C/year threshold." if k in ["Increasing","Decreasing","Relatively stable"] else "Trend unavailable.") for k in TREND_COLORS]
    else:
        items = [("#45B97C","Assessed","Sufficient data for scoring."),("#8A9690","Not assessed","Not scored; not assumed to be low risk.")]
    html = "<div class='section-card'><h4 style='margin-top:0'>Map Legend</h4>"
    for color, label, explanation in items:
        html += f"<p style='color:{color};font-weight:750;font-size:17px;margin-bottom:2px'>● {label}</p><p class='small-note'>{explanation}</p>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ------------------------------ Grid-level details ------------------------------
st.markdown("---")
st.markdown("## 🔎 Grid-Level Details")
st.markdown("<p class='small-note'>Select a grid to inspect its environmental indicators and historical temperature trend. Scores include descriptions of what each value means.</p>", unsafe_allow_html=True)
grid_ids = sorted(gdf["grid_cell_id"].dropna().astype(str).unique().tolist())
if not grid_ids:
    st.warning("No grid IDs are available."); st.stop()
selected_grid_id = st.selectbox("Select grid ID", grid_ids, help="Choose a grid to view indicators.")
selected_row = gdf[gdf["grid_cell_id"].astype(str) == selected_grid_id].iloc[0]
selected_is_assessed = bool(get_assessed_mask(gdf.loc[[selected_row.name]]).iloc[0])
coord_cols = st.columns(3)
coord_cols[0].markdown("**Grid ID**"); coord_cols[0].write(str(selected_row["grid_cell_id"]))
coord_cols[1].markdown("**Latitude**"); coord_cols[1].write(format_number(selected_row.get("map_lat", np.nan),5))
coord_cols[2].markdown("**Longitude**"); coord_cols[2].write(format_number(selected_row.get("map_lon", np.nan),5))
if not selected_is_assessed:
    st.warning("This grid is marked Not assessed. Missing scores must not be interpreted as zero or low environmental risk.")
heat_value = selected_row.get("persistent_heat_score", np.nan)
priority_value = selected_row.get("green_priority_score", np.nan)
ndvi_value = selected_row.get("median_ndvi", np.nan)
ndbi_value = selected_row.get("median_ndbi", np.nan)
priority_label = selected_row.get("priority_category", "Not assessed")
heat_display = format_number(heat_value,1) if selected_is_assessed else "N/A"
priority_display = format_number(priority_value,1) if selected_is_assessed else "N/A"
ndvi_display = format_number(ndvi_value,3) if selected_is_assessed else "N/A"
metric_cols = st.columns(3, gap="medium")
with metric_cols[0]: metric_card("Heat Score", heat_display, "Relative persistent-heat score from 0–100. Higher means the grid ranks higher in historical heat relative to assessed grids; it is not temperature in °C.")
with metric_cols[1]: metric_card("Green-Planning Priority", priority_display, f"Exploratory score. Category: {priority_label if selected_is_assessed else 'Not assessed'}. Higher values suggest higher relative priority for green-planning review, not an official risk grade.")
with metric_cols[2]: metric_card("Median NDVI", ndvi_display, "Median Normalized Difference Vegetation Index. Higher values generally indicate a stronger vegetation signal; it is not a tree count.")

st.markdown("### Additional Environmental Indicators")
extra = st.columns(3)
with extra[0]: metric_card("Median NDBI", format_number(ndbi_value,3) if selected_is_assessed else "N/A", "Normalized Difference Built-up Index. A relatively higher value can indicate a stronger built-up spectral signal; not a direct building-density measure.")
with extra[1]: metric_card("Priority Category", priority_label if selected_is_assessed else "Not assessed", "Relative category derived from the project's exploratory green-planning priority score.")
with extra[2]:
    cats = f"Heat: {selected_row.get('heat_category','Not assessed')}<br>Vegetation: {selected_row.get('vegetation_category','Not assessed')}<br>Built-up: {selected_row.get('builtup_category','Not assessed')}"
    metric_card("Relative Categories", "Grid profile", cats)

# ------------------------------ Vegetation intervention priority ------------------------------
st.markdown("---")
st.markdown("## 🌳 Where should DNCC increase vegetation first?")
st.markdown("<p class='small-note'>This shortlist highlights grids that may deserve earlier field review for tree planting, pocket parks, or other vegetation interventions. It ranks relative need; it does not predict how many degrees planting would reduce temperature.</p>", unsafe_allow_html=True)
priority_df = gdf.copy()
priority_df["Intervention priority"] = priority_df["vegetation_intervention_category"]
priority_df["Why this grid is flagged"] = priority_df.apply(intervention_reason, axis=1)
priority_df = priority_df[priority_df["vegetation_intervention_score"].notna()].sort_values("vegetation_intervention_score", ascending=False)
if priority_df.empty:
    st.info("A vegetation-intervention shortlist cannot be created because required heat and vegetation indicators are unavailable.")
else:
    top_priority = priority_df.head(min(20, len(priority_df))).copy()
    top_priority.insert(0, "Rank", range(1, len(top_priority) + 1))
    display_cols = ["Rank", "grid_cell_id", "Intervention priority", "vegetation_intervention_score", "persistent_heat_score", "median_ndvi", "median_ndbi", "Why this grid is flagged"]
    display = top_priority[display_cols].rename(columns={"grid_cell_id":"Grid ID", "vegetation_intervention_score":"Priority score", "persistent_heat_score":"Heat score", "median_ndvi":"Median NDVI", "median_ndbi":"Median NDBI"})
    for c in ["Priority score", "Heat score", "Median NDVI", "Median NDBI"]:
        display[c] = pd.to_numeric(display[c], errors="coerce").round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download priority shortlist (CSV)", data=display.to_csv(index=False).encode("utf-8-sig"), file_name="greenguard_vegetation_intervention_priority.csv", mime="text/csv")
    st.caption("Scores are exploratory relative ranks. Field verification is required before intervention decisions.")

# ------------------------------ Historical temperature analysis ------------------------------
st.markdown("---")
st.markdown("## 🌡️ Historical Temperature Trend Analysis")
st.markdown("<p class='small-note'>Uses existing 2014–2025 annual hot-season Landsat LST summaries and grid-wise linear trends. This is retrospective analysis, not a future temperature forecast.</p>", unsafe_allow_html=True)
if not YEARLY_PATH.exists() or not TREND_PATH.exists():
    st.warning("Trend outputs are not found yet. Run src/analyze_temperature_trends.py to create the historical trend CSV files.")
else:
    try:
        yearly = pd.read_csv(YEARLY_PATH)
        yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
        for c in ["mean_lst_c","median_lst_c","std_lst_c"]:
            if c in yearly.columns: yearly[c] = pd.to_numeric(yearly[c], errors="coerce")
        trend_df = pd.read_csv(TREND_PATH)
        if "grid_cell_id" in trend_df.columns:
            trend_df["grid_cell_id"] = trend_df["grid_cell_id"].astype(str)
        trend_df["slope_c_per_year"] = pd.to_numeric(trend_df.get("slope_c_per_year"), errors="coerce")
        trend_df["r_squared"] = pd.to_numeric(trend_df.get("r_squared"), errors="coerce")
        if "trend_category" not in trend_df.columns:
            trend_df["trend_category"] = "Insufficient observations"
        trend_df["trend_category"] = trend_df["trend_category"].fillna("Insufficient observations")

        trend_counts = trend_df["trend_category"].value_counts()
        total_trend_grids = int(trend_df["grid_cell_id"].nunique()) if "grid_cell_id" in trend_df.columns else len(trend_df)
        increasing_n = int(trend_counts.get("Increasing",0))
        stable_n = int(trend_counts.get("Relatively stable",0))
        decreasing_n = int(trend_counts.get("Decreasing",0))
        tc = st.columns(3)
        tc[0].metric("Increasing slope", f"{increasing_n:,}", f"{increasing_n/total_trend_grids:.1%} of trend records" if total_trend_grids else None)
        tc[1].metric("Relatively stable", f"{stable_n:,}", f"{stable_n/total_trend_grids:.1%} of trend records" if total_trend_grids else None)
        tc[2].metric("Decreasing slope", f"{decreasing_n:,}", f"{decreasing_n/total_trend_grids:.1%} of trend records" if total_trend_grids else None)

        st.markdown("### Grid-wise historical trend")
        trend_map_col, trend_detail_col = st.columns([1.15,1])
        with trend_map_col:
            st.markdown("**Trend category distribution**")
            counts_df = trend_df["trend_category"].value_counts().rename_axis("Trend category").reset_index(name="Grid count")
            fig_counts = px.bar(counts_df, x="Trend category", y="Grid count", color="Trend category", color_discrete_map=TREND_COLORS, text="Grid count")
            fig_counts.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font={"color":TEXT}, showlegend=False, margin={"l":10,"r":10,"t":25,"b":10}, height=350, xaxis_title="", yaxis_title="Number of grids")
            fig_counts.update_traces(textposition="outside")
            st.plotly_chart(fig_counts, use_container_width=True)
        with trend_detail_col:
            st.markdown("**Selected grid's fitted trend**")
            trend_ids = sorted(trend_df["grid_cell_id"].dropna().astype(str).unique()) if "grid_cell_id" in trend_df.columns else []
            if trend_ids:
                default_index = trend_ids.index(selected_grid_id) if selected_grid_id in trend_ids else 0
                trend_grid_id = st.selectbox("Grid for trend details", trend_ids, index=default_index, key="trend_grid_selector")
                tr = trend_df[trend_df["grid_cell_id"].astype(str) == trend_grid_id].iloc[0]
                slope = tr.get("slope_c_per_year", np.nan)
                r2 = tr.get("r_squared", np.nan)
                net = tr.get("net_change_c", np.nan)
                cat = tr.get("trend_category", "Insufficient observations")
                st.metric("Linear slope", f"{slope:+.3f} °C/year" if pd.notna(slope) else "N/A", help="Slope of a fitted straight line through annual grid-level LST observations. It is not a forecast.")
                small = st.columns(2)
                small[0].metric("Net first-to-last change", f"{net:+.2f} °C" if pd.notna(net) else "N/A")
                small[1].metric("R²", format_number(r2,3))
                st.markdown(f"**Trend category:** `{cat}`")
                st.caption("R² describes how closely a straight line fits the observations; it does not prove causation or statistical significance. The ±0.10°C/year category threshold is exploratory.")
            else:
                st.info("No grid IDs found in the trend file.")


    except Exception as exc:
        st.error(f"Could not load or plot trend outputs: {exc}")

# Clustering outputs remain available in the processed data; advanced cluster tables are hidden from the public dashboard.

# ------------------------------ Experimental ML Prediction ------------------------------
st.markdown("---")
st.markdown("## Experimental AI Heat Prediction")
st.caption("Random Forest model trained on historical grid-level features. Predictions are experimental and are not validated climate projections.")
ML_MODEL_PATH = ROOT / "data" / "processed" / "greenguard_random_forest.joblib"
ML_DATA_PATH = ROOT / "data" / "processed" / "greenguard_temporal_dataset_2014_2025.csv"
ML_FEATURES = ["LST_C_mean", "NDVI_mean", "NDBI_mean", "air_temp_c", "relative_humidity_pct", "wind_speed_ms", "rainfall_mm_season"]
if not ML_MODEL_PATH.exists() or not ML_DATA_PATH.exists():
    st.warning("ML model or temporal feature dataset is missing.")
else:
    try:
        import joblib
        @st.cache_resource
        def load_ml_model(path):
            return joblib.load(path)
        @st.cache_data
        def load_ml_features(path):
            return pd.read_csv(path)
        ml_model = load_ml_model(str(ML_MODEL_PATH))
        ml_data = load_ml_features(str(ML_DATA_PATH))
        required_columns = ["grid_cell_id", "year"] + ML_FEATURES
        if not set(required_columns).issubset(ml_data.columns):
            st.error("Temporal dataset is missing required model features.")
        else:
            ml_data["grid_cell_id"] = ml_data["grid_cell_id"].astype(str)
            ml_data["year"] = pd.to_numeric(ml_data["year"], errors="coerce")
            ml_data = ml_data.sort_values("year")
            latest_ml = ml_data.drop_duplicates(subset=["grid_cell_id"], keep="last").copy()
            latest_ml[ML_FEATURES] = latest_ml[ML_FEATURES].apply(pd.to_numeric, errors="coerce")
            latest_ml = latest_ml.dropna(subset=ML_FEATURES)
            if latest_ml.empty:
                st.warning("No complete feature rows are available for prediction.")
            else:
                selected_ml_grid = st.selectbox("Select grid for ML prediction", sorted(latest_ml["grid_cell_id"].unique()), key="experimental_ml_grid")
                row = latest_ml[latest_ml["grid_cell_id"] == selected_ml_grid].iloc[0]
                input_data = pd.DataFrame([[row[col] for col in ML_FEATURES]], columns=ML_FEATURES)
                predicted_lst = float(ml_model.predict(input_data)[0])
                last_observed_lst = float(row["LST_C_mean"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Experimental RF estimate", f"{predicted_lst:.2f} °C", help="Random Forest estimate for the next target year.")
                col2.metric("Latest observed LST", f"{last_observed_lst:.2f} °C", f"Feature year: {int(row['year'])}")
                col3.metric("RF vs. latest LST", f"{predicted_lst - last_observed_lst:+.2f} °C", help="Difference from the latest input LST; not a guaranteed temperature change.")
                st.caption(f"Grid: {selected_ml_grid} | Input year: {int(row['year'])} | Target year: {int(row['year']) + 1}")
                with st.expander("View model input features"):
                    st.dataframe(input_data, use_container_width=True, hide_index=True)
                st.warning("Validation note: the current Random Forest did not consistently outperform the persistence baseline in previous temporal tests. Treat this as an experimental model demonstration, not a reliable forecast.")
    except Exception as exc:
        st.error(f"Could not generate the ML prediction: {exc}")

# ------------------------------ Methodology ------------------------------
st.markdown("---")
st.markdown("## 📘 Methodology & Interpretation")
with st.expander("How to interpret the scores", expanded=False):
    st.markdown("""
    ### 1. Persistent historical heat score
    Relative score from 0–100 summarizing historical heat rank. It is not temperature in °C.

    ### 2. Green-planning priority
    The exploratory score combines 50% persistent heat, 30% vegetation deficit and 20% built-up indicator. These project-defined weights are not scientifically validated or officially adopted municipal weights.

    ### 3. NDVI and NDBI
    NDVI is a satellite-derived vegetation signal; NDBI is a spectral built-up indicator. Neither is a direct tree count or building-density measurement. Categories are relative to assessed grids.

    ### 4. Historical temperature trends
    Each grid's annual LST series is fitted with a linear trend. The slope is in °C/year. The script labels slopes >= +0.10 as Increasing and <= −0.10 as Decreasing; the threshold is exploratory, not a statistical significance test. R² measures linear fit, not causation.

    ### 5. Experimental Random Forest prediction
    The model uses historical grid-level LST, NDVI, NDBI and weather features to produce an experimental next-year estimate. Previous temporal tests showed it did not consistently outperform a persistence baseline, so the output is a model demonstration rather than a reliable forecast.

    ### 6. Unassessed grids
    Grids excluded for insufficient data coverage remain unassessed. They are not assigned zero scores and must not be interpreted as cooler or lower priority.
    """)
with st.expander("Data and study scope", expanded=False):
    st.markdown("""
    - **Study area:** Dhaka North City Corporation (DNCC)
    - **Grid size:** 500 m
    - **Study period:** 2014–2025
    - **Satellite indicators:** Landsat 8/9 land-surface temperature, NDVI and NDBI
    - **Seasonal scope:** Hot-season analysis
    - **Purpose:** Historical heat-pattern analysis and exploratory green-planning support

    This is an environmental decision-support prototype. The Random Forest output is experimental, not a validated climate projection. Year-to-year LST changes can reflect observation timing, cloud/coverage conditions, and surface variability.
    """)

st.markdown("---")
st.markdown("<div style='text-align:center;padding:16px 8px 6px;color:#9FBBA8;font-size:12px;line-height:1.8'><b style='color:#6BE0A1'>GreenGuard AI</b><br>AI-Powered Urban Heat Intelligence & Green Planning for DNCC<br>Team 404: Brain Not Found</div>", unsafe_allow_html=True)
