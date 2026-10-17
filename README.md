# GreenGuard AI
### AI-Powered Urban Heat Intelligence & Green Planning for DNCC

**GreenGuard AI** is a geospatial environmental analysis and decision-support platform designed to explore urban heat patterns and support vegetation-based green-planning decisions across Dhaka North City Corporation (DNCC), Bangladesh.

The platform combines satellite-derived environmental indicators, historical meteorological data, geospatial analysis, K-Means clustering, and experimental machine learning to help users investigate persistent relative heat patterns and identify areas that may warrant further environmental assessment.

**Live Demo:** https://greenguard-ai.streamlit.app/

**GitHub Repository:** https://github.com/prottoy000/GreenGuard-AI

---

## 1. Problem Statement

Rapid urbanization and the expansion of built-up surfaces can influence local land surface temperatures and vegetation patterns. Understanding the spatial distribution of these environmental characteristics is important for urban environmental assessment and green-planning discussions.

However, satellite-derived environmental information can be difficult to interpret without accessible visualization and spatial analysis tools.

GreenGuard AI addresses this challenge by providing an interactive, grid-based dashboard for exploring historical heat patterns, vegetation conditions, built-up surface indicators, and potential priorities for vegetation interventions across DNCC.

## 2. Project Objectives

- Analyze historical land surface temperature patterns across DNCC.
- Explore relationships between heat, vegetation, and built-up surface indicators.
- Identify relatively persistent heat hotspots among assessed grid cells.
- Apply K-Means clustering to explore groups of grid cells with similar heat-pattern characteristics.
- Develop an exploratory vegetation intervention-priority score.
- Investigate the potential of machine learning for land surface temperature estimation.
- Present environmental information through an accessible, interactive geospatial dashboard.

## 3. Key Features

### Historical Heat Analysis
Explore the spatial distribution of historical land surface temperature and identify grid cells that repeatedly exhibit relatively high heat levels.

### Vegetation Assessment
Visualize vegetation conditions using the Normalized Difference Vegetation Index (NDVI), a satellite-derived indicator of vegetation greenness.

### Built-up Surface Analysis
Examine built-up surface signals using the Normalized Difference Built-up Index (NDBI).

### AI-Based Heat Pattern Clustering
Use K-Means clustering to group grid cells based on selected environmental characteristics, supporting exploratory analysis of similarities in heat patterns.

### Temperature Trend Analysis
Explore historical land surface temperature trends and identify grid cells with increasing, decreasing, or relatively stable trends, subject to data coverage and the implemented classification criteria.

### Green-Planning Prioritization
Explore an intervention-priority score combining relative heat, vegetation deficit, and built-up surface signals to identify locations for further investigation.

### Experimental Machine Learning
Explore model-based land surface temperature estimates generated using selected environmental features. Model outputs are experimental and require further validation.

### Interactive Geospatial Dashboard
Navigate map layers, examine environmental indicators, compare grid-level characteristics, and explore potential green-planning priorities.

## 4. Methodology

GreenGuard AI follows a geospatial data-processing and analysis workflow.

1. **Data Collection:** Obtain satellite-derived observations and relevant meteorological data for the study area.
2. **Quality Control:** Identify invalid observations and account for data-quality and coverage limitations.
3. **Environmental Feature Extraction:** Derive land surface temperature and calculate environmental indicators such as NDVI and NDBI.
4. **Spatial Aggregation:** Aggregate relevant observations into grid-based geographic units to support spatial comparisons.
5. **Historical Heat Analysis:** Examine temperature patterns across available years to identify relatively persistent hotspots.
6. **K-Means Clustering:** Group grid cells using selected environmental features to explore similarities in heat-pattern characteristics.
7. **Trend Analysis:** Estimate historical temperature trends using the available time-series data.
8. **Intervention-Priority Assessment:** Combine selected heat, vegetation, and built-up indicators into an exploratory relative priority score.
9. **Machine Learning Experimentation:** Evaluate model-based land surface temperature estimates against appropriate validation baselines.
10. **Visualization:** Present the resulting spatial analyses through the interactive Streamlit dashboard.

The clustering, trend, prioritization, and prediction components serve different purposes. Clusters describe similarities between observations; trend analysis examines change over time; intervention prioritization supports exploratory screening; and the experimental model estimates land surface temperature.

## 5. Data Sources

The project uses or is designed to incorporate the following environmental data sources, depending on the processing pipeline and availability of individual variables.

| Data Source | Purpose |
|---|---|
| Landsat 8/9 | Satellite imagery, land surface temperature, vegetation indicators, and built-up surface indicators |
| ERA5-Land | Historical meteorological variables, such as air temperature, humidity, wind speed, and rainfall |
| WorldPop | Gridded population estimates, where incorporated into the relevant analysis |

### Environmental Indicators

**Land Surface Temperature (LST)**

Represents the temperature of the Earth's surface estimated from satellite thermal observations. It is different from near-surface air temperature.

**Normalized Difference Vegetation Index (NDVI)**

An indicator derived from satellite reflectance that helps characterize vegetation greenness.

**Normalized Difference Built-up Index (NDBI)**

A spectral indicator used to explore built-up surface characteristics.

**Meteorological Variables**

Air temperature, relative humidity, wind speed, and rainfall provide additional environmental context where available.

**Population Density**

Gridded population estimates may provide demographic context for future exposure-oriented analyses. Population density should not be interpreted as a direct measurement of individual heat exposure.

The presence of a variable in the broader project does not necessarily mean it is used in every dashboard calculation or machine-learning model.

## 6. K-Means Clustering

K-Means is an unsupervised machine-learning algorithm that partitions observations into a specified number of clusters based on similarity in their selected feature space.

In GreenGuard AI, clustering supports the exploratory identification of groups of grid cells with similar environmental characteristics.

The general workflow is:

1. Select the environmental features used by the clustering pipeline.
2. Prepare the feature values for analysis.
3. Apply K-Means to assign each eligible observation to a cluster.
4. Visualize the cluster assignments geographically.
5. Compare the environmental characteristics of the resulting groups.

Clustering is not the same as supervised prediction. It does not independently establish that a location is dangerous, nor does a cluster label automatically represent an official heat-risk category.

The interpretation of each cluster depends on the actual input features, preprocessing, number of clusters, and validation performed.

## 7. Green-Planning Priority Assessment

The intervention-priority component is designed to help screen locations that may warrant further investigation for vegetation-based interventions.

The current dashboard's exploratory score combines three dimensions:

- **Relative heat:** Higher persistent heat receives greater priority.
- **Vegetation deficit:** Lower vegetation indicator values increase priority.
- **Built-up surface signal:** Higher built-up indicator values increase priority.

The score is intended for relative comparison among assessed grid cells. It is not an officially approved DNCC planning index, a direct measurement of heat exposure, or proof that a proposed intervention will reduce local temperature.

## 8. Machine Learning and Validation

The project includes an experimental machine-learning component for estimating land surface temperature from selected environmental features.

The current experimental model uses a Random Forest approach and environmental inputs selected by the implemented pipeline.

Model evaluation and comparison with a simple persistence baseline are important because a complex model does not necessarily outperform a simpler method on historical data.

The current experimental model has not demonstrated consistently reliable performance over the persistence baseline. Therefore, its estimates should not be presented as validated operational forecasts.

Further work is needed on temporal validation, feature evaluation, uncertainty analysis, baseline comparison, and generalization to unseen years.

## 9. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Geospatial Analysis | GeoPandas and related geospatial tools |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Clustering | K-Means |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 10. Limitations

GreenGuard AI is an exploratory environmental analysis prototype, not a validated climate-prediction or operational risk-management system.

- Historical heat categories are relative to the assessed grid cells and should not be interpreted as universal temperature-based risk classifications.
- NDVI is an indirect vegetation indicator; it does not directly measure tree counts or canopy coverage.
- NDBI is an indirect built-up surface indicator; it does not directly measure building density.
- K-Means clustering identifies similarities within selected features but does not independently establish causality or official risk levels.
- Intervention priorities are exploratory and do not prove the effectiveness of vegetation planting at any specific location.
- Satellite observation quality, cloud masking, spatial coverage, and temporal consistency can affect the results.
- Experimental machine-learning estimates require additional validation before practical predictive use.
- Unassessed grid cells should not be interpreted as low-risk locations.

## 11. Future Scope

Potential future improvements include:

- Strengthening temporal and spatial validation of temperature estimates.
- Evaluating additional environmental and demographic features.
- Improving cluster interpretation and assessing clustering stability.
- Quantifying prediction uncertainty and comparing alternative models.
- Refining intervention priorities using locally validated environmental evidence.
- Improving data coverage and documenting data-processing assumptions.
- Evaluating the platform with urban planners and environmental domain experts.

## 12. Running Locally

### Prerequisites

- Python 3.10 or a compatible Python version supported by the project's dependencies.
- Git.
- The project's required data and model files.

### Installation

Clone the repository:

```bash
git clone https://github.com/prottoy000/GreenGuard-AI.git
cd GreenGuard-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, usually:

`http://localhost:8501`

**Note:** Local execution requires the data files and saved model artifacts referenced by the application. Confirm that all required files are included or that the project provides instructions for obtaining them.

## 13. Live Application

The deployed dashboard is available at:

**https://greenguard-ai.streamlit.app/**

The application can be accessed through a web browser without installing Python locally.

## 14. Team

**Team Imperio**

## 15. Project Status

GreenGuard AI is a hackathon prototype focused on historical urban heat analysis, environmental pattern exploration, and exploratory green-planning support for Dhaka North City Corporation.

The project demonstrates how geospatial data processing, unsupervised clustering, and experimental machine learning can be brought together in an interactive environmental analysis platform.

---

*GreenGuard AI is intended to support environmental exploration and evidence-informed discussion. Its outputs should be interpreted in the context of the underlying data, methodology, and validation limitations.*
