import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import xarray as xr
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gamma, norm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# === CONFIGURATION ===
# Paths to local raster folders
PR_FOLDER = "chelsa_data/pr"
TASMAX_FOLDER = "chelsa_data/tasmax"
TASMIN_FOLDER = "chelsa_data/tasmin"
COASTAL_FLOOD_PATH = "inuncoast_rcp4p5_wtsub_2030_rp0010_0.tif"
RIVERINE_FLOOD_PATH = "inunriver_rcp4p5_0000HadGEM2-ES_2030_rp00010.tif"
FIRE_FLOOD_PATH = "fwixd_ann_HadGEM3-GC31-LL_ssp245_r1i1p1f3_g025.nc"
WIND_SI_PATH = "STORM_FIXED_RETURN_PERIODS_SI_50_YR_RP.tif"
WIND_NI_PATH = "STORM_FIXED_RETURN_PERIODS_NI_50_YR_RP.tif"
WIND_NA_PATH = "STORM_FIXED_RETURN_PERIODS_NA_50_YR_RP.tif"
LANDSLIDE_PATH = "LS_RF_Mean_1980-2018_COG.tif"

# Load OnSSET results CSV
SETTLEMENT_CSV = "SE4ALLCombinedResults.csv"

# Exposure thresholds
DROUGHT_SPI_THRESHOLD = -2.0 # spi threshold for extreme drought
HEAT_THRESHOLD = 38          # °C max monthly max temperature
HIGH_PRECIP_THRESHOLD = 300  # mm max monthly precipitation
FLOOD_THRESHOLD = 0.5        # meters water depth for flood exposure
COLD_THRESHOLD = 0           # °C min monthly min temperature
FIRE_THRESHOLD = 50          # FWI
WIND_THRESHOLD = 25          # m/s threshold for extreme wind
LANDSLIDE_ABSOLUTE_THRESHOLD = 0.01
LANDSLIDE_PERCENTILE_THRESHOLD = 0.95

PR_SCALE = 0.1               # CHELSA precipitation scale factor to convert raw values to mm/month

# Hazard toggles
RUN_DROUGHT = True
RUN_HEAT = True
RUN_HIGH_PRECIP = True
RUN_FLOOD = True
RUN_COLD = True
RUN_FIRE = True
RUN_WIND = True
RUN_LANDSLIDE = True

# Months as strings (two-digit)
MONTHS = [str(m).zfill(2) for m in range(1, 13)]

# === Risk Ranking by Technology and Hazard ===
TECH_HAZARD_RISK = {
    "Grid": {
        "flood": 3,
        "drought": 2,
        "heat": 2,
        "cold": 3,
        "high_precip": 3,
        "fire": 3,
        "landslide": 3,
        "wind": 3
    },
    "Mini-grid": {
        "flood": 3,
        "drought": 2,
        "heat": 2,
        "cold": 2,
        "high_precip": 2,
        "fire": 3,
        "landslide": 3,
        "wind": 3
    },
    "Standalone": {
        "flood": 3,
        "drought": 1,
        "heat": 2,
        "cold": 2,
        "high_precip": 1,
        "fire": 2,
        "landslide": 1,
        "wind": 1
    }
}

# === Map type toggle ===
# Choose map type: "exposure" or "risk"
MAP_TYPE = "risk"
plot_combined_risk = True

# === HELPER FUNCTIONS ===

def classify_tech(row):
    if row['MinimumOverallLCOE2030'] == row['Grid2030']:
        return 'Grid'
    else:
        return row['Minimum_Tech_Off_grid2030']

def categorize_technology(tech):
    tech = str(tech).lower()
    if tech == "grid":
        return "Grid"
    elif tech.startswith("mg_"):
        return "Mini-grid"
    elif tech.startswith("sa_"):
        return "Standalone"
    else:
        return "Other"
    
def sample_raster(gdf_points, raster_path):
    with rasterio.open(raster_path) as src:
        # 1) Ensure points are in the raster's CRS
        if gdf_points.crs is None:
            raise ValueError(f"GeoDataFrame has no CRS; raster is {src.crs}.")
        pts = gdf_points if gdf_points.crs == src.crs else gdf_points.to_crs(src.crs)

        xs = pts.geometry.x.values
        ys = pts.geometry.y.values

        # 2) Convert to row/col (per-point because rasterio.index isn't array-vectorized)
        rows_cols = [src.index(float(x), float(y)) for x, y in zip(xs, ys)]
        rows = np.array([rc[0] for rc in rows_cols], dtype=int)
        cols = np.array([rc[1] for rc in rows_cols], dtype=int)

        # 3) Bounds check: mark out-of-bounds as NaN instead of indexing error
        oob = (rows < 0) | (rows >= src.height) | (cols < 0) | (cols >= src.width)

        data = src.read(1)  # read band once
        vals = np.full(xs.shape, np.nan, dtype="float32")
        if (~oob).any():
            rr = rows[~oob]
            cc = cols[~oob]
            picked = data[rr, cc]
            if src.nodata is not None:
                picked = np.where(picked == src.nodata, np.nan, picked)
            vals[~oob] = picked

        return vals

def stack_and_sample_monthly(gdf_points, folder_path, variable):
    """Loads 12 monthly tif files for a variable and samples all at once"""
    monthly_values = []
    for m in MONTHS:
        filename = f"CHELSA_gfdl-esm4_r1i1p1f1_w5e5_ssp370_{variable}_{m}_2011_2040_norm.tif"
        path = os.path.join(folder_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        vals = sample_raster(gdf_points, path)
        monthly_values.append(vals)
    return np.array(monthly_values)  # shape: (12 months, n_points)

from scipy.interpolate import RegularGridInterpolator

def sample_fire_risk_smooth(gdf_points, fire_nc_path, year_index=15):
    # Open dataset
    ds = xr.open_dataset(fire_nc_path)

    # Get the actual coords
    if "lat" in ds.coords:
        lats = ds["lat"].values
    else:
        lats = ds["latitude"].values

    if "lon" in ds.coords:
        lons = ds["lon"].values
    else:
        lons = ds["longitude"].values

    # Extract array
    fwi = ds["fwixd"].isel(time=year_index).values

    # Ensure shape matches
    if fwi.shape != (len(lats), len(lons)):
        raise ValueError(f"Shape mismatch: fwi.shape={fwi.shape}, "
                         f"expected ({len(lats)}, {len(lons)})")

    # Build interpolator
    interp = RegularGridInterpolator(
        (lats, lons), fwi, bounds_error=False, fill_value=np.nan
    )

    # Settlement points (lat, lon order)
    points = np.vstack([gdf_points.geometry.y.values,
                        gdf_points.geometry.x.values]).T
    fire_vals = interp(points)

    return fire_vals

""" def sample_fire_risk(gdf_points, fire_nc_path, year_index=15):
    ds = xr.open_dataset(fire_nc_path)
    lons = ds['lon'].values
    lats = ds['lat'].values
    fwi = ds['fwixd'][year_index, :, :].values # shape (lat, lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    tree = cKDTree(grid_points)
    points = np.vstack([gdf_points.geometry.x.values,
gdf_points.geometry.y.values]).T
    _, idx = tree.query(points, k=1)
    fwi_flat = fwi.ravel()
    fire_vals = fwi_flat[idx]
    return fire_vals """

def rolling_3month(series): # For drought SPI
    """Compute 3-month rolling totals (with wrap-around for Dec-Feb)."""
    out = []
    for start in range(12):
        idx = [(start + i) % 12 for i in range(3)]
        out.append(series[idx].sum())
    return np.array(out)

def compute_spi3(baseline_monthly, future_monthly):
    """
    baseline_monthly: array shape (12,) = baseline monthly climatology for 1981-2010
    future_monthly: array shape (12,) = scenario monthly climatology for 2011-2040
    Returns: SPI-3 value (lowest 3-mo period standardized)
    """
    # build 3-month totals
    baseline_3m = rolling_3month(baseline_monthly)
    future_3m = rolling_3month(future_monthly)

    # fit gamma to baseline
    baseline_3m = baseline_3m[baseline_3m > 0]  # avoid zero/negatives
    if len(baseline_3m) < 5:  # too few values to fit
        return np.nan

    shape, loc, scale = gamma.fit(baseline_3m, floc=0)

    # compare *lowest* 3-mo total in scenario
    fut_val = np.nanmin(future_3m)

    # cumulative probability
    prob = gamma.cdf(fut_val, shape, loc=loc, scale=scale)
    # convert to SPI (standard normal)
    spi = norm.ppf(prob)
    return spi

# === MAIN PROCESS ===

def main():
    print("Loading settlements...")
    df = pd.read_csv(SETTLEMENT_CSV)
    df['Selected_Technology'] = df.apply(classify_tech, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X_deg'], df['Y_deg']), crs='EPSG:4326')

    if RUN_FLOOD:
        print("Sampling flood rasters...")
        gdf['flood_coastal'] = sample_raster(gdf, COASTAL_FLOOD_PATH)
        gdf['flood_river'] = sample_raster(gdf, RIVERINE_FLOOD_PATH)
        gdf['flood_max'] = gdf[['flood_coastal', 'flood_river']].max(axis=1)
        gdf['flood_exposed'] = gdf['flood_max'] > FLOOD_THRESHOLD
    else:
        gdf['flood_exposed'] = False

    if RUN_DROUGHT or RUN_HEAT or RUN_HIGH_PRECIP or RUN_COLD:
        print("Sampling climate rasters...")
        if RUN_DROUGHT or RUN_HIGH_PRECIP:
            print("Sampling precipitation (pr)...")
            pr_monthly = stack_and_sample_monthly(gdf, PR_FOLDER, "pr") * PR_SCALE
        else:
            pr_monthly = None

        if RUN_HEAT:
            print("Sampling max temperature (tasmax)...")
            tasmax_monthly = stack_and_sample_monthly(gdf, TASMAX_FOLDER, "tasmax")
            tasmax_monthly = (tasmax_monthly / 10) - 273.15
            print("tasmax values (°C): min =", np.nanmin(tasmax_monthly), 
                  "max =", np.nanmax(tasmax_monthly))
        else:
            tasmax_monthly = None

        if RUN_COLD:
            print("Sampling min temperature (tasmin)...")
            tasmin_monthly = stack_and_sample_monthly(gdf, TASMIN_FOLDER, "tasmin")
            tasmin_monthly = (tasmin_monthly / 10) - 273.15
        else:
            tasmin_monthly = None

        if RUN_DROUGHT:
            # === Load baseline climatology (1981-2010) ===
            baseline_monthly = []
            for m in MONTHS:
                fname = f"CHELSA_pr_{m}_1981-2010_V.2.1.tif"
                path = os.path.join(PR_FOLDER, fname)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing baseline file: {path}")
                vals = sample_raster(gdf, path)
                baseline_monthly.append(vals)
            baseline_monthly = np.array(baseline_monthly) * PR_SCALE # (12, n_points)

            # === Already have scenario monthly from earlier ===
            n_months, n_points = pr_monthly.shape  

            drought_spi = []
            for i in range(n_points):
                spi_val = compute_spi3(baseline_monthly[:, i], pr_monthly[:, i])
                drought_spi.append(spi_val)

            gdf['drought_spi3'] = drought_spi
            # SPI threshold for extreme drought: ≤ threshold
            gdf['drought_exposed'] = gdf['drought_spi3'] <= DROUGHT_SPI_THRESHOLD
        else:
            gdf['drought_exposed'] = False
            gdf['drought_spi3'] = np.nan

            """ n_months, n_points = pr_monthly.shape

            rolling_sums = np.zeros((12, n_points))
            for start in range(12):
                idx = [(start + i) % 12 for i in range(3)]
                rolling_sums[start, :] = pr_monthly[idx, :].sum(axis=0)

            # Lowest 3-month precip total
            min_3mo_precip = rolling_sums.min(axis=0)

            gdf['min_3mo_precip'] = min_3mo_precip
            gdf['drought_exposed'] = min_3mo_precip < DROUGHT_THRESHOLD
        else:
            gdf['drought_exposed'] = False """

        if RUN_HEAT:
            max_temp = np.nanmax(tasmax_monthly, axis=0)
            gdf['max_temp'] = max_temp
            gdf['heat_exposed'] = max_temp > HEAT_THRESHOLD
        else:
            gdf['heat_exposed'] = False

        if RUN_COLD:
            min_temp = np.nanmin(tasmin_monthly, axis=0)
            gdf['min_temp'] = min_temp
            gdf['cold_exposed'] = min_temp < COLD_THRESHOLD
        else:
            gdf['cold_exposed'] = False

        if RUN_HIGH_PRECIP:
            max_pr = np.nanmax(pr_monthly, axis=0)
            gdf['max_monthly_pr'] = max_pr
            gdf['high_precip_exposed'] = max_pr > HIGH_PRECIP_THRESHOLD
        else:
            gdf['high_precip_exposed'] = False
    else:
        gdf['drought_exposed'] = False
        gdf['heat_exposed'] = False
        gdf['high_precip_exposed'] = False
        gdf['cold_exposed'] = False

    if RUN_FIRE:
        print("Sampling fire weather index...")
        fire_values = sample_fire_risk_smooth(gdf, FIRE_FLOOD_PATH, year_index=15)
        gdf['fire_risk'] = fire_values
        gdf['fire_exposed'] = gdf['fire_risk'] > FIRE_THRESHOLD
    else:
        gdf['fire_risk'] = np.nan
        gdf['fire_exposed'] = False
    
    if RUN_WIND:
        print("Sampling wind risk rasters...")
        gdf['wind_si'] = sample_raster(gdf, WIND_SI_PATH)
        gdf['wind_ni'] = sample_raster(gdf, WIND_NI_PATH)
        gdf['wind_na'] = sample_raster(gdf, WIND_NA_PATH)
        gdf['wind_max'] = gdf[['wind_si', 'wind_ni', 'wind_na']].max(axis=1)
        gdf['wind_exposed'] = gdf['wind_max'] > WIND_THRESHOLD
    else:    
        gdf['wind_exposed'] = False
    
    if RUN_LANDSLIDE:
        print("Sampling landslide raster...")
        gdf['landslide_value'] = sample_raster(gdf, LANDSLIDE_PATH)
        absolute_floor = LANDSLIDE_ABSOLUTE_THRESHOLD
        percentile_cutoff = gdf['landslide_value'].quantile(LANDSLIDE_PERCENTILE_THRESHOLD)
        gdf['landslide_exposed'] = (
            (gdf['landslide_value'] > absolute_floor) &
            (gdf['landslide_value'] > percentile_cutoff)
        )
    else:
        gdf['landslide_exposed'] = False
        gdf['landslide_value'] = np.nan

    gdf['Tech_Category'] = gdf['Selected_Technology'].apply(categorize_technology)

    for hazard in ["flood", "drought", "heat", "cold", "high_precip", "fire", "landslide", "wind"]:
        colname = f"{hazard}_risk"
        gdf[colname] = gdf["Tech_Category"].apply(
            lambda tech: TECH_HAZARD_RISK.get(tech, {}).get(hazard, 0)
        )

    #summarise
    summary = gdf.groupby('Tech_Category').agg(
        Flood_Exposure=('flood_exposed', 'mean'),
        Drought_Exposure=('drought_exposed', 'mean'),
        Heat_Exposure=('heat_exposed', 'mean'),
        HighPrecip_Exposure=('high_precip_exposed', 'mean'),
        Cold_Exposure=('cold_exposed','mean'),
        Fire_Exposure=('fire_exposed', 'mean'),
        Landslide_Exposure=('landslide_exposed', 'mean'),
        Wind_Exposure=('wind_exposed', 'mean')
    ).reset_index()

    print("\nExposure Summary by Technology:")
    print(summary)

        # === Investment & Population exposure analysis ===
    hazards_for_summary = [
        "flood", "drought", "heat", "cold",
        "high_precip", "fire", "wind", "landslide"
    ]

    # Build wide-format table: hazards as columns, 3 rows per tech
    wide_summary = {}

    for tech, subset in gdf.groupby("Tech_Category"):
        total_invest = subset["InvestmentCost2030"].sum()
        tech_rows = {
            "% Settlements Exposed": {},
            "Total Investment Exposed USD": {},
            "Total Population Exposed": {}
        }
        
        for hazard in hazards_for_summary:
            exp_col = f"{hazard}_exposed"
            exposed_subset = subset[subset[exp_col] == True]

            # Investment
            exposed_invest = exposed_subset["InvestmentCost2030"].sum()
            pct_sett = (len(exposed_subset) / len(subset)) * 100 if len(subset) > 0 else 0

            # Population
            exposed_pop = exposed_subset["Pop2030"].sum()

            # Fill metrics
            tech_rows["% Settlements Exposed"][hazard] = round(pct_sett, 2)
            tech_rows["Total Investment Exposed USD"][hazard] = round(exposed_invest, 0)
            tech_rows["Total Population Exposed"][hazard] = int(exposed_pop)
        
        wide_summary[tech] = tech_rows

    # Flatten into DataFrame
    rows = []
    for tech, metrics in wide_summary.items():
        for metric, hazard_values in metrics.items():
            row = {"Technology": tech, "Metric": metric}
            row.update(hazard_values)
            rows.append(row)

    wide_df = pd.DataFrame(rows)

    # Reorder columns: Technology, Metric, hazards
    wide_df = wide_df[["Technology", "Metric"] + hazards_for_summary]

    print("\nWide-format Exposure Summary (Investments & Population):")
    print(wide_df)

    # Export to Excel nicely
    excel_filename = "exposure_summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
        wide_df.to_excel(writer, sheet_name="ExposureSummary", index=False)

        workbook = writer.book
        worksheet = writer.sheets["ExposureSummary"]

        # Adjust column widths
        worksheet.set_column("A:A", 15)  # Technology
        worksheet.set_column("B:B", 30)  # Metric
        worksheet.set_column("C:Z", 18)  # Hazards

    print(f"\n✅ Exported wide-format summary (no % Investment) to {excel_filename}")

    return gdf

# === PLOTTING FUNCTIONS ===

def plot_hazard_map(gdf, exposure_column, title, output_filename, hazard_label):
    # Color map for technology + exposure status
    color_map = {
        ("Grid", False): "#fdbf6f",
        ("Grid", True): "#ff7f00",
        ("Mini-grid", False): "#a6cee3",
        ("Mini-grid", True): "#377eb8",
        ("Standalone", False): "#cab2d6",
        ("Standalone", True): "#984ea3",
        ("Other", False): "#cccccc",
        ("Other", True): "#555555"
    }

    # Safety check
    if exposure_column not in gdf.columns:
        print(f"[Warning] Column {exposure_column} not found in GeoDataFrame. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add map background
    ax.set_extent([-20, 55, -30, 23])  # Focus on SSA
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.set_axis_off()  # Remove ticks and frame

    # Plot non-exposed settlements first
    non_exposed = gdf[gdf[exposure_column] == False].copy()
    non_exposed['color'] = non_exposed.apply(
        lambda row: color_map.get((row["Tech_Category"], False), "#cccccc"), axis=1
    )
    non_exposed.plot(ax=ax, color=non_exposed['color'], markersize=0.2, alpha=0.6)

    # Then exposed settlements on top
    exposed = gdf[gdf[exposure_column] == True].copy()
    exposed['color'] = exposed.apply(
        lambda row: color_map.get((row["Tech_Category"], True), "#555555"), axis=1
    )
    exposed.plot(ax=ax, color=exposed['color'], markersize=0.2, alpha=0.9)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#ff7f00", label=f"Grid ({hazard_label} Exposed)"),
        mpatches.Patch(color="#fdbf6f", label="Grid (Not Exposed)"),
        mpatches.Patch(color="#377eb8", label=f"Mini-grid ({hazard_label} Exposed)"),
        mpatches.Patch(color="#a6cee3", label="Mini-grid (Not Exposed)"),
        mpatches.Patch(color="#984ea3", label=f"Standalone ({hazard_label} Exposed)"),
        mpatches.Patch(color="#cab2d6", label="Standalone (Not Exposed)"),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='small')

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_risk_map(gdf, risk_column, title, output_filename, hazard):
    # Risk color mapping
    risk_colors = {
        3: "#990000",   # Red – high risk
        2: "#FF9900",   # Orange – medium risk
        1: "#FFFF66",   # Yellow – low/uncertain risk
        0: "#cccccc"    # Grey – not exposed or no risk
    }

    # Determine exposure column
    exposure_col = f"{hazard}_exposed"
    if exposure_col not in gdf.columns:
        print(f"[Warning] Exposure column '{exposure_col}' not found in GeoDataFrame.")
        return

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add geographic features
    ax.set_extent([-20, 55, -30, 23])  # SSA focus
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.set_axis_off()  # Remove borders and ticks

    # Settlements not exposed – all grey
    not_exposed = gdf[gdf[exposure_col] == False].copy()
    not_exposed.plot(ax=ax, color=risk_colors[0], markersize=0.2, alpha=0.6)

    # Exposed settlements – colored by risk score
    for risk_level in [1, 2, 3]:  # Plot in order so high risk appears on top
        subset = gdf[(gdf[exposure_col] == True) & (gdf[risk_column] == risk_level)].copy()
        if not subset.empty:
            subset.plot(ax=ax, color=risk_colors[risk_level], markersize=0.2, alpha=0.9)

    # Custom legend
    legend_elements = [
        mpatches.Patch(color="#990000", label="Risk of direct damage or system failure"),
        mpatches.Patch(color="#FF9900", label="Risk of reduced performance or operational disruption"),
        mpatches.Patch(color="#FFCC00", label="Indirect or context-dependent risk"),
        mpatches.Patch(color="#cccccc", label="Risk unlikely or settlement not exposed")
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='small')

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_combined_risk_map(gdf, output_filename="combined_risk_map.png"):
    # Hazard exposure and risk columns
    hazard_cols = [
        ("flood_exposed", "flood_risk"),
        ("drought_exposed", "drought_risk"),
        ("heat_exposed", "heat_risk"),
        ("cold_exposed", "cold_risk"),
        ("high_precip_exposed", "high_precip_risk"),
        ("fire_exposed", "fire_risk"),
        ("landslide_exposed", "landslide_risk"),
        ("wind_exposed", "wind_risk"),
    ]

    max_risks = []
    for _, row in gdf.iterrows():
        risks = [row[risk] for exp, risk in hazard_cols if row.get(exp, False) and pd.notnull(row[risk])]
        max_risks.append(max(risks) if risks else 0)

    gdf["combined_max_risk_level"] = max_risks

    # Color mapping for max risk level
    risk_colors = {
        3: "#990000",  # red: high
        2: "#FF9900",  # orange: medium
        1: "#FFFF66",  # yellow: low
        0: "#cccccc"  # grey: not exposed
    }
    gdf["risk_color"] = gdf["combined_max_risk_level"].apply(lambda x: risk_colors.get(x, "#000000"))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([-20, 55, -30, 23])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.set_axis_off()

    gdf_sorted = gdf.sort_values("combined_max_risk_level")
    gdf_sorted.plot(ax=ax, color=gdf_sorted["risk_color"], markersize=0.2, alpha=0.9)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#990000", label="Risk of direct damage or system failure"),
        mpatches.Patch(color="#FF9900", label="Risk of reduced performance or operation"),
        mpatches.Patch(color="#FFFF66", label="Indirect or context-dependent risk"),
        mpatches.Patch(color="#bdbdbd", label="Risk unlikely or settlement not exposed")
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='16', frameon=False)
    #ax.set_title("Combined Climate Hazard Risk Level (Max per Settlement)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_map(gdf, hazard, hazard_label):
    if MAP_TYPE == "exposure":
        exposure_col = f"{hazard}_exposed"
        title = f"Settlement Exposure to {hazard_label} by Technology Type"
        filename = f"{hazard}_exposure_map.png"
        plot_hazard_map(gdf, exposure_col, title, filename, hazard)
    elif MAP_TYPE == "risk":
        risk_col = f"{hazard}_risk"
        title = f"Settlement Risk Scores for {hazard_label} by Technology Type"
        filename = f"{hazard}_risk_map.png"
        plot_risk_map(gdf, risk_col, title, filename, hazard)
    else:
        print(f"[Warning] Unknown MAP_TYPE '{MAP_TYPE}'. No map plotted.")

def plot_exposure_panel(gdf, hazards, hazard_labels, output_filename="exposure_panel.png"):
    # Color map for technology + exposure status
    color_map = {
        ("Grid", False): "#fdbf6f",
        ("Grid", True): "#ff7f00",
        ("Mini-grid", False): "#a6cee3",
        ("Mini-grid", True): "#377eb8",
        ("Standalone", False): "#cab2d6",
        ("Standalone", True): "#984ea3",
        ("Other", False): "#cccccc",
        ("Other", True): "#555555"
    }

    # Create figure with subplots (panel layout)
    # Create figure with custom GridSpec layout
    fig = plt.figure(figsize=(28, 16))  # taller figure for more vertical space

    # Grid: 2 rows, 4 columns
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1, 1], hspace=0.25, wspace=0.1
    )

    # Top row: 4 hazards
    axes_top = [fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(4)]

    # Bottom row: 3 hazards, centered (skip first column)
    axes_bottom = [fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree()) for i in range(4)]

    # Combine into single list so the rest of code works as before
    axes = axes_top + axes_bottom

    for i, (hazard, label) in enumerate(zip(hazards, hazard_labels)):
        ax = axes[i]
        exposure_col = f"{hazard}_exposed"

        ax.set_extent([-20, 55, -30, 23])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
        ax.set_axis_off()

        # Plot non-exposed
        non_exposed = gdf[gdf[exposure_col] == False].copy()
        non_exposed['color'] = non_exposed.apply(
            lambda row: color_map.get((row["Tech_Category"], False), "#cccccc"), axis=1
        )
        non_exposed.plot(
            ax=ax,
            color=non_exposed['color'],
            markersize=0.2,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            aspect=None
        )

        # Plot exposed
        exposed = gdf[gdf[exposure_col] == True].copy()
        exposed['color'] = exposed.apply(
            lambda row: color_map.get((row["Tech_Category"], True), "#555555"), axis=1
        )
        
        exposed.plot(
            ax=ax,
            color=exposed['color'],
            markersize=0.2,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            aspect=None
        )

        ax.set_title(label, fontsize=26, pad=10)

    # Remove any unused subplot
    if len(hazards) < len(axes):
        axes[-1].remove()

    # Global title
    fig.suptitle("Exposure to Climate Hazards by Electrification Technology", fontsize=24, y=0.92)

    # Single shared legend
    legend_elements = [
        mpatches.Patch(color="#ff7f00", label="Grid (Exposed)"),
        mpatches.Patch(color="#fdbf6f", label="Grid (Not Exposed)"),
        mpatches.Patch(color="#377eb8", label="Mini-grid (Exposed)"),
        mpatches.Patch(color="#a6cee3", label="Mini-grid (Not Exposed)"),
        mpatches.Patch(color="#984ea3", label="Standalone (Exposed)"),
        mpatches.Patch(color="#cab2d6", label="Standalone (Not Exposed)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=18,
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    plt.savefig(output_filename, dpi=300)
    plt.close()

# === RUN ===

if __name__ == "__main__":
    gdf=main()

    if MAP_TYPE == "exposure":
        hazards = ["flood", "wind", "drought", "heat", "cold", "high_precip", "landslide", "fire"]
        hazard_labels = [
            "Flood", "Extreme Wind", "Drought", "Extreme Heat", "Extreme Cold",
            "High Precipitation", "Landslide", "Wildfire"
        ]
        print("Plotting exposure panel map...")
        plot_exposure_panel(gdf, hazards, hazard_labels, "exposure_panel.png")

    # if RUN_FLOOD:
    #     print(f"Plotting flood {MAP_TYPE} map...")
    #     plot_map(gdf, "flood", "Flood")

    # if RUN_DROUGHT:
    #     print(f"Plotting drought {MAP_TYPE} map...")
    #     plot_map(gdf, "drought", "Drought")

    # if RUN_HEAT:
    #     print(f"Plotting heat {MAP_TYPE} map...")
    #     plot_map(gdf, "heat", "Extreme Heat")

    # if RUN_COLD:
    #     print(f"Plotting cold {MAP_TYPE} map...")
    #     plot_map(gdf, "cold", "Extreme Cold")

    # if RUN_HIGH_PRECIP:
    #     print(f"Plotting high precipitation {MAP_TYPE} map...")
    #     plot_map(gdf, "high_precip", "High Precipitation")

    # if RUN_FIRE:
    #     print(f"Plotting fire {MAP_TYPE} map...")
    #     plot_map(gdf, "fire", "Wildfire")
    
    # if RUN_LANDSLIDE:
    #     print(f"Plotting landslide {MAP_TYPE} map...")
    #     plot_map(gdf, "landslide", "Landslide")

    # if RUN_WIND:
    #     print(f"Plotting wind {MAP_TYPE} map...")
    #     plot_map(gdf, "wind", "Extreme Wind")

    if plot_combined_risk:
        print("Plotting combined risk map...")
        plot_combined_risk_map(gdf)

        total_settlements = len(gdf)
        risk_counts = gdf["combined_max_risk_level"].value_counts().sort_index()

        for level, label in {
            0: "Grey (Not Exposed)",
            1: "Yellow (Low Risk)",
            2: "Orange (Medium Risk)",
            3: "Red (High Risk)"
        }.items():
            count = risk_counts.get(level, 0)
            percent = (count / total_settlements) * 100
            print(f"{label}: {count:,} settlements ({percent:.1f}%)")