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

# === CONFIG ===
# Paths to local data
PR_FOLDER = "chelsa_data/pr"
TASMAX_FOLDER = "chelsa_data/tasmax"
TASMIN_FOLDER = "chelsa_data/tasmin"
COASTAL_FLOOD_PATH = "inuncoast_rcp4p5_wtsub_2030_rp0010_0.tif"
RIVERINE_FLOOD_PATH = "inunriver_rcp4p5_0000HadGEM2-ES_2030_rp00010.tif"
FIRE_FLOOD_PATH = "fwixd_ann_HadGEM3-GC31-LL_ssp245_r1i1p1f3_g025.nc"
WIND_SI_PATH = "STORM_FIXED_RETURN_PERIODS_SI_50_YR_RP.tif"
WIND_NA_PATH = "STORM_FIXED_RETURN_PERIODS_NA_50_YR_RP.tif"
WIND_NI_PATH = "STORM_FIXED_RETURN_PERIODS_NI_50_YR_RP.tif"
LANDSLIDE_PATH = "LS_RF_Mean_1980-2018_COG.tif"
CC_CSV_PATH = "cc_ssa1.csv"
GPKG_FOLDER = "ccgpkg"  # folder with multiple .gpkg files

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
    "Electrical": {
        "flood": 3, "drought": 1, "heat": 2, "cold": 1, "high_precip": 1, "fire":1, "landslide": 1, "wind": 1
    },
    "LPG": {
        "flood": 3, "drought": 0, "heat": 2, "cold": 1, "high_precip": 1, "fire": 3, "landslide": 1, "wind": 1
    },
    "Biomass": {
        "flood": 2, "drought": 3, "heat": 1, "cold": 1, "high_precip": 2, "fire": 2, "landslide": 1, "wind": 0
    }
}

# === Map type toggle ===
# Choose map type: "exposure" or "risk"
MAP_TYPE = "risk"
plot_combined_risk = True



def categorize_technology(tech):
    t = (tech or "").strip().lower()
    if t in ["electricity", "electrical"]:
        return "Electrical"
    elif t in ["lpg", "ethanol"]:
        return "LPG"
    elif t in ["biomass forced draft", "pellets forced draft", "biogas", "traditional"]:
        return "Biomass"
    else:
        return "Other" # For any unmapped technologies

def sample_raster(gdf_points, raster_path):
    with rasterio.open(raster_path) as src:
        xs = gdf_points.geometry.x.values
        ys = gdf_points.geometry.y.values

        rows_cols = [src.index(x, y) for x, y in zip(xs, ys)]
        data = src.read(1)  # read entire raster once

        vals = []
        for (row, col) in rows_cols:
            if (0 <= row < src.height) and (0 <= col < src.width):
                v = data[row, col]
                if v == src.nodata:
                    v = np.nan
            else:
                v = np.nan
                # Debug: show which points are outside raster
                # print(f"Point outside raster: row={row}, col={col}")
            vals.append(v)

    return np.array(vals)

""" def sample_raster(gdf_points, raster_path):
    with rasterio.open(raster_path) as src:
        coords = [(x,y) for x,y in zip(gdf_points.geometry.x, gdf_points.geometry.y)]
        vals = list(src.sample(coords))
        vals = [v[0] if v[0] != src.nodata else np.nan for v in vals]
    return np.array(vals) """

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

    # Extract array (year_index might be time or year axis)
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
    print("Loading clean cooking points from CSV...")
    df = pd.read_csv(CC_CSV_PATH)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry']), crs="EPSG:3857")
    gdf = gdf.to_crs(epsg=4326)

    if gdf.empty:
        print("No clean cooking data loaded. Exiting.")
        return gdf

    # Categorize by max_benefit_tech
    gdf["Tech_Category"] = gdf["max_benefit_tech"].apply(categorize_technology)
    # Check for uncategorized technologies
    unique_techs = gdf["max_benefit_tech"].dropna().unique()
    tech_cats = gdf["Tech_Category"].unique()
    print("Technologies found in CSV:", list(unique_techs))
    print("Technology categories assigned:", list(tech_cats))

    if "Other" in tech_cats:
        other_techs = gdf[gdf["Tech_Category"] == "Other"]["max_benefit_tech"].unique()
        print("⚠️ Warning: The following technologies were not categorized:", list(other_techs))
        raise ValueError("Unmapped technologies detected. Please update categorize_technology() to handle all types.")

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
            tasmax_monthly = (tasmax_monthly / 10) - 273.15 # Convert from decicelsius to Celsius and Kelvin to Celsius
            print("tasmax values (°C): min =", np.nanmin(tasmax_monthly),
                  "max =", np.nanmax(tasmax_monthly))
        else:
            tasmax_monthly = None

        if RUN_COLD:
            print("Sampling min temperature (tasmin)...")
            tasmin_monthly = stack_and_sample_monthly(gdf, TASMIN_FOLDER, "tasmin")
            tasmin_monthly = (tasmin_monthly / 10) - 273.15 # Convert from decicelsius to Celsius and Kelvin to Celsius
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
            # SPI threshold for extreme drought: ≤ -2
            gdf['drought_exposed'] = gdf['drought_spi3'] <= DROUGHT_SPI_THRESHOLD
        else:
            gdf['drought_exposed'] = False
            gdf['drought_spi3'] = np.nan

        """ if RUN_DROUGHT:
            djf_precip = pr_monthly[[11,0,1], :].sum(axis=0) # Dec, Jan, Feb
            gdf['dry_season_precip'] = djf_precip
            gdf['drought_exposed'] = djf_precip < DROUGHT_THRESHOLD
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
        gdf['fire_risk_raw'] = fire_values # Keep raw values for potential analysis
        gdf['fire_exposed'] = gdf['fire_risk_raw'] > FIRE_THRESHOLD
    else:
        gdf['fire_risk_raw'] = np.nan
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
        total_invest = subset["investment_costs"].sum()
        tech_rows = {
            "% Settlements Exposed": {},
            "Total Investment Exposed USD": {},
            "Total Population Exposed": {}
        }
        
        for hazard in hazards_for_summary:
            exp_col = f"{hazard}_exposed"
            exposed_subset = subset[subset[exp_col] == True]

            # Investment: per settlement cost × households, then summed
            exposed_invest = (exposed_subset["investment_costs"] * exposed_subset["Households"]).sum()

            # % settlements exposed
            pct_sett = (len(exposed_subset) / len(subset)) * 100 if len(subset) > 0 else 0

            # Population exposed
            exposed_pop = exposed_subset["Calibrated_pop"].sum()

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
    excel_filename = "exposure_summary_cc.xlsx"
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
        ("Electrical", False): "#fdbf6f",
        ("Electrical", True): "#ff7f00",
        ("LPG", False): "#a6cee3",
        ("LPG", True): "#377eb8",
        ("Biomass", False): "#cab2d6",
        ("Biomass", True): "#984ea3",
        ("Other", False): "#cccccc",
        ("Other", True): "#555555"
    }

    # Plot order so LPG goes underneath, then Electrical, then Biomass on top
    tech_order = ["LPG", "Electrical", "Biomass", "Other"]

    if exposure_column not in gdf.columns:
        print(f"[Warning] Column {exposure_column} not found in GeoDataFrame. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Same map setup as electrification, but borders drawn last
    ax.set_extent([-20, 55, -30, 23])
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)
    ax.set_axis_off()

    # Plot each technology in the desired order
    for tech in tech_order:
        tech_gdf = gdf[gdf["Tech_Category"] == tech].copy()
        if tech_gdf.empty:
            continue

        # Non-exposed first for this tech
        non_exposed = tech_gdf[tech_gdf[exposure_column] == False].copy()
        if not non_exposed.empty:
            non_exposed.plot(
                ax=ax,
                color=color_map.get((tech, False), "#cccccc"),
                markersize=0.2,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                aspect=None
            )

        # Exposed on top of non-exposed for this tech
        exposed = tech_gdf[tech_gdf[exposure_column] == True].copy()
        if not exposed.empty:
            exposed.plot(
                ax=ax,
                color=color_map.get((tech, True), "#555555"),
                markersize=0.2,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                aspect=None
            )

    # Re-apply extent after plotting
    ax.set_extent([-20, 55, -30, 23])

    # Draw boundaries last so they remain visible
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=11)

    legend_elements = [
        mpatches.Patch(color="#ff7f00", label=f"Electrical ({hazard_label} Exposed)"),
        mpatches.Patch(color="#fdbf6f", label="Electrical (Not Exposed)"),
        mpatches.Patch(color="#377eb8", label=f"LPG ({hazard_label} Exposed)"),
        mpatches.Patch(color="#a6cee3", label="LPG (Not Exposed)"),
        mpatches.Patch(color="#984ea3", label=f"Biomass ({hazard_label} Exposed)"),
        mpatches.Patch(color="#cab2d6", label="Biomass (Not Exposed)"),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='small')

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_risk_map(gdf, risk_column, title, output_filename, hazard):
    risk_colors = {
        3: "#990000",
        2: "#FF9900",
        1: "#FFFF66",
        0: "#cccccc"
    }

    exposure_col = f"{hazard}_exposed"
    if exposure_col not in gdf.columns:
        print(f"[Warning] Exposure column '{exposure_col}' not found in GeoDataFrame.")
        return

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.set_extent([-20, 55, -30, 23])
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)
    ax.set_axis_off()

    # Not exposed first
    not_exposed = gdf[gdf[exposure_col] == False].copy()
    if not not_exposed.empty:
        not_exposed.plot(
            ax=ax,
            color=risk_colors[0],
            markersize=0.2,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            aspect=None
        )

    # Exposed by risk level, low to high so high risk sits on top
    for risk_level in [1, 2, 3]:
        subset = gdf[(gdf[exposure_col] == True) & (gdf[risk_column] == risk_level)].copy()
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=risk_colors[risk_level],
                markersize=0.2,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                aspect=None
            )

    ax.set_extent([-20, 55, -30, 23])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=11)

    legend_elements = [
        mpatches.Patch(color="#990000", label="Risk of direct damage or system failure"),
        mpatches.Patch(color="#FF9900", label="Risk of reduced performance or operational disruption"),
        mpatches.Patch(color="#FFFF66", label="Indirect or context-dependent risk"),
        mpatches.Patch(color="#cccccc", label="Risk unlikely or settlement not exposed")
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='small')

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_combined_risk_map(gdf, output_filename="combined_risk_map_clean_cooking.png"):
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
        risks_for_exposed_hazards = [
            row[risk_col] for exp_col, risk_col in hazard_cols
            if row.get(exp_col, False) and pd.notnull(row.get(risk_col))
        ]
        max_risks.append(max(risks_for_exposed_hazards) if risks_for_exposed_hazards else 0)

    gdf["combined_max_risk_level"] = max_risks

    risk_colors = {
        3: "#990000",
        2: "#FF9900",
        1: "#FFFF66",
        0: "#cccccc",
    }

    gdf["risk_color"] = gdf["combined_max_risk_level"].apply(
        lambda x: risk_colors.get(x, "#000000")
    )

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.set_extent([-20, 55, -30, 23])
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)
    ax.set_axis_off()

    # Lower risk first, higher risk last
    for level in [0, 1, 2, 3]:
        subset = gdf[gdf["combined_max_risk_level"] == level].copy()
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=subset["risk_color"],
                markersize=0.2,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                aspect=None
            )

    ax.set_extent([-20, 55, -30, 23])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=11)

    legend_elements = [
        mpatches.Patch(color="#990000", label="Risk of direct damage or system failure"),
        mpatches.Patch(color="#FF9900", label="Risk of reduced performance or operation"),
        mpatches.Patch(color="#FFFF66", label="Indirect or context-dependent risk"),
        mpatches.Patch(color="#cccccc", label="Risk unlikely or settlement not exposed")
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='16', frameon=False)
    #ax.set_title("Combined Climate Hazard Risk Level (Max per Settlement)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_map(gdf, hazard, hazard_label):
    if MAP_TYPE == "exposure":
        exposure_col = f"{hazard}_exposed"
        title = f"Clean Cooking Exposure to {hazard_label} by Technology Type"
        filename = f"{hazard}_exposure_map_cc.png"
        plot_hazard_map(gdf, exposure_col, title, filename, hazard_label)
    elif MAP_TYPE == "risk":
        risk_col = f"{hazard}_risk"
        title = f"Clean Cooking Risk Scores for {hazard_label} by Technology Type"
        filename = f"{hazard}_risk_map_cc.png"
        plot_risk_map(gdf, risk_col, title, filename, hazard)
    else:
        print(f"[Warning] Unknown MAP_TYPE '{MAP_TYPE}'. No map plotted.")


def plot_exposure_panel(gdf, hazards, hazard_labels, output_filename="exposure_panel_cc.png"):
    color_map = {
        ("Electrical", False): "#fdbf6f",
        ("Electrical", True): "#ff7f00",
        ("LPG", False): "#a6cee3",
        ("LPG", True): "#377eb8",
        ("Biomass", False): "#cab2d6",
        ("Biomass", True): "#984ea3",
        ("Other", False): "#cccccc",
        ("Other", True): "#555555"
    }

    # LPG first, then Electrical, then Biomass
    tech_order = ["LPG", "Electrical", "Biomass", "Other"]

    # Same structure as electrification
    fig = plt.figure(figsize=(28, 16))

    gs = fig.add_gridspec(
        2, 4, height_ratios=[1, 1], hspace=0.25, wspace=0.1
    )

    axes_top = [fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(4)]
    axes_bottom = [fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree()) for i in range(4)]
    axes = axes_top + axes_bottom

    for i, (hazard, label) in enumerate(zip(hazards, hazard_labels)):
        ax = axes[i]
        exposure_col = f"{hazard}_exposed"

        ax.set_extent([-20, 55, -30, 23])
        ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)
        ax.set_axis_off()

        # Plot technologies in desired order
        for tech in tech_order:
            tech_gdf = gdf[gdf["Tech_Category"] == tech].copy()
            if tech_gdf.empty:
                continue

            non_exposed = tech_gdf[tech_gdf[exposure_col] == False].copy()
            if not non_exposed.empty:
                non_exposed.plot(
                    ax=ax,
                    color=color_map.get((tech, False), "#cccccc"),
                    markersize=0.2,
                    alpha=0.6,
                    transform=ccrs.PlateCarree(),
                    aspect=None
                )

            exposed = tech_gdf[tech_gdf[exposure_col] == True].copy()
            if not exposed.empty:
                exposed.plot(
                    ax=ax,
                    color=color_map.get((tech, True), "#555555"),
                    markersize=0.2,
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                    aspect=None
                )

        # Re-apply extent after GeoPandas plotting
        ax.set_extent([-20, 55, -30, 23])

        # Borders last, so they remain visible
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=11)

        # Larger titles
        ax.set_title(label, fontsize=26, pad=10)

    # Remove unused subplot if needed
    if len(hazards) < len(axes):
        axes[-1].remove()

    fig.suptitle("Exposure to Climate Hazards by Clean Cooking Technology", fontsize=24, y=0.92)

    legend_elements = [
        mpatches.Patch(color="#ff7f00", label="Electrical (Exposed)"),
        mpatches.Patch(color="#fdbf6f", label="Electrical (Not Exposed)"),
        mpatches.Patch(color="#377eb8", label="LPG (Exposed)"),
        mpatches.Patch(color="#a6cee3", label="LPG (Not Exposed)"),
        mpatches.Patch(color="#984ea3", label="Biomass-based (Exposed)"),
        mpatches.Patch(color="#cab2d6", label="Biomass-based (Not Exposed)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=18,
        ncol=3,
        frameon=False
    )

    # Keep same as electrification; do NOT use bbox_inches="tight"
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    plt.savefig(output_filename, dpi=300)
    plt.close()

# === RUN ===

if __name__ == "__main__":
    cc = main()
    if cc.empty:
        print("No data to plot, exiting.")
        exit()

    if MAP_TYPE == "exposure":
        hazards = ["flood", "wind", "drought", "heat", "cold", "high_precip", "landslide", "fire"]
        hazard_labels = [
            "Flood", "Extreme Wind", "Drought", "Extreme Heat", "Extreme Cold",
            "High Precipitation", "Landslide", "Wildfire"
        ]
        print("Plotting exposure panel map...")
        plot_exposure_panel(cc, hazards, hazard_labels, "exposure_panel_cc.png")

    # if RUN_FLOOD:
    #     print(f"Plotting flood {MAP_TYPE} map...")
    #     plot_map(cc, "flood", "Flood")

    # if RUN_DROUGHT:
    #     print(f"Plotting drought {MAP_TYPE} map...")
    #     plot_map(cc, "drought", "Drought")

    # if RUN_HEAT:
    #     print(f"Plotting heat {MAP_TYPE} map...")
    #     plot_map(cc, "heat", "Extreme Heat")

    # if RUN_COLD:
    #     print(f"Plotting cold {MAP_TYPE} map...")
    #     plot_map(cc, "cold", "Extreme Cold")

    # if RUN_HIGH_PRECIP:
    #     print(f"Plotting high precipitation {MAP_TYPE} map...")
    #     plot_map(cc, "high_precip", "High Precipitation")

    # if RUN_FIRE:
    #     print(f"Plotting fire {MAP_TYPE} map...")
    #     plot_map(cc, "fire", "Fire Weather Risk")
    
    # if RUN_LANDSLIDE:
    #     print(f"Plotting landslide {MAP_TYPE} map...")
    #     plot_map(cc, "landslide", "Landslide")

    # if RUN_WIND:
    #     print(f"Plotting wind {MAP_TYPE} map...")
    #     plot_map(cc, "wind", "Extreme Wind")

    if plot_combined_risk:
        print("Plotting combined risk map...")
        plot_combined_risk_map(cc)

        # Optional: Print combined risk statistics
        total_points = len(cc)
        if total_points > 0:
            risk_counts = cc["combined_max_risk_level"].value_counts().sort_index()
            print("\nCombined Risk Level Summary:")
            for level, label in {
                0: "Grey (Not Exposed)",
                1: "Yellow (Low Risk)",
                2: "Orange (Medium Risk)",
                3: "Red (High Risk)"
            }.items():
                count = risk_counts.get(level, 0)
                percent = (count / total_points) * 100
                print(f"{label}: {count:,} points ({percent:.1f}%)")