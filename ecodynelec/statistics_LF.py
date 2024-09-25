import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def get_footprint_time_series(ds_CI):
    """Return statistics below a given threesold"""
    # Calculate Consumption footprint
    df = pd.DataFrame()
    df["Consumption_mix"] = (
        ds_CI.sel(Method = 'Consumption').sum(
            dim=["Emission_phase", "Energies", "Production_Countries"], skipna=None
        )
        .rename({"Consumption_Countries": "Countries"})
        .to_dataframe("CF").CF
    )
    # Calculate Production footprint
    df["Production_mix"] = (
        ds_CI.sel(Method = 'Production').isel(Consumption_Countries = 0).sum(
            dim=["Emission_phase", "Energies"], skipna=None
        )
        .rename({"Production_Countries": "Countries"})
        .to_dataframe("CF").CF
    )
    df = df.unstack().swaplevel(0, 1, axis=1)
    df.columns.names = ["Countries", "Mix"]
    return df


def get_load_factor_below_limit(df_TS, GHG_limit=18 * 3.6):
    """Limit expressed of carbon footprint in gCO2eq/kWh. Default value is 18 gCO2eq/MJ (~65 gCO2eq/kWh) corresponding to the European Delegated Act."""
    df_LF = (df_TS < GHG_limit).mean().unstack()
    return df_LF


def get_geodraframe(df):
    """Convert dataframe into a geodataframe by adding geometry data"""
    gdf_eurostat = gpd.read_file("data/geometry_files/NUTS_RG_20M_2021_3035.shp")
    gdf_eurostat[gdf_eurostat.LEVL_CODE == 0]
    gdf_eurostat.index = gdf_eurostat.NUTS_ID
    list_common_countries = list(set(df.index).intersection(gdf_eurostat.index))
    df = df.loc[list_common_countries]
    df["geometry"] = gdf_eurostat.loc[df.index].geometry
    gdf = gpd.GeoDataFrame(df)
    return gdf


def plot_statistics_map(gdf):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    gdf.plot(
        "Production_mix",
        ax=ax[0],
        cmap="Reds",
        edgecolor="k",
        legend=True,
        vmin=0,
        vmax=1,
        legend_kwds={
            "location": "bottom",
            "label": "Percentage of time below the threshold",
        },
    )
    ax[0].set_xlim([2e6, 6e6])
    ax[0].set_ylim([1.4e6, 5e6])

    gdf.plot(
        "Consumption_mix",
        ax=ax[1],
        cmap="Blues",
        edgecolor="k",
        legend=True,
        vmin=0,
        vmax=1,
        legend_kwds={
            "location": "bottom",
            "label": "Percentage of time below the threshold",
        },
    )
    ax[1].set_xlim([2e6, 6e6])
    ax[1].set_ylim([1.4e6, 5e6])

    ax[0].grid(False)
    ax[1].grid(False)
    ax[0].axis("off")
    ax[1].axis("off")

    ax[0].set_title("Production mix")
    ax[1].set_title("Consumption mix")

    return fig, ax
