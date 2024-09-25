import matplotlib.pyplot as plt
import pycountry
import pandas as pd
import numpy as np

dict_colors = {
    "Wind": "lightblue",
    "Solar": "orange",
    "Other Renewable": "#8EBA42",
    "Hydro": "#348ABD",
    "Nuclear": "purple",
    "Fossil": "#777777",
    "Other": "#FFB5B8",
}

list_color_for_plot = list(dict_colors.values())
list_ordered_for_plot = list(dict_colors.keys())

def plot_generation_data(country_code, ds_gen, label_for_unit="Power (MW)"):
    """This function plot the"""
    if country_code == "XK":
        country_name = "Kosovo"
    else:
        country_name = pycountry.countries.get(alpha_2=country_code).name
    print(country_name)

    fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True, sharex=True)
    ax = ax.ravel()

    fig.suptitle(country_name, y=1.05, fontsize=18)
    dfp = (
        ds_gen.sel(Countries=country_code)
        .ActualGenerationOutput.to_dataset("Energies")
        .to_dataframe()
        .dropna(axis=1, how="all")
    )
    dfp.plot(
        ax=ax[0],
        kind="area",
        stacked=True,
        alpha=0.4,
        legend=False,
    )
    ax[0].set_title("Actual Generation Output")
    ax[0].set_ylabel(label_for_unit)

    dfp = (
        ds_gen.sel(Countries=country_code)
        .ActualConsumption.to_dataset("Energies")
        .to_dataframe()
        .dropna(axis=1, how="all")
    )
    dfp.plot(ax=ax[1], kind="area", stacked=True, alpha=0.4, legend=reversed)

    dfp = (
        ds_gen.sel(Countries=country_code)
        .sum(dim="Energies")
        .to_dataframe()
        .dropna(axis=1, how="all")
    )
    dfp.plot(ax=ax[2], kind="line", legend=True, alpha=0.8)
    if "Hydro Pumped Storage" in ds_gen.sel(Countries=country_code).Energies:
        dfp = (
            ds_gen.sel(Countries=country_code)
            .sel(Energies="Hydro Pumped Storage")
            .to_dataframe()
            .ActualConsumption
        )
        dfp.plot(ax=ax[2], legend=True)
    ax[0].set_title("Actual Generation Output")
    ax[1].set_title("Own Consumption")
    ax[2].set_title("Actual Net Generation")


def plot_exchange_data(country_code, ds_exchange, label_for_unit="Power (MW)"):
    country_name = pycountry.countries.get(alpha_2=country_code).name

    fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True, sharex=True)

    dfp = (
        ds_exchange.sel(Importing_Countries=country_code)
        .FlowValue.to_dataset(dim="Exporting_Countries")
        .to_dataframe()
        .dropna(axis=1)
        .drop("Importing_Countries", axis=1)
    )
    dfp = dfp.replace(0, np.nan).dropna(how = 'all', axis = 1)
    dfp.plot(ax=ax[0], kind="area", alpha=0.4, stacked=True)
    # dfp.sum(axis=1).plot(ax=ax[0], c="k")
    ax[0].set_title(f"Import of electricity to {country_name}")
    ax[0].set_ylabel(label_for_unit)

    dfp = (
        ds_exchange.sel(Exporting_Countries=country_code)
        .FlowValue.to_dataset(dim="Importing_Countries")
        .to_dataframe()
        .dropna(axis=1)
        .drop("Exporting_Countries", axis=1)
    )
    dfp = dfp = dfp.replace(0, np.nan).dropna(how = 'all', axis = 1)
    dfp.plot(ax=ax[1], kind="area", alpha=0.4, stacked=True)
    # dfp.sum(axis=1).plot(ax=ax[1], c="k")
    ax[1].set_title(f"Export of electricity from {country_name}")

    df_total_exchange = pd.DataFrame()
    df_total_exchange["Import"] = (
        ds_exchange.sel(Importing_Countries=country_code)
        .sum(dim="Exporting_Countries")
        .to_dataframe()
        .FlowValue
    )

    df_total_exchange["Export"] = (
        ds_exchange.sel(Exporting_Countries=country_code)
        .sum(dim="Importing_Countries")
        .to_dataframe()
        .FlowValue
    )
    df_total_exchange["Net import"] = (
        df_total_exchange.Import - df_total_exchange.Export
    )
    df_total_exchange.plot(ax=ax[2], legend=True, alpha=0.8)
    ax[2].set_title("Electricity exchanges with neighbours")


def plot_generation_and_trade_data(country_code, ds_gen, ds_exchange, label_for_unit="Power (MW)", sharey = True):
    """This function plot the"""
    if country_code == "XK":
        country_name = "Kosovo"
    else:
        country_name = pycountry.countries.get(alpha_2=country_code).name
    print(country_name)

    fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=sharey, sharex=True)
    fig.suptitle(country_name, y=1.05, fontsize=18)

    #Generation data
    dfp = (
        ds_gen.sel(Countries=country_code)
        .ActualGenerationOutput.to_dataset("Energies")
        .to_dataframe()#.dropna(axis=1, how="all")
    )
    dfp = dfp[list_ordered_for_plot]
    dfp.plot(
        ax=ax[0],
        kind="area",
        stacked=True,
        alpha=0.4,
        legend=True,
        color = list_color_for_plot
    )
    ax[0].set_title(f"Generation Output in {country_name}")
    ax[0].set_ylabel(label_for_unit)

    #Import
    dfp = (
        ds_exchange.sel(Importing_Countries=country_code)
        .FlowValue.to_dataset(dim="Exporting_Countries")
        .to_dataframe()
        .dropna(axis=1)
        .drop("Importing_Countries", axis=1)
    )
    dfp = dfp.replace(0, np.nan).dropna(how = 'all', axis = 1)
    dfp.plot(ax=ax[1], kind="area", alpha=0.4, stacked=True)
    # dfp.sum(axis=1).plot(ax=ax[0], c="k")
    ax[1].set_title(f"Import of electricity in {country_name}")

    dfp = (
        ds_exchange.sel(Exporting_Countries=country_code)
        .FlowValue.to_dataset(dim="Importing_Countries")
        .to_dataframe()
        .dropna(axis=1)
        .drop("Exporting_Countries", axis=1)
    )
    dfp = dfp.replace(0, np.nan).dropna(how = 'all', axis = 1)
    dfp.plot(ax=ax[2], kind="area", alpha=0.4, stacked=True)
    # dfp.sum(axis=1).plot(ax=ax[1], c="k")
    ax[2].set_title(f"Export of electricity from {country_name}")
    return fig, ax


def plot_exchange_data_with_external_countries(
    country_code, ds_exchange, label_for_unit="Power (MW)"
):
    country_name = pycountry.countries.get(alpha_2=country_code).name

    fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True, sharex=True)

    dfp = (
        ds_exchange.sel(Countries=country_code)
        .FlowValue.to_dataset(dim="Exporting_Countries")
        .to_dataframe()
        .dropna(axis=1)
        .drop("Countries", axis=1)
    )
    dfp.plot(ax=ax[0], kind="area", alpha=0.4, stacked=True)
    # dfp.sum(axis=1).plot(ax=ax[0], c="k")
    ax[0].set_title(f"Import of electricity to {country_name}")
    ax[0].set_ylabel(label_for_unit)

    dfp = (
        ds_exchange.sel(Exporting_Countries=country_code)
        .FlowValue.to_dataset(dim="Countries")
        .to_dataframe()
        .dropna(axis=1)
        .drop("Exporting_Countries", axis=1)
    )

    dfp.plot(ax=ax[1], kind="area", alpha=0.4, stacked=True)
    # dfp.sum(axis=1).plot(ax=ax[1], c="k")
    ax[1].set_title(f"Export of electricity from {country_name}")

    df_total_exchange = pd.DataFrame()
    df_total_exchange["Import"] = (
        ds_exchange.sel(Countries=country_code)
        .sum(dim="Exporting_Countries")
        .to_dataframe()
        .FlowValue
    )

    df_total_exchange["Export"] = (
        ds_exchange.sel(Exporting_Countries=country_code)
        .sum(dim="Countries")
        .to_dataframe()
        .FlowValue
    )
    df_total_exchange["Net import"] = (
        df_total_exchange.Import - df_total_exchange.Export
    )
    df_total_exchange.plot(ax=ax[2], legend=True, alpha=0.8)
    ax[2].set_title("Electricity exchanges with neighbours")


def plot_prod_conso_and_trade(
    country_code,
    ds_generation,
    ds_consumption_tracked,
    ds_consumption_untracked,
    dict_country_name,
    ax=None,
):
    country_name = dict_country_name[country_code]
    ds_consumption_tracked = ds_consumption_tracked.sum(dim="Production_Countries")
    if type(ax) != np.ndarray:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=True, sharex=True)
    # Dataframe preparation
    dfp = (
        ds_generation.ActualNetGeneration.sel(Countries=country_code)
        .drop("Countries")
        .to_dataset("Energies")
        .to_dataframe()
    )
    dfp[dfp < 0] = 0
    # Subplot
    dfp[list_ordered_for_plot].plot(
        ax=ax[0], kind="area", alpha=0.4, color=list_color_for_plot
    )
    ax[0].set_title(f"Electricity production mix for {country_name}")
    ax[0].set_ylabel("Power (MW)")

    # Dataframe preparation
    dfp = (
        (
            ds_consumption_tracked.sel(Consumption_Countries=country_code)
            * ds_consumption_untracked.Consumption.sel(Countries=country_code)
        )
        .to_dataset("Energies")
        .to_dataframe()
    )
    dfp = dfp[list_ordered_for_plot]
    dfp[dfp < 0] = 0
    dfp[list_ordered_for_plot].plot(
        ax=ax[1], kind="area", alpha=0.4, color=list_color_for_plot
    )
    ax[1].set_title(f"Electricity consumption mix for {country_name}")
    ax[1].set_ylabel("Power (MW)")

    # DataFrame preparation
    dfp = (
        ds_consumption_untracked.sel(Countries=country_code)
        .drop("Countries")
        .to_dataframe()
    )
    dfp[["ActualNetGeneration", "Consumption"]].plot(
        ax=ax[2], alpha=0.3, kind="area", stacked=False
    )
    dfp[["Import", "Export"]].plot(ax=ax[2], alpha=0.8)
    ax[2].set_ylabel("Power (MW)")
    ax[2].set_title(f"Power balance for {country_name}")
    for axi in ax[0:2]:
        handles, labels = axi.get_legend_handles_labels()
        axi.legend(handles[::-1], labels[::-1], loc=2)
    plt.tight_layout()
    return ax


def plot_carbon_footprint(country_code, ds_CI, dict_country_name, ax=None):
    if type(ax) != np.ndarray:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=True, sharex=True)
    dfp = (
        ds_CI.sel(Method = 'Production', Production_Countries=country_code)
        .isel(Consumption_Countries = 0) #independant of the consumption countries for the production country
        .drop_vars(['Method','Production_Countries','Consumption_Countries'])
        .sum(dim= "Emission_phase")
        .to_dataset("Energies")
        .to_dataframe()
    )
    dfp[dfp < 0] = 0
    dfp[list_ordered_for_plot].plot(
        ax=ax[0], kind="area", alpha=0.4, color=list_color_for_plot
    )

    dfp = (
        ds_CI.sel(Method = 'Consumption', Consumption_Countries=country_code).drop_vars(['Method','Consumption_Countries'])
        .sum(dim=["Production_Countries", "Emission_phase"])
        .to_dataset("Energies")
        .to_dataframe()
    )
    dfp[dfp < 0] = 0
    dfp[list_ordered_for_plot].plot(
        ax=ax[1], kind="area", alpha=0.4, color=list_color_for_plot, legend=False
    )

    dfp = pd.DataFrame()
    dfp["Production_mix"] = (
        ds_CI.sel(Method = 'Production', Production_Countries=country_code)
        .isel(Consumption_Countries = 0) #independant of the consumption countries for the production country
        .sum(dim= ["Emission_phase",'Energies'])
        .to_dataframe('CF').CF
    )
    dfp["Consumption_mix"] =(
        ds_CI.sel(Method = 'Consumption', Consumption_Countries=country_code)
        .sum(dim=["Production_Countries", "Emission_phase", 'Energies'])
        .to_dataframe('CF').CF
    )
    dfp.plot(ax=ax[2], kind="line", alpha=0.8)

    ax[0].set_ylabel("gCO2eq/kWh")
    ax2b = ax[2].twinx()
    ax2b.set_ylabel("gCO2eq/MJ")
    ax[2].axhline(y=18.3 * 3.6, ls="--", c="k", label="Threshold*")
    ax[2].legend()
    ax2b.grid(False)
    ax2b.set_ylim([ax[2].get_ylim()[0] / 3.6, ax[2].get_ylim()[1] / 3.6])

    country_name = dict_country_name[country_code]

    ax[0].set_title(f"Carbon intensity of production mix for {country_name}")
    ax[1].set_title(f"Carbon intensity of consumption mix for {country_name}")
    ax[2].set_title("Comparison of carbon intensities")
    for axi in ax:
        handles, labels = axi.get_legend_handles_labels()
        axi.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    return ax


def plot_untracked_consumption(country_code, ds_conso, label_for_unit="Power (MW)"):
    fig, ax = plt.subplots(figsize=(6, 3))
    dfp = ds_conso.sel(Countries=country_code).to_dataframe().drop("Countries", axis=1)
    dfp.plot(ax=ax)
    ax.set_ylabel(label_for_unit)
    country_name = pycountry.countries.get(alpha_2=country_code).name
    ax.set_title(country_name)


def plot_mix_and_carbon_footprint(country_code, ede, freq=None):
    if freq == None:
        freq = ede.freq
    fig, axes = plt.subplots(
        nrows=2, ncols=3, sharex=True, sharey="row", figsize=(16, 8)
    )
    plot_prod_conso_and_trade(
        country_code=country_code,
        ds_generation=ede.ds_generation_grouped.resample(DateTime=freq).mean(),
        ds_consumption_tracked=ede.ds_consumption_tracked_grouped.resample(
            DateTime=freq
        ).mean(),
        ds_consumption_untracked=ede.ds_consumption_untracked.resample(
            DateTime=freq
        ).mean(),
        dict_country_name=ede.dict_country_name,
        ax=axes[0],
    )
    plot_carbon_footprint(
        country_code=country_code,
        ds_CI=ede.ds_CI_grouped.resample(DateTime=freq).mean(),
        dict_country_name=ede.dict_country_name,
        ax=axes[1],
    )


def plot_check_exchange_data(countries, ede):
    """Take a list of two countries as input"""
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    ede.ds_exchange.sel(
        Exporting_Countries=countries[0], Importing_Countries=countries[1]
    ).resample(DateTime="D").mean().to_dataframe().FlowValue.plot(
        kind="line", ax=ax[0], alpha=0.4
    )
    ax[0].set_title(
        f"{pycountry.countries.get(alpha_2=countries[0]).name} -> {pycountry.countries.get(alpha_2=countries[1]).name}"
    )
    ede.ds_exchange.sel(
        Exporting_Countries=countries[1], Importing_Countries=countries[0]
    ).resample(DateTime="D").mean().to_dataframe().FlowValue.plot(
        kind="line", ax=ax[1], alpha=0.4
    )
    ax[1].set_title(
        f"{pycountry.countries.get(alpha_2=countries[1]).name} -> {pycountry.countries.get(alpha_2=countries[0]).name}"
    )
    ax[0].set_ylabel("Power Exchange (MW)")
    ax[1].set_ylabel("Power Exchange (MW)")
    # plt.savefig(f"plot/Exchange_{countries[0]}_{countries[1]}.png")
    return fig, ax
