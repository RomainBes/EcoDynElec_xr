from ecodynelec.settings import *
from ecodynelec.parameter import Parameter
from ecodynelec.downloading import download
from ecodynelec.energy_grouping import get_net_exchange
from ecodynelec.neighbours import find_neighbours_for_countries
from ecodynelec.tracking import compute_tracking
from ecodynelec.energy_grouping import add_energy_groups
import os
import pandas as pd
import xarray as xr
import numpy as np
import pycountry
from countrygroups import EUROPEAN_UNION


class EcoDynElec_xr:

    def __init__(self, year) -> None:
        """Create folter for data and plt"""
        self.datapath = datapath
        self.plotpath = plotpath
        self.username = username
        self.password = password
        self.year = year

        # Create folders
        for folder in [datapath, plotpath]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        # Create subfolders
        for subfolder in ["generation", "exchange", "autosave", "results"]:
            folder = f"{datapath}/{subfolder}"
            if not os.path.exists(folder):
                os.makedirs(folder)

    def set_download_config(self):
        """Set the configuration data for direct downloading"""
        ### Initialize the configuration
        config = Parameter()
        ### Set the dates (to select files to download)
        config.start = f"{self.year}-01-01 00:00"
        config.end = f"{self.year}-12-31 00:00"
        config.freq = "H"  # "15min"

        ### Indicate where to save the generation data
        config.path.generation = f"{self.datapath}/generation/"
        ### Indicate where to save the exchange data
        config.path.exchanges = f"{self.datapath}/exchange/"
        config.path.savedir = f"{self.datapath}//autosave/"

        ### Configure the server connection
        config.server.useServer = True  # Specifically ask to download data
        config.server.host = "sftp-transparency.entsoe.eu"  # This server is already set per default after initialization
        config.server.port = (
            22  # This port is already set per default after initialization
        )

        # Personal data to connect to the ENTSO-E API
        config.server.username = self.username
        config.server.password = self.password

        self.config = config
        return self.config

    def download_ENTSOE_data(self, force=False):
        """Download the ENTSOE data"""
        # Ask confirmation to avoid downloading heavy data for nothing.
        list_file = os.listdir(self.config.path.generation)
        if force or len(list_file) == 0:
            answer = input("Do you want to download the data? \nIf so reply by 'yes'.")
            if answer.lower() in ["y", "yes"]:
                download(
                    config=self.config, is_verbose=True
                )  # is_verbose does display some text while downloading
            else:
                print("Data download was not confirmed.")
        else:
            print(
                "Data were not download or redownloaded because they already exists. You have to specify 'force = True' if you want to redonwload them."
            )

    def format_generation_data(self, final_file_name):
        """Format csv files downloaded from the ENTSOE website to nice xarray dataset"""
        print(
            "Loading and formatting the data from the csv into a more efficient xarray dataset..."
        )
        list_files = os.listdir(self.config.path.generation)
        list_files = [file for file in list_files if file.endswith(".csv")]
        list_files = [
            file for file in list_files if int(file.split("_")[0]) == self.year
        ]

        list_ds = []
        for file in list_files:
            print(file)
            # Load the heavy csv file
            df = pd.read_csv(self.config.path.generation + file, sep="\t")
            # Filter data to get only country level data
            df = df[df.AreaTypeCode == "CTY"]
            # Replace nan by zero not to have missing column later on
            df["ActualConsumption"] = df.ActualConsumption.replace(np.nan, 0)
            # Reshape the dataframe
            df = df.pivot_table(
                index="DateTime",
                columns=["MapCode", "ProductionType"],
                values=["ActualGenerationOutput", "ActualConsumption"],
            )
            df.columns.names = ["Consumption", "Countries", "Energies"]
            # Define an hourly datetime index
            df.index = pd.to_datetime(df.index)
            df = df.resample("H").mean()
            # Convert dataframe to an xarray dataset
            ds = df.stack([1, 2]).to_xarray()
            list_ds.append(ds)
        ds = xr.concat(list_ds, dim="DateTime")
        ds.to_netcdf(final_file_name)

    def get_generation_data(self, force=False):
        """Return the generation dataset and create it if needed"""
        final_file_name = f"{self.config.path.generation}/ds_generation_{self.year}.nc"
        if not os.path.exists(final_file_name) or force:
            self.format_generation_data(final_file_name=final_file_name)
        ds = xr.open_dataset(final_file_name)
        ds["ActualNetGeneration"] = ds.ActualGenerationOutput.fillna(
            0
        ) - ds.ActualConsumption.fillna(0)
        self.ds_generation = ds

        # Set grouped data as attribute as well
        ds_generation_grouped = add_energy_groups(self.ds_generation)
        ds_generation_grouped = ds_generation_grouped.sum(dim="Subenergies")
        self.ds_generation_grouped = ds_generation_grouped
        return ds

    def format_exchange_data(self, final_file_name):
        print(
            "Loading and formatting the data from the csv into a more efficient xarray dataset..."
        )
        list_files = os.listdir(self.config.path.exchanges)
        list_files = [file for file in list_files if file.endswith(".csv")]
        list_files = [
            file for file in list_files if int(file.split("_")[0]) == self.year
        ]
        list_ds = []
        for file in list_files:
            print(file)
            # Load heavy csv data
            df = pd.read_csv(self.config.path.exchanges + file, sep="\t")
            # Filter to get only country level data
            df = df[(df.OutAreaTypeCode == "CTY") & (df.InAreaTypeCode == "CTY")]
            # Reshape the dataframe
            df = df.pivot_table(
                index="DateTime",
                columns=["InMapCode", "OutMapCode"],
                values=["FlowValue"],
            )
            df.columns.names = [
                "Flows",
                "Importing_Countries",
                "Exporting_Countries",
            ]
            # Defining an hourly datetime index
            df.index = pd.to_datetime(df.index)
            df = df.resample("H").mean()
            # Convert to an xarray dataset
            ds = df.stack([1, 2]).to_xarray()
            list_ds.append(ds)
        ds = xr.concat(list_ds, dim="DateTime")
        ds.to_netcdf(final_file_name)

    def get_exchange_data(self, force=False):
        """Format csv files downloaded from the ENTSOE website to nice xarray dataset"""
        final_file_name = f"{self.config.path.exchanges}/ds_exchange_{self.year}.nc"
        if not os.path.exists(final_file_name) or force:
            self.format_exchange_data(final_file_name=final_file_name)
        ds = xr.open_dataset(final_file_name)
        self.ds_exchange = ds
        return self.ds_exchange

    def get_consumption_data(self):
        # Generation data in MW
        ds_prod = self.ds_generation
        ds_prod = ds_prod.sum(dim="Energies")
        ds_prod.attrs["unit"] = "MW"
        # Exchanges data in MW
        ds_exchange = self.ds_exchange
        ds_exchange.attrs["unit"] = "MW"
        ds = ds_prod.copy()
        # Calculate trade total
        ds["Import"] = (
            ds_exchange.sum(dim="Exporting_Countries")
            .rename({"Importing_Countries": "Countries"})
            .FlowValue
        )
        ds["Export"] = (
            ds_exchange.sum(dim="Importing_Countries")
            .rename({"Exporting_Countries": "Countries"})
            .FlowValue
        )
        ds["Net_export"] = ds.Export - ds.Import
        ds["Consumption"] = ds.ActualGenerationOutput - ds.Net_export
        ds.to_netcdf(
            f"{self.datapath}/results/ds_consumption_untracked_MW_{self.year}.nc"
        )
        ds_prod.close()
        ds.close()
        self.ds_consumption_untracked = ds
        return self.ds_consumption_untracked

    def get_dataset_energy(
        self,
        freq,
        list_countries,
        net_generation=True,
        net_exchange=False,
        n_hours: int = 2,
    ):
        # Load data and resample data
        ds_prod = self.ds_generation
        ds_prod = ds_prod.ffill(dim="DateTime", limit=n_hours).bfill(
            dim="DateTime", limit=n_hours
        )
        ds_prod = ds_prod.resample(DateTime=freq).mean()

        ds_exchange = self.ds_exchange
        ds_exchange = ds_exchange.ffill(dim="DateTime", limit=n_hours).bfill(
            dim="DateTime", limit=n_hours
        )
        ds_exchange = ds_exchange.resample(DateTime=freq).mean()

        # Filter data for list of countries, and group external neighbours together in the exchanges dataset
        ds_prod = ds_prod.sel(Countries=list_countries)
        self.list_energies = list(ds_prod.Energies.values)

        if net_exchange:
            ds_exchange = get_net_exchange(ds_exchange=ds_exchange)

        # Get external neighbours
        list_external_neighbours = find_neighbours_for_countries(
            list_countries=list_countries,
            ds_exchange=ds_exchange.resample(DateTime="M").max(),
        )

        # Exchange dasaset for external neighbours only
        ds_exchange_external_neighbours = xr.concat(
            [
                ds_exchange.sel(Exporting_Countries=list_external_neighbours).sum(
                    dim="Exporting_Countries"
                )
            ],
            dim="Exporting_Countries",
        )
        ds_exchange_external_neighbours = ds_exchange_external_neighbours.assign_coords(
            coords={"Exporting_Countries": ["External_neighbours"]}
        )
        # Merging list of countries and external neighbours together
        ds_exchange_with_external_neighbours = xr.merge(
            [
                ds_exchange.sel(Exporting_Countries=list_countries),
                ds_exchange_external_neighbours,
            ]
        )

        # Rename to be consistent with ds_prod dimensions name
        ds_exchange_with_external_neighbours = (
            ds_exchange_with_external_neighbours.rename(
                {"Importing_Countries": "Countries"}
            )
        )
        ds_exchange_with_external_neighbours = ds_exchange_with_external_neighbours.sel(
            Countries=list_countries
        )
        da_exchange_with_external_neighbours = (
            ds_exchange_with_external_neighbours.FlowValue.rename(
                {"Exporting_Countries": "Electricity_sources"}
            )
        )
        da_exchange_with_external_neighbours.name = "value_MW"

        # Select the right generation data, net or not
        if net_generation:
            ds_prod["ActualNetGeneration"] = ds_prod.ActualGenerationOutput.fillna(
                0
            ) - ds_prod.ActualConsumption.fillna(0)
            da_prod = ds_prod.ActualNetGeneration.rename(
                {"Energies": "Electricity_sources"}
            )
        else:
            da_prod = ds_prod.ActualGenerationOutput.rename(
                {"Energies": "Electricity_sources"}
            )
        da_prod.name = "value_MW"

        # Merge
        ds_energy = xr.merge([da_prod, da_exchange_with_external_neighbours])
        # Normalise
        ds_energy["value_normalised"] = ds_energy.value_MW / ds_energy.value_MW.sum(
            dim="Electricity_sources"
        )
        self.ds_energy = ds_energy
        return self.ds_energy

    def get_dict_country_name(self):
        list_countries = list(self.ds_generation.Countries.values)
        dict_country_name = {
            country_code: pycountry.countries.get(alpha_2=country_code).name
            for country_code in list_countries
            if country_code != "XK"
        }
        dict_country_name["XK"] = "Kosovo"
        self.dict_country_name = dict_country_name
        return self.dict_country_name

    def fill_grid_loss_data(self, ds, filling_quantile=0.9):
        # Missing countries
        list_missing_countries = list(
            set(self.ds_consumption_tracked.Production_Countries.values)
            - set(ds.Countries.values)
        )
        print(f"Missing countries (filled by quantile {filling_quantile}):")
        print(list_missing_countries)
        df = ds.to_dataframe("value")
        # Add missing countries
        df_missing = pd.DataFrame(columns=["value"], index=list_missing_countries)
        df_missing.index.name = df.index.name
        df = pd.concat([df, df_missing])
        df = df.fillna(df.quantile(filling_quantile))
        ds = df.value.to_xarray()
        ds = ds.sel(Countries=self.ds_consumption_tracked.Production_Countries.values)
        return ds

    def get_grid_loss_data(self, grid_losses, filling_quantile=0.9):
        ds_grid_loss = xr.load_dataarray("data/ds_eurostat_loss.nc")
        ds_grid_loss = (
            ds_grid_loss.sel(nrg_bal=grid_losses).ffill("DateTime").isel(DateTime=-1)
        )
        ds_grid_loss = ds_grid_loss.drop_vars(["nrg_bal", "Energies", "DateTime"])
        ds_grid_loss = self.fill_grid_loss_data(
            ds_grid_loss, filling_quantile=filling_quantile
        )
        self.ds_grid_loss = ds_grid_loss
        return self.ds_grid_loss

    def track_mix(
        self,
        freq,
        list_countries=None,
        net_generation=True,
        net_exchange=False,
        grid_loss="TRANSL",
        force=False,
        grouped=True,
    ):
        # File name
        filepath = f"{self.datapath}/results/ds_mix_{self.year}_{freq}"
        filepath += net_generation * "_net_generation"
        filepath += net_exchange * "_net_exchange"
        # filepath += grid_losses * "_grid_losses"
        filepath += ".nc"

        # set list countries if None
        if list_countries == None:
            list_countries = list(self.ds_generation.Countries.values)
        self.list_countries = list_countries
        self.freq = freq
        self.net_generation = net_generation
        self.net_exchange = net_exchange

        if not os.path.exists(filepath) or force:
            print("Loading data...")
            ds_energy = self.get_dataset_energy(
                freq=freq,
                list_countries=list_countries,
                net_generation=net_generation,
                net_exchange=net_exchange,
            )
            self.ds_energy = ds_energy
            print("Data loaded.")
            print("Let's track the electricity mix...")
            ds_mix = compute_tracking(
                ds_energy=ds_energy,
                list_energies=self.list_energies,
                list_countries=list_countries,
            )
            ds_mix.to_netcdf(filepath)

        self.ds_consumption_tracked = xr.open_dataarray(
            filepath)
        self.ds_consumption_tracked = self.ds_consumption_tracked.chunk(chunks = {'DateTime':self.ds_consumption_tracked.DateTime.size})
        if grid_loss != None:
            ds_grid_loss = self.get_grid_loss_data(grid_loss, filling_quantile=0.9)
            ds_grid_loss = ds_grid_loss.rename({"Countries": "Consumption_Countries"})
            self.ds_consumption_tracked = self.ds_consumption_tracked / (
                1 - ds_grid_loss
            )

        # Group by energy groups
        if grouped:
            print("Calculating energy groups...")
            ds_consumption_tracked_grouped = add_energy_groups(
                self.ds_consumption_tracked
            )
            ds_consumption_tracked_grouped = ds_consumption_tracked_grouped.sum(
                dim="Subenergies"
            )
            self.ds_consumption_tracked_grouped = ds_consumption_tracked_grouped
        return self.ds_consumption_tracked

    def fill_efficiency_missing_data(self, ds, filling_quantile=0.10):
        # Missing countries
        list_missing_countries = list(
            set(self.ds_consumption_tracked.Production_Countries.values)
            - set(ds.Production_Countries.values)
        )
        print(f"Missing countries (filled by quantile {filling_quantile}):")
        print(list_missing_countries)
        df = ds.to_dataset("Energies").to_dataframe()
        # Removing negative values due to missing data
        df[df < 0] = np.nan
        # Add missing countries
        df_missing = pd.DataFrame(columns=df.columns, index=list_missing_countries)
        df_missing.index.name = df.index.name
        df = pd.concat([df, df_missing])
        df.columns.name = "Energies"
        df = df.fillna(df.quantile(filling_quantile))
        ds = df.unstack().to_xarray()
        ds = ds.sel(
            Production_Countries=self.ds_consumption_tracked.Production_Countries.values
        )
        return ds

    def get_efficiency_data(self, filling_quantile=0.5):
        """Return the efficiency dataset based on EUROSTAT data"""
        ds_efficiency = xr.open_dataarray("data/ds_efficiency_combined.nc")
        ds_efficiency = self.fill_efficiency_missing_data(
            ds=ds_efficiency, filling_quantile=filling_quantile
        )
        self.ds_efficiency = ds_efficiency
        return self.ds_efficiency

    def get_EU_countries(self):
        list_EU_countries = [
            pycountry.countries.get(alpha_3=country).alpha_2
            for country in EUROPEAN_UNION
        ]
        list_EU_countries = [country for country in list_EU_countries if country not in ['MT','CY']]
        self.list_EU_countries = list_EU_countries

    def get_EU_value(self, filling_quantile=0.5):
        if not hasattr(self, 'list_EU_countries'):
            self.get_EU_countries()
        ds_final_energy = self.ds_generation.ActualNetGeneration.sel(
            Countries=self.list_EU_countries
        ).rename({"Countries": "Production_Countries"})
        ds_efficiency = self.get_efficiency_data(filling_quantile=filling_quantile)
        ds_primary_energy = ds_final_energy / ds_efficiency
        ds_EF = self.get_emission_factor_data()
        ds_CI = ds_primary_energy * ds_EF
        EU_value = ds_CI.sum(
            dim=["Energies", "Production_Countries", "DateTime", "Emission_phase"]
        ) / ds_final_energy.sum(dim=["Energies", "Production_Countries", "DateTime"])
        self.EU_value = float(EU_value.values)
        return self.EU_value

    def get_emission_factor_data(self):
        """Return the emission factor data"""
        if not hasattr(self, "ds_EF"):
            ds_EF = xr.open_dataarray("data/ds_emissions_factors.nc")
            ds_EF = ds_EF.drop("Energies_EUROSTAT")
            ds_EF = ds_EF.rename({"Energies_ENTSOE": "Energies"})
            self.ds_EF = ds_EF
        return self.ds_EF

    def get_carbon_footprint_consumed(self):
        """Calculate the carbon footprint of electricity mix"""
        ds_final_energy_consumed = self.ds_consumption_tracked
        ds_primary_energy_consumed = ds_final_energy_consumed / self.ds_efficiency
        ds_CI_consumed = ds_primary_energy_consumed * self.ds_EF
        self.ds_CI_consumed = ds_CI_consumed
        return ds_CI_consumed
    
    def get_carbon_footprint_produced(self):
        """Calculate the carbon footprint of electricity mix"""
        ds_final_energy_produced = self.ds_generation.rename({"Countries": "Production_Countries"}).resample(DateTime = self.freq).sum()
        #Normalisation per kWh produced in one country at each timestep
        ds_final_energy_produced = ds_final_energy_produced / ds_final_energy_produced.sum('Energies')
        ds_primary_energy_produced = ds_final_energy_produced / self.ds_efficiency
        ds_CI_produced = ds_primary_energy_produced * self.ds_EF
        self.ds_CI_produced = ds_CI_produced
        return ds_CI_produced
    
    def get_carbon_footprint(self, grouped=True, filling_quantile = 0.5):
        """Calculate both the carbon footprint of production and consumption electricity mix"""
        # Load and format emission factor dataset
        self.get_emission_factor_data()
        self.get_efficiency_data(filling_quantile)
        ds_CI_produced = self.get_carbon_footprint_produced()
        ds_CI_consumed = self.get_carbon_footprint_consumed()
        ds_CI = xr.concat([ds_CI_produced.ActualNetGeneration, 
                           ds_CI_consumed], dim = 'Method')
        ds_CI.name = 'Carbon_Intensity'
        ds_CI = ds_CI.assign_coords(coords={"Method": ['Production','Consumption']})
        self.ds_CI = ds_CI
        if grouped:
            ds_CI_grouped = add_energy_groups(ds_CI)
            ds_CI_grouped = ds_CI_grouped.sum(dim="Subenergies")
            self.ds_CI_grouped = ds_CI_grouped
        
    def compute(self):
        self.ds_consumption_tracked = self.ds_consumption_tracked.compute()
        self.ds_consumption_tracked_grouped = self.ds_consumption_tracked_grouped.compute()
        self.ds_CI = self.ds_CI.compute()
        self.ds_CI_grouped = self.ds_CI_grouped.compute()