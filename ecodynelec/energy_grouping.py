import xarray as xr

dict_energy_groups = {
    "Fossil": [
        "Fossil Brown coal/Lignite",
        "Fossil Coal-derived gas",
        "Fossil Gas",
        "Fossil Hard coal",
        "Fossil Oil",
        "Fossil Oil shale",
        "Fossil Peat",
    ],
    "Nuclear": ["Nuclear"],
    "Hydro": [
        "Hydro Pumped Storage",
        "Hydro Run-of-river and poundage",
        "Hydro Water Reservoir",
    ],
    "Solar": ["Solar"],
    "Wind": ["Wind Offshore", "Wind Onshore"],
    "Other Renewable": ["Biomass", "Geothermal", "Other renewable", "Waste", "Marine"],
    "Other": ["Other"],
}


def add_energy_groups(ds):
    """This function add a dimension to groupby energy groups"""
    list_energy_groups = [key for key in dict_energy_groups]
    ds = ds.rename({"Energies": "Subenergies"})
    ds = xr.concat(
        [
            ds.sel(Subenergies=dict_energy_groups[energy_family])
            for energy_family in list_energy_groups
        ],
        dim="Energies",
    )
    ds = ds.assign_coords(coords={"Energies": list_energy_groups})
    return ds


def get_net_exchange(ds_exchange):
    """Return net exchanges dataset"""
    ds_exchange_reversed = ds_exchange.rename(
        {
            "Importing_Countries": "Exporting_Countries",
            "Exporting_Countries": "Importing_Countries",
        }
    )
    ds_exchange = (
        (ds_exchange - ds_exchange_reversed)
        .where(ds_exchange > ds_exchange_reversed)
        .fillna(0)
    )
    return ds_exchange
