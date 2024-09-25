import numpy as np

def find_neighbours_for_country(country_code, ds_exchange):
    list_neighboors_exporting_to_country = (
        ds_exchange.sel(Importing_Countries=country_code)
        .FlowValue.to_dataset(dim="Exporting_Countries")
        .to_dataframe()
        .replace(0, np.nan)
        .dropna(axis=1, how="all")
        .columns[0:-1]
    )

    list_neighboors_importing_to_country = (
        ds_exchange.sel(Importing_Countries=country_code)
        .FlowValue.to_dataset(dim="Exporting_Countries")
        .to_dataframe()
        .replace(0, np.nan)
        .dropna(axis=1, how="all")
        .columns[0:-1]
    )

    list_neighbours = list(
        set(list_neighboors_exporting_to_country)
        | set(list_neighboors_importing_to_country)
    )

    return list_neighbours

def find_neighbours_for_countries(list_countries, ds_exchange):
    list_neightbours = []
    for country in list_countries:
        list_neightbours_country = find_neighbours_for_country(country_code=country, ds_exchange=ds_exchange)
        list_neightbours +=list_neightbours_country
    list_neighbours_outside_list = list(
    set(list_neightbours)
    - set(list_countries)
)
    return list_neighbours_outside_list