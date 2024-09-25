import pandas as pd
import numpy as np
import xarray as xr
import scipy
import pyprind


def build_technosphere_matrix(ds_energy_t, list_energies, list_countries):
    # First piece of A matrix
    df1 = (
        ds_energy_t.value_normalised.sel(
            Electricity_sources=list(ds_energy_t.Countries.values)
            + ["External_neighbours"]
        )
        .to_dataset("Countries")
        .to_dataframe()
    )

    # Second part of matrix
    df = ds_energy_t.value_normalised.sel(
        Electricity_sources=list_energies
    ).to_dataframe()
    # Iteration over coutnries to reshape the matrix as explained: https://ecodynelec.readthedocs.io/en/latest/structure/tracking.html
    for country in list_countries:
        df[country] = df.index.get_level_values(0)
        df[country] = df[country] == country
        df[country] = df[country] * df.value_normalised
    df2 = df[list_countries]

    # Concat the two piece
    df_A = pd.concat([df1, df2], axis=0)

    # Add zero to make the matrix square
    df_A_square = pd.DataFrame(index=df_A.index, columns=df_A.index)
    df_A_square.loc[df_A.index, df_A.columns] = df_A
    df_A_square = df_A_square.fillna(0)

    return df_A_square


def invert_technology_matrix(df_A):
    df_identity = pd.DataFrame(
        data=np.identity(df_A.shape[0]),
        index=df_A.index,
        columns=df_A.columns,
    )

    df_matrix_to_invert = df_identity - df_A

    if scipy.sparse.issparse(df_matrix_to_invert):
        M = scipy.sparse.linalg.inv(df_matrix_to_invert)
    else:
        M = scipy.linalg.inv(df_matrix_to_invert)
    df_mix = pd.DataFrame(data=M, index=df_A.index, columns=df_A.columns)
    return df_mix


def format_to_xarray_dataset(df_mix, list_countries):
    df_mix = df_mix[list_countries].drop(list_countries)
    df_mix = df_mix.rename({"External_neighbours": ("External_neighbours", "Grid")})
    df_mix.index = pd.MultiIndex.from_tuples(
        df_mix.index, names=["Production_Countries", "Energies"]
    )
    df_mix.columns.name = "Consumption_Countries"
    ds_mix_t = df_mix.stack().to_xarray()
    return ds_mix_t


def compute_tracking_at_time_t(ds_energy_t, list_energies, list_countries):
    df_A = build_technosphere_matrix(
        ds_energy_t, list_energies=list_energies, list_countries=list_countries
    )
    df_mix = invert_technology_matrix(df_A)
    ds_mix_t = format_to_xarray_dataset(df_mix, list_countries)
    return ds_mix_t


def compute_tracking(ds_energy, list_energies, list_countries):
    bar = pyprind.ProgBar(ds_energy.DateTime.values.size, monitor=True, track_time=True)
    list_ds_mix_t = []
    for DateTime in ds_energy.DateTime.values:
        ds_energy_t = ds_energy.sel(DateTime=DateTime).drop("DateTime")
        ds_mix_t = compute_tracking_at_time_t(
            ds_energy_t, list_energies=list_energies, list_countries=list_countries
        )
        list_ds_mix_t.append(ds_mix_t)
        bar.update()
    ds_mix = xr.concat(list_ds_mix_t, dim="DateTime")
    ds_mix = ds_mix.assign_coords(coords={"DateTime": ds_energy.DateTime.values})
    return ds_mix
