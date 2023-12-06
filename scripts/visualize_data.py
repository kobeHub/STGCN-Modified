# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:48:02 2023

@author: starm
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


def plot_stations(save_path):
    file = open("./dataset/PeMSD7_M_Station_Info.csv")

    csvreader = csv.reader(file)
    lat = []
    lon = []
    for row in csvreader:
        lat.append(row[5])
        lon.append(row[6])

    lat.pop(0)
    lon.pop(0)

    for i in range(1, 229):
        i = i - 1
        lat[i] = float(lat[i])
        lon[i] = float(lon[i])
    # Create a map with Cartopy
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot the points on the map
    ax.scatter(
        lon,
        lat,
        color="red",
        marker="o",
        label="Sensor Stations",
        transform=ccrs.Geodetic(),
    )
    # ax.set_extent([-119, -117, 33, 35], crs=ccrs.PlateCarree())
    # Add coastlines for better context
    # Add more details to the map
    ax.add_feature(cfeature.COASTLINE, linestyle=":")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="lightblue")
    ax.add_feature(cfeature.RIVERS, edgecolor="blue", facecolor="lightblue")
    ax.add_feature(cfeature.STATES, linestyle="--", edgecolor="black", facecolor="none")

    # Add gridlines
    ax.gridlines(draw_labels=True, linestyle="--")
    plt.legend()

    # Set plot title
    plt.title("PeMS sensor network in District 7 of California")
    plt.savefig(save_path, dpi=300)
    plt.clf()

    # Show the plot
    # plt.show()


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    """
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f"ERROR: input file was not found in {file_path}.")

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.0
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


def plot_matrix(matrix, save_path):
    plt.imshow(matrix, cmap="Blues", vmin=0.5, vmax=1.0, interpolation="nearest")
    plt.title("The correlations across stations")
    plt.xlabel("Station ID")
    plt.ylabel("Station ID")
    plt.colorbar()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    plt.clf()


# Example 2D matrix
# matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
if __name__ == "__main__":
    plot_stations("./figures/stations.png")
    W = weight_matrix("./dataset/PeMSD7_W_228.csv")
    plot_matrix(W, "./figures/datapoint.png")
