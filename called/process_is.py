import random as rd
from time import time

import numpy as np
import pandas as pd
from called.utile import make_save_dir, print_progress, print_progress_bar


def create_dirs(save_dir, args):
    """Create directories and csv files for each instance

    Args:
        save_dir (str): the save directory
        args (argparse.Namespace): args of the program
    """
    for instance in range(args.n_instances_pre + 1, args.n_instances + 1):
        save_dir_instance = save_dir + "INST" + str(instance) + "/"
        make_save_dir(save_dir_instance, args)
        with open(save_dir_instance+"labels.csv", "a", encoding="utf8") as file:
            file.write("Name,Date,Leadtime,Member,Gigafile,Localindex,Importance\n")
            file.close()

def importance(grid, parameters, args):
    """Compute the importance of a grid

    Args:
        grid (numpy.array): a numpy array grid with several channels of shape = [4, x, y]
        parameters (tuple[float]): Parameters of importance sampling (q_min, m, p, q, s_rr, s_w)
        args (argparse.Namespace): args of the program
    Returns:
        float: the importance. If greater than 1, return 1
    """
    q_min, m, p, q, s_rr, s_w = parameters
    i_s_rr = 1-np.exp(-grid[0]/s_rr)
    i_s_rr = np.expand_dims(i_s_rr, axis=0)
    grid = np.append(grid, i_s_rr, axis=0)
    i_s_rr_sum = grid[4].sum()
    if args.wind_importance:
        w = np.sqrt(grid[1]**2+grid[2]**2)
        i_s_w = (1-np.exp(-w/s_w))
        i_s_w = np.expand_dims(i_s_w, axis=0)
        grid = np.append(grid, i_s_w, axis=0)
        i_s_w_sum = grid[5].sum()
    grid_size = grid.shape[1]*grid.shape[2] # shape = [5 or 6, x, y]
    if args.wind_importance:
        return q_min + (m / grid_size) * (p * i_s_rr_sum + q * i_s_w_sum)
    return q_min + (m / grid_size) * (p * i_s_rr_sum)

def importance_sig(grid, parameters, args):
    """Compute the importance of a grid

    Args:
        grid (numpy.array): a numpy array grid with several channels of shape = [4, x, y]
        parameters (tuple[float]): Parameters of importance sampling (q_min, m, p, q, s_rr, s_w)
        args (argparse.Namespace): args of the program
    Returns:
        float: the importance. If greater than 1, return 1
    """
    q_min, m, p, q, s_rr, s_w = parameters
    slope = 1
    i_s_rr = 1 / (1 + np.exp(-(grid[0] - s_rr) / slope))
    i_s_rr = np.expand_dims(i_s_rr, axis=0)
    grid = np.append(grid, i_s_rr, axis=0)
    i_s_rr_sum = grid[4].sum()
    if args.wind_importance:
        w = np.sqrt(grid[1]**2+grid[2]**2)
        i_s_w = 1 / (1 + np.exp(-(w - s_w) / slope))
        i_s_w = np.expand_dims(i_s_w, axis=0)
        grid = np.append(grid, i_s_w, axis=0)
        i_s_w_sum = grid[5].sum()
    grid_size = grid.shape[1]*grid.shape[2] # shape = [5 or 6, x, y]
    if args.wind_importance:
        return q_min - m / (1 + np.exp(s_rr / slope)) + (m / grid_size) * (p * i_s_rr_sum + q * i_s_w_sum)
    return q_min - m / (1 + np.exp(s_rr / slope)) + (m / grid_size) * (p * i_s_rr_sum)

def sample_for_instance(save_dir, p_importance, row, args):
    """For each instance, sample and writes data in the csv file

    Args:
        save_dir (str): the save directory
        p_importance (float): the importance computed
        row (_type_): A row from a dataframe
        args (argparse.Namespace): args of the program
    """
    for instance in range(args.n_instances_pre + 1, args.n_instances + 1):
        save_dir_instance = save_dir + "INST" + str(instance) + "/"
        p_uniform = rd.uniform(0, 1)
        if p_uniform <= p_importance:
            with open(save_dir_instance+"labels.csv", "a", encoding="utf8") as file:
                file.write(row["Name"]+","+row["Date"]+","+str(row["Leadtime"]) +","+str(row["Member"])+","+ str(row["Gigafile"]) + "," + str(row["Localindex"]) + "," +str(p_importance)+"\n")
                file.close()

def importance_sampling(parameters, dirs, args):
    """Compute importance sampling with parameters parameters

    Args:
        parameters (tuple[float]): Parameters of importance sampling (q_min, m, p, q, s_rr, s_w)
        dirs (str): Directories with which data interact (csv_dir, data_dir, save_dir)
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 1: print("Importance sampling...")
    csv_dir, data_dir, save_dir = dirs
    create_dirs(save_dir, args)

    dataframe = pd.read_csv(csv_dir + "labels.csv")
    dataframe = dataframe.reset_index()
    n_grid = len(dataframe)
    current_patch = 0
    n_patch = dataframe.iloc[-1]["Gigafile"]
    start_time = time()
    for index, row in dataframe.iterrows():
        if args.verbose >= 2 and (index + 1) % (n_grid // args.refresh) == 0:
            print_progress(index, n_grid, start_time)
        if row["Gigafile"] != current_patch:
            current_patch += 1
            if args.verbose >= 2: print("\nLoading patch", current_patch, "/", n_patch, "...")
            l_grid = np.load(data_dir + str(current_patch) + ".npy")
            if args.verbose >= 2: print("Done.\n")
        grid = l_grid[row["Localindex"]]
        if args.verbose >= 3: print("Grid", index, "out of", n_grid)
        if args.sigmoid:
            p_importance = importance_sig(grid, parameters, args)
        else:
            p_importance = importance(grid, parameters, args)
        sample_for_instance(save_dir, p_importance, row, args)
    if args.verbose >= 1: print("DONE importance sampling.")


if __name__ == "__main__":
    pass
