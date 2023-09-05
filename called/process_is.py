import os
import random as rd
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

try:
    from called.utile import make_save_dir, print_progress, print_progress_bar
except ModuleNotFoundError:
    from utile import make_save_dir, print_progress, print_progress_bar



def create_dirs(save_dir, args):
    """Create directories and csv files for each instance

    Args:
        save_dir (str): the save directory
        args (argparse.Namespace): args of the program
    """
    for instance in range(1, args.n_instances + 1):
        save_dir_instance = f"{save_dir}INST{instance}/"
        make_save_dir(save_dir_instance, args)
        with open(f"{save_dir_instance}labels.csv", "w", encoding="utf8") as file:
            file.write(f"Name,Date,Leadtime,Member,Gigafile,Localindex,Importance\n")

def compute_c(s_rr, q_min, m, l_c):
    filter_func = lambda c: m + ((q_min - m) / np.tanh(-s_rr / c)) * np.tanh((1 - s_rr) / c) - 1
    l_c = np.abs(fsolve(filter_func, l_c))
    if np.abs(l_c[0] - l_c[1]) < 0.01:
        return l_c[0]
    raise RuntimeError(f"fsolve didn't converge: {l_c}")

def filter(x, s_rr, q_min, m, c):
    return m + ((q_min - m) / np.tanh(-s_rr / c)) * np.tanh((x - s_rr / c))
    
def rough_filter(x, s_rr, q_min, m):
    return m * (x >= s_rr) + q_min * (x < s_rr)

def ravuri_filter(x, s_rr, q_min, m):
    return q_min + m * (1 - np.exp(-x/s_rr))

def importance(grid, parameters, gridshape, variable, args):
    """Compute the importance of a grid

    Args:
        grid (numpy.array): a numpy array grid with several channels of shape = [4, x, y]
        parameters (tuple[float]): Parameters of importance sampling (s_rr, q_min, m, c)
        args (argparse.Namespace): args of the program
    Returns:
        float: the importance.
    """
    if args.verbose >= 5 and not args.progress_bar: print(f"Computing importance...")
    s_rr, q_min, m, c = parameters
    if args.rough:
        i_grid = rough_filter(grid[variable], s_rr, q_min, m)
    elif args.ravuri:
        i_grid = ravuri_filter(grid[variable], s_rr, q_min, m)
    else:
        i_grid = filter(grid[variable], s_rr, q_min, m, c)
    grid_size = gridshape[0]*gridshape[1] # shape = [4, x, y]
    return np.sum(i_grid) / grid_size

def sample_from_instance(save_dir, p_importance, row, args):
    """For each instance, sample and writes data in the csv file

    Args:
        save_dir (str): the save directory
        p_importance (float): the importance computed
        row (_type_): A row from a dataframe
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 5 and not args.progress_bar: print(f"Sampling...")
    for instance in range(1, args.n_instances + 1):
        save_dir_instance = f"{save_dir}INST{instance}/"
        p_uniform = rd.uniform(0, 1)
        if p_uniform <= p_importance:
            with open(f"{save_dir_instance}labels.csv", "a", encoding="utf8") as file:
                file.write(f"{row['Name']},{row['Date']},{row['Leadtime']},{row['Member']},{row['Gigafile']},{row['Localindex']},{p_importance}\n")

def importance_sampling(parameters, dirs, gridshape, variable, args):
    """Compute importance sampling with parameters parameters

    Args:
        parameters (tuple[float]): Parameters of importance sampling (s_rr, q_min, m, c)
        dirs (str): Directories with which data interact (csv_dir, data_dir, save_dir)
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 1: print(f"Importance sampling...")
    csv_dir, data_dir, save_dir = dirs
    create_dirs(save_dir, args)
    dataframe = pd.read_csv(f"{csv_dir}labels.csv")
    s_gigafile = {gigafile for gigafile in os.scandir(data_dir) if gigafile.name != "labels.csv" and "~" not in gigafile.name}
    n_gigafile = len(s_gigafile)
    start_time = perf_counter()
    for idx_gigafile, gigafile in enumerate(s_gigafile):
        if args.verbose >= 2:
            print(f"Loading patch {gigafile.name} ({idx_gigafile + 1}/{n_gigafile})...")
            if (idx_gigafile + 1) % ((n_gigafile // args.refresh) + 1) == 0:
                print_progress(idx_gigafile, n_gigafile, start_time)
        l_grid = np.load(f"{gigafile.path}")
        dataframe_gigafile = dataframe.groupby("Gigafile").get_group(int(gigafile.name[:-4]))
        for idx_grid, grid in enumerate(l_grid):
            if args.progress_bar: print_progress_bar(idx_grid, len(l_grid))
            p_importance = importance(grid, parameters, gridshape, variable, args)
            sample_from_instance(save_dir, p_importance, dataframe_gigafile.iloc[idx_grid], args)
        del l_grid
    if args.verbose >= 2: print(f"All gigafiles processed.")
    if args.verbose >= 1: print(f"Importance sampling for parameters {parameters} DONE.")


if __name__ == "__main__":
    try:
        compute_c(5, 0.001, 500)
    except RuntimeError as error:
        print(f"{repr(error)}")