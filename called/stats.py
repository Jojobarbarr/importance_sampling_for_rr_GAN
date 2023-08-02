import fileinput
import os
from argparse import ArgumentParser
from time import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from called.utile import (make_save_dir, parse_float, print_progress,
                          print_progress_bar)
from matplotlib import pyplot as plt


def run_stat(variable, gridshape, dirs, threshold, args):
    """main function of this file, run the stats about variable on data from dirs

    Args:
        variable (int): index of the channels corresponding to the variable
        gridshape (tuple[int]): shape of the grid
        dirs (str): Directories with which data interact (csv_dir, csv_dir_is, data_dir, save_dir)
        threshold (float): the number we want the values to be above
        args (argparse.Namespace): args of the program
    """
    csv_dir, csv_dir_is, data_dir, save_dir = dirs
    if args.verbose >= 1: print("Running stats...")
    i_dataframe_is, i_global_rate, i_files_rate, i_rate_calc, i_mean_calc, i_variance_calc, i_values = handle_patch(variable, gridshape, csv_dir, csv_dir_is, data_dir, threshold, args)

    if args.verbose >= 1:
        print("\n")
        print("Saving results...")
    for instance in range(args.n_instances - args.n_instances_pre):
        if args.verbose >= 3: print("\n---Instance", args.n_instances_pre + instance + 1, " saving---")
        save_dir_instance = save_dir + "INST" + str(args.n_instances_pre + instance + 1) + "/threshold_" + parse_float(str(threshold)) + "/"
        make_save_dir(save_dir_instance, args)
        hist_from_arr(i_values[instance], save_dir_instance, args)
        save_stats(i_rate_calc[instance], save_dir_instance, "rate_grid", args)
        save_stats(i_mean_calc[instance], save_dir_instance, "mean_grid", args)
        save_stats(i_variance_calc[instance], save_dir_instance, "variance_grid", args)
        with open(save_dir_instance + "log.txt", "a", encoding="utf8") as file:
            file.write("Number of files: "+str(len(i_dataframe_is[instance])))
            file.write("\nThreshold: "+str(threshold))
            file.write("\nGlobal rate: "+str(i_global_rate[instance]))
    if args.verbose >= 1: print("DONE run stats.")

def handle_patch(variable, gridshape, csv_dir, csv_dir_is, data_dir, threshold, args):
    """Load data gigafile by gigafile (patch) and for each instance select the data sampled from the patch and compute the metrics

    Args:
        variable (int): index of the channels corresponding to the variable
        gridshape (tuple[int]): shape of the grid
        csv_dir (str): directory from where the global data information is read
        csv_dir_is (str): directory from where the data sampled to be evaluate is read
        data_dir (str): directory from where the data is loaded
        threshold (float): the number we want the values to be above
        args (argparse.Namespace): args of the program

    Returns:
        tuple[list[numpy.array]]: the different stats
    """
    dataframe = pd.read_csv(csv_dir + "labels.csv")
    n_patch = int(dataframe.iloc[-1]["Gigafile"])

    i_dataframe_is = [pd.read_csv(csv_dir_is + "INST" + str(instance) + "/" + "labels.csv") for instance in range(args.n_instances_pre + 1, args.n_instances + 1)]

    i_n_grid = [len(dataframe_is) for dataframe_is in i_dataframe_is]
    i_global_r = [0 for _ in range(args.n_instances_pre, args.n_instances)]
    i_files_r = [[] for _ in range(args.n_instances_pre, args.n_instances)]
    i_rate_map = [np.zeros([gridshape[0], gridshape[1]]) for _ in range(args.n_instances_pre, args.n_instances)]
    i_mean_map = [np.zeros([gridshape[0], gridshape[1]]) for _ in range(args.n_instances_pre, args.n_instances)]
    i_variance_map = [np.zeros([gridshape[0], gridshape[1]]) for _ in range(args.n_instances_pre, args.n_instances)]
    i_values_patch = [[] for _ in range(args.n_instances_pre, args.n_instances)]
    
    i_gigafile_group = [dataframe_is.groupby("Gigafile") for dataframe_is in i_dataframe_is]
    patch_stats(n_patch, variable, threshold, data_dir, i_gigafile_group, i_global_r, i_files_r, i_rate_map, i_mean_map, i_variance_map, i_values_patch, args)
    for instance in range(args.n_instances - args.n_instances_pre):
        if i_n_grid[instance] == 0:
            print("Instance", instance, "no value.")
        else:
            i_global_r[instance] /= i_n_grid[instance]*gridshape[0]*gridshape[1]
            i_rate_map[instance] /= i_n_grid[instance]
            i_mean_map[instance] /= i_n_grid[instance]
            i_variance_map[instance] /= i_n_grid[instance]
            i_variance_map[instance] -= i_mean_map[instance]**2
    return i_dataframe_is, i_global_r, i_files_r, i_rate_map, i_mean_map, i_variance_map, i_values_patch

def patch_stats(n_patch, variable, threshold, data_dir, i_gigafile_group, i_global_r, i_files_r, i_rate_map, i_mean_map, i_variance_map, i_values_patch, args):
    """Perform stat on the patch and add them to the previous computed stats

    Args:
        n_patch (int): number of patch in total
        data_dir (str): directory from which data is loaded
        i_gigafile_group (list): Instance list of pandas dataframe groups by gigafiles
        i_global_r (list): Instance list for global rate
        i_files_r (list): Instance list for files rate
        i_rate_map (list): Instance list for rate map
        i_mean_map (list): Instance list for mean map
        i_variance_map (list): Instance list for variance map
        i_values_patch (list): Instance list for values extracted
        args (argparse.Namespace): args of the program
    """
    start_time = time()
    for gigafile_index in range(1, n_patch + 1):
        if args.verbose >= 1: print("\nLoading patch", gigafile_index, "/", n_patch, "...")
        l_grid = np.load(data_dir + str(gigafile_index) + ".npy")
        if args.verbose >= 1: print("Done.\n")
        if args.verbose >= 2 and (gigafile_index + 1) % (n_patch // args.refresh) == 0:
            print_progress(gigafile_index, n_patch, start_time)
        i_data = [[] for _ in range(args.n_instances_pre, args.n_instances)]
        for instance in range(args.n_instances - args.n_instances_pre):
            if args.verbose >= 3: print("\n---Instance", args.n_instances_pre + instance + 1, "---")
            if gigafile_index in i_gigafile_group[instance].groups:
                instance_dataframe = i_gigafile_group[instance].get_group(gigafile_index)
                for row in instance_dataframe.itertuples():
                    i_data[instance].append(l_grid[row.Localindex])
                interm_global_sum, interm_files_rate = rate_gt(variable, i_data[instance], threshold, args)
                i_global_r[instance] += interm_global_sum
                i_files_r[instance] += interm_files_rate
                i_rate_map[instance] = np.add(i_rate_map[instance], pixel_rate(variable, i_data[instance], threshold, args))
                i_mean_map[instance] = np.add(i_mean_map[instance], mean_samples(variable, i_data[instance], args))
                i_variance_map[instance] = np.add(i_variance_map[instance], variance_samples(variable, i_data[instance], args))
                i_values_patch[instance] += list(extract_values_greater_than(variable, i_data[instance], threshold, args))
        del l_grid

def rate_gt(variable, data, x_min, args):
    """Compute the sum of gridpoint with a value greater than x over the total number of
    gridpoint in the grid for each file and globally

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        x_min (float): the number we want the values to be above
        args (argparse.Namespace): args of the program

    Returns:
        tuple[float, list[float]]: return the global sum and a list of rate for each file
    """
    if args.verbose >= 3: print("\nComputing rate_gt...")
    grid = data[0]
    grid_size = grid.shape[1] * grid.shape[2]
    n_grid = len(data)
    total_count = 0
    file_results = []
    for index, grid in enumerate(data):
        if args.verbose >= 3: print_progress_bar(index, n_grid)
        local_count = np.count_nonzero(grid[variable] >= x_min)
        file_results.append(local_count / grid_size)
        total_count += local_count
    return total_count, file_results


def pixel_rate(variable, data, x_min, args):
    """Compute the sum of gridpoint with a value greater than x over the total number of
    gridpoint at the same position in the dataset

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        x_min (float): the number we want the values to be above
        args (argparse.Namespace): args of the program

    Returns:
        numpy.array: grid where each gridpoint has the sum of each time the value is greater than x
    """
    if args.verbose >= 3: print("\nComputing pixel_rate...")
    grid = data[0]
    n_grid = len(data)
    rate_map = np.zeros([grid.shape[1], grid.shape[2]])
    for index, grid in enumerate(data):
        if args.verbose >= 3: print_progress_bar(index, n_grid)
        rate_map += 1 * (grid[variable] >= x_min)
    return rate_map

def mean_samples(variable, data, args):
    """Compute the sum for each gridpoint.

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        args (argparse.Namespace): args of the program
    
    Returns:
        numpy.array: the grid resulting from the sum of every grid
    """
    if args.verbose >= 3: print("\nComputing mean_samples...")
    grid = data[0]
    n_grid = len(data)
    temp_mean_grid = np.zeros([grid.shape[1], grid.shape[2]])
    for index, grid in enumerate(data):
        if args.verbose >= 3: print_progress_bar(index, n_grid)
        temp_mean_grid += grid[variable]
    return temp_mean_grid


def variance_samples(variable, data, args):
    """Compute the sum of the values squared for each gridpoint

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        args (argparse.Namespace): args of the program

    Returns:
        numpy.array: the grid resulting from the sum of the values squared of every grid
    """
    if args.verbose >= 3: print("\nComputing variance_samples...")
    grid = data[0]
    n_grid = len(data)
    variance_grid = np.zeros([grid.shape[1], grid.shape[2]])
    for index, grid in enumerate(data):
        if args.verbose >= 3: print_progress_bar(index, n_grid)
        variance_grid += grid[variable]**2
    return variance_grid


def extract_values_greater_than(variable, data, threshold, args):
    """Extract from each grid all the values greater than threshold and store them in a list

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        threshold (float): the number we want the values to be above
        args (argparse.Namespace): args of the program

    Returns:
        list[float]: store every value greater than the threshold
    """
    if args.verbose >= 3: print("\nExtracting values greater than ", threshold, "...", sep="")
    n_grid = len(data)
    for index, grid in enumerate(data):
        if args.verbose >= 3: print_progress_bar(index, n_grid)
        mask = grid[variable] > threshold
        extracted_values = grid[variable][mask]
    return extracted_values

def hist_from_arr(arr, save_dir, args):
    """Create the histogram of arr.

    Args:
        arr (list): list of values
        save_dir (str): directory where the histogram is saved
        bins (int): the number of bins in the histogram
        args (argparse.Namespace): args of the program
    """
    plt.clf()
    np.save(save_dir + "hist.npy", arr, allow_pickle=True)
    plt.hist(arr, bins=50, density=True)
    plt.savefig(save_dir + "hist.png")
    if args.verbose >= 2: print("saved histogram.")
    


def save_stats(grid, save_dir, filename, args):
    """Save the figures

    Args:
        grid (numpy.array): grid of values
        save_dir (str): directory where the histogram is saved
        filename (str): name of the file to output
        args (argparse.Namespace): args of the program
    """
    fig, axes = plt.subplots()
    img = axes.imshow(grid, origin="lower")
    fig.colorbar(img)
    fig.savefig(save_dir + filename + ".png")
    np.save(save_dir + filename + ".npy", grid, allow_pickle=True)
    plt.close(fig)
    if args.verbose >= 2: print("saved ", filename, ".", sep="")

def save_mix(param_string, threshold, dirs, args):
    """Combine the instances stats to provide a mean of every stat over a same set of parameter (param_string)

    Args:
        param_string (str): the parameters over whiches are computed the stats
        threshold (float): the value under which, values aren't considered
        dirs (tuple[str]): the directories of work : data_dir and save_dir
        args (argparse.Namespace): args of the program
    """
    data_dir, save_dir = dirs
    histname = "hist"
    rate_name = "rate_grid"
    mean_name = "mean_grid"
    variance_name = "variance_grid"
    l_names = [rate_name, mean_name, variance_name]
    l_arr_glob, arr_glob, l_data_log_glob = load_into_global_arrays(data_dir, threshold, histname, l_names, args)
    for idx in range(len(l_data_log_glob)):
        l_data_log_glob[idx] /= args.n_instances
    with open(save_dir + "log_" + param_string + ".txt", "a", encoding="utf8") as logfile:
        logfile.write("Mean number of files: " + str(l_data_log_glob[0]))
        logfile.write("\nThreshold: " + str(threshold))
        logfile.write("\nMean global rate: " + str(l_data_log_glob[1]))
    histname += "_" + param_string
    save_global(arr_glob, l_arr_glob, save_dir, histname, param_string, args)

def load_into_global_arrays(data_dir, threshold, histname, args):
    """Load by iterating over the instances the stats in a global array

    Args:
        data_dir (str): directory where the stats are stored
        threshold (float): the value under which, values aren't considered
        histname (str): names of the histogram files
        args (argparse.Namespace): args of the program
    """
    rate_name = "rate_grid"
    mean_name = "mean_grid"
    variance_name = "variance_grid"
    l_names = [rate_name, mean_name, variance_name]
    l_arr_glob = [[], [], []]
    arr_glob = np.empty(0)
    l_data_log_glob = [0, 0]
    for instance in range(1, args.n_instances + 1):
        filename_hist = data_dir + "INST" + str(instance) + "/threshold_" + threshold + "/" + histname + ".npy"
        arr = np.load(filename_hist, allow_pickle=True)
        arr_glob = np.concatenate((arr_glob, arr), axis=None)
        filename_log = data_dir + "INST" + str(instance) + "/threshold_" + threshold + "/" + "log.txt"
        with open(filename_log, 'r', encoding='utf8') as logfile:
            data_log = []
            for line in logfile:
                data_log.append(line)
            l_data_log_glob[0] += int(data_log[0][17:-1])
            l_data_log_glob[1] += float(data_log[2][13:])
        for name in l_names:
            filename = data_dir + "INST" + str(instance) + "/threshold_" + threshold + "/" + name + ".npy"
            arr = np.load(filename, allow_pickle=True)
            l_arr_glob[l_names.index(name)].append(arr)
    return l_arr_glob, arr_glob, l_data_log_glob

def save_global(arr_glob, l_arr_glob, save_dir, histname, param_string, args):
    """Plot and save the global stats

    Args:
        arr_glob (list[float]): the list of values greater than threshold to plot the histogramm
        l_arr_glob (list): list of the grid containing global grid stats (mean_map, variance_map, ...)
        save_dir (str): Where plot and arrays will be saved
        histname (str): name of the histogram files
        param_string (str): the parameters over whiches are computed the stats
        args (argparse.Namespace): args of the program
    """
    plt.clf()
    plt.hist(arr_glob, bins=50, density=True)
    plt.savefig(save_dir + histname + ".png")
    np.save(save_dir + histname + ".npy", arr_glob, allow_pickle=True)
    if args.verbose >= 2: print("saved numpy", histname)
    for idx, name in enumerate(l_names):
        plt.clf()
        grid = l_arr_glob[idx]
        grid = np.array(grid).mean(axis = 0)
        filename = name + "_" + param_string
        fig, axes = plt.subplots()
        img = axes.imshow(grid, origin="lower")
        fig.colorbar(img)
        fig.savefig(save_dir + filename + ".png")
        np.save(save_dir + filename + ".npy", grid, allow_pickle=True)
        plt.close(fig)
        if args.verbose >= 2: print("saved ", filename, ".", sep="")


def stat_on_source(data_dir, save_dir, args):
    """Compute stats on the unsampled data

    Args:
        data_dir (str): the unsampled data directory
        save_dir (str): the directory where stats are saved
        args (argparse.Namespace): args of the program
    """
    dataframe = pd.read_csv(data_dir + "labels.csv")
    n_grid = len(dataframe)
    n_patch = dataframe.iloc[-1].Gigafile
    current_patch = 0
    start_time = time()
    val = []
    for index, row in dataframe.iterrows():
        if args.verbose >= 2 and (index + 1) % (n_grid // args.refresh) == 0:
            print_progress(index, n_grid, start_time)
        if row["Gigafile"] != current_patch:
            current_patch += 1
            if args.verbose >= 1: print("\nLoading patch", current_patch, "/", n_patch, "...")
            l_grid = np.load(data_dir + str(current_patch) + ".npy")
            if args.verbose >= 1: print("Done.\n")
            val += list(extract_values_greater_than(0, l_grid, 0, args))
    hist_from_arr(val, save_dir, args)





if __name__ == "__main__":
    ## ARGPARSE ##
    parser = ArgumentParser()
    parser.add_argument("--refresh", type=int, default=10)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    #### STATS ON SOURCE ####
    # ## PATH ##
    # DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/crop_processed_giga/"
    # SAVE_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/analysis/source/"
    # make_save_dir(SAVE_DIR, args)

    # stat_on_source(DATA_DIR, SAVE_DIR, args)

    #### VERIFY SPLIT ####
    # ## PATH ##
    # RAW_DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/brochetc/float32_t2m_u_v_rr/"
    # DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/pre_proc_31-07-10h/cropped_giga/"
    # SAVE_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/pre_proc_31-07-10h/draft/"
    
    # GRID_NUM = 0
    # VMAX = 0.02
    # make_save_dir(SAVE_DIR, args)
    # l_grid = np.load(DATA_DIR + str(1) + ".npy", allow_pickle=True)
    # fig, axes = plt.subplots()
    # img = axes.imshow(l_grid[GRID_NUM][0], origin="lower", vmin=0, vmax=VMAX)
    # fig.colorbar(img)
    # fig.savefig(SAVE_DIR + "rr.png")
    # dataframe = pd.read_csv(DATA_DIR + "labels.csv")
    # row = dataframe.iloc[GRID_NUM]
    # orig_grid = np.load(RAW_DATA_DIR + row["Date"] + "_rrlt1-24.npy", allow_pickle=True)
    # fig, axes = plt.subplots()
    # img = axes.imshow(orig_grid[:, :, int(row["Leadtime"])-1, int(row["Member"])-1], origin="lower", vmin=0, vmax=VMAX)
    # fig.colorbar(img)
    # fig.savefig(SAVE_DIR + "rr_orig.png")
    


