import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from called.utile import make_save_dir, parse_float


def mix(working_dir, threshold, args):
    """Riddle over parameters and plot and save histogram

    Args:
        working_dir (str): directory of interest
        threshold (float): the value under which, values aren't considered
        args (argparse.Namespace): args of the program
    """
    l_entries = []
    for entry_param in os.scandir(working_dir):
        if entry_param.name != "mix":
            l_entries.append(entry_param)
    l_entries_crib_2_by_2 = [(j, k) for idx, j in enumerate(l_entries) for k in l_entries[idx:] if j!= k]
    for param_set_entries in l_entries_crib_2_by_2:
        final_filename = "hist_mix_"
        plt.clf()
        for count, param_set_entry in enumerate(param_set_entries):
            filename = param_set_entry.path + "/mix/threshold_" + parse_float(str(threshold)) + "/hist_" + param_set_entry.name + ".npy"
            arr = np.load(filename, allow_pickle=True)
            plt.hist(arr, bins=args.bins, alpha=0.5, density=True, label="param: " + param_set_entry.name, range=args.range)
            final_filename = final_filename + param_set_entry.name + "_x_" * (count == 0)
        plt.legend()
        save_dir_spe = working_dir + "mix/"
        make_save_dir(save_dir_spe, args)
        range_str = "_".join([str(limit) for limit in args.range])
        final_filename += "_range_" + range_str + ".png"
        plt.savefig(save_dir_spe + final_filename)
        if args.verbose >= 2: print("Saved histogram", final_filename)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity. Ex: -v (verbosity normal), -vvv (verbosity max)")
    parser.add_argument("-r", "--range", type=int, nargs="*", help="Range of the x axis of the histogram. If not specified, take the values [0, 30]. Ex: -r 10 30 (for a range [10, 30])")
    parser.add_argument("-t", "--threshold", type=float, default=0.1, help="Threshold for the statistics.")
    parser.add_argument("-b", "--bins", type=int, default=50, help="Number of bins for the histogram")
    parser.add_argument("data_dir", type=str, help="Directory of interest. Ex: 27-07-16h30/ (don't forget the '/' at the end)")

    args = parser.parse_args()
    
    if args.range is None:
        args.range = [0, 30]

    #### GLOBAL ####
    THRESHOLD = args.threshold

    #### PATH ####
    MAIN_PATH = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/analysis/" + args.data_dir
    
    mix(MAIN_PATH, THRESHOLD, args)