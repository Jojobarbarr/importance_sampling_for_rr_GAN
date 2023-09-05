import json
import os
from argparse import ArgumentParser
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from called.utile import make_save_dir, parse_float


def compare(data_dir, args):
    dirs_list = [directory for directory in os.scandir(data_dir) if "~" not in directory.name]
    n_dirs = len(dirs_list)
    riddle_list = [(dir1, dir2) for idx_dir1, dir1 in enumerate(dirs_list) for dir2 in dirs_list[idx_dir1:] if dir1 != dir2]
    v_couple_list = [(0, 20), (1, 15), (4, 20), (20, 70), (40, 60)]
    for v_couple in v_couple_list:
        for couple in riddle_list:
            plt.clf()
            for directory in couple:
                with open(f"{directory.path}/~stats/extracted_values.json") as countfile:
                    counter = Counter(json.load(countfile))
                save_dir_img = f"{data_dir}~pictures/"
                end_filename = f""
                if len(counter):
                    keys = np.array(sorted(list(map(float, counter.keys()))))
                    values = np.array([counter[str(key)] for key in keys])
                    v_min, v_max = v_couple
                    v_min_transformed, v_max_transformed = v_min, v_max
                    # for _ in range(args.log_transform):
                    #     save_dir_img += "log/"
                    #     keys = np.log(1 + keys)
                    #     end_filename += f"_log"
                    #     v_min_transformed, v_max_transformed = np.log(1 + v_min_transformed), np.log(1 + v_max_transformed)
                    sns.histplot(x=keys, weights=values, stat="density", bins=int(v_max-v_min) * 2, binrange=(v_min_transformed, v_max_transformed), label=directory.name, alpha=0.6)
            plt.xlabel("rr")
            plt.legend()
            make_save_dir(save_dir_img, args)
            plt.savefig(f"{save_dir_img}extracted_values_{couple[0].name}_{couple[1].name}_{v_min}_{v_max}_{end_filename}.png")
    




def mix(working_dir, threshold, args):
    """Riddle over parameters and plot and save histogram

    Args:
        working_dir (str): directory of interest
        threshold (float): the value under which, values aren't considered
        args (argparse.Namespace): args of the program
    """
    save_dir_spe = working_dir + "mix/" + str(threshold) + "/"
    if args.log_transform:
        save_dir_spe += "transformed/"
    if args.double_log_transform:
        save_dir_spe += "double_transformed/"
    if args.std:
        if args.med:
            save_dir_spe += "median/"
        else:
            save_dir_spe += "std/"
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
            if args.log_transform:
                arr = np.log(1 + arr)
                final_filename += "transformed_"
            elif args.double_log_transform:
                arr = np.log(1+np.log(1 + arr))
                final_filename += "double-transformed_"
            if args.std:
                mean, maxi, med = compute_mean_max_med(arr, args)
                if args.med:
                    arr = (arr - med) / maxi * 0.95
                    final_filename += "MEDIAN_"
                else:
                    arr = (arr - mean) / maxi * 0.95
                final_filename += "STD_"
            plt.hist(arr, bins=args.bins, alpha=0.5, density=True, label="param: " + param_set_entry.name, range=args.range)
            final_filename += param_set_entry.name + "_x_" * (count == 0)
        plt.legend()
        make_save_dir(save_dir_spe, args)
        range_str = "_".join([str(limit) for limit in args.range])
        final_filename += "_range_" + range_str + ".png"
        plt.savefig(save_dir_spe + final_filename)
        if args.verbose >= 2: print("Saved histogram", final_filename)

def compute_mean_max_med(arr, args):
    maxi = arr.max()
    mean = arr.mean()
    med = None
    if args.med:
        med = sorted(arr)[len(arr)//2]
    maxi = maxi - mean
    return mean, maxi, med

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity. Ex: -v (verbosity normal), -vvv (verbosity max)")
    parser.add_argument("-r", "--range", type=float, nargs="*", help="Range of the x axis of the histogram. If not specified, take the values [0, 30]. Ex: -r 10 30 (for a range [10, 30])")
    parser.add_argument("-t", "--threshold", type=float, default=0.1, help="Threshold for the statistics.")
    parser.add_argument("-b", "--bins", type=int, default=100, help="Number of bins for the histogram")
    group_log_transformation = parser.add_mutually_exclusive_group()
    group_log_transformation.add_argument("-l", "--log_transform", action="count", default=0, help="If used, transform data to log(1+data) times l is written")
    parser.add_argument("-s", "--std", action="store_true", help="If used, transform data to ((data - mean) / max) * 0.95")
    parser.add_argument("-m", "--med", action="store_true", help="If used, transform data to ((data - med) / max) * 0.95")
    parser.add_argument("-p", "--path", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/", help="Main path of work")
    parser.add_argument("data_dir", type=str, help="Directory of interest. Ex: 27-07-16h30/ (don't forget the '/' at the end)")

    args = parser.parse_args()
    if args.range is None:
        args.range = [0, 30]

        # self.means = list(tuple(Means))
        # self.stds = list(tuple((1.0/0.95)*(Maxs)))
    #### GLOBAL ####
    THRESHOLD = args.threshold

    #### PATH ####
    MAIN_PATH = args.path + args.data_dir
    
    # mix(MAIN_PATH, THRESHOLD, args)
    compare(MAIN_PATH, args)
