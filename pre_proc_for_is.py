import glob
import os
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
from called.utile import make_save_dir, print_progress, print_progress_bar


def rename_rr_files(raw_data_dir):
    """rr files needs to be rename to be in the same format as u, v and t2m files.

    Args:
        raw_data_dir (str): Directory of the raw data
    """
    filenames = glob.glob(raw_data_dir + "*zeroed*.npy")
    for filename in filenames:
        os.rename(filename, raw_data_dir + filename[len(raw_data_dir)+7:len(raw_data_dir)+27] +
                  "_rrlt1-24.npy")


def process_all(raw_data_dir, save_dir, cropping, args):
    """transform the datafiles in a way more practical with one file for each day / lead time / member and with all 4 variables being the channels in the order (rr, u, v, t2mS)

    Args:
        raw_data_dir (str): Directory of the raw data
        save_dir (str): Directory where the data is saved
        cropping (bool): Whether or not cropping is done along with splitting
        args (argparse.Namespace): args of the program
    """
    save_dir_splitted = save_dir + "splitted/"
    make_save_dir(save_dir_splitted, args)
    with open(save_dir_splitted + "labels.csv", "w", encoding="utf8") as file:
        file.write("Name,Date,Leadtime,Member\n")
        file.close()

    filenames = glob.glob(raw_data_dir + "*lt1*.npy")
    n_days = len(filenames)//4 # 4 files for each date (rr, u, v, t2m)
    done_days = []
    count = 0
    start_time = time()
    for index, filename in enumerate(filenames):
        l_grid = []
        day_name = filename[len(raw_data_dir):len(raw_data_dir)+20]
        if day_name not in done_days:
            count += 1
            l_grid = process_day(day_name, count, save_dir_splitted, args)
            done_days.append(day_name)
            if cropping:
                crop(l_grid, save_dir, args)
            if args.verbose >= 1: 
                print("Remains", n_days-index, "days.")
                if args.verbose >= 2:
                    print("\n")
                    print_progress(index, n_days, start_time)


def process_day(day_name, day_number, save_dir, args):
    """Gather the files concerning a same say

    Args:
        day_name (str): Name of the day
        day_number (int): This day is the day_numberth day to be processed
        save_dir (str): Directory where the data is saved
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 1: print("Day:", day_name)

    list_mat = merge_variables(day_name, raw_data_dir, args) # shape = [4, 717, 1121, 24, 16]
    list_mat = split_by_lead_time_and_member(list_mat, args) # shape = [384, 4, 717, 1121]

    n_files = len(list_mat) # for each day, we have 384 files
    sample_num = (day_number-1)*n_files # to have a unique name for each file

    l_grid = []
    for idx, grid in enumerate(list_mat): # each grid is shape = [4, 717, 1121]
        leadtime = idx // 16 + 1 # cf split_by_lead_time_and_member
        member = idx % 16 + 1
        sample_name = "_sample"+str(sample_num+idx) # unique name
        np.save(save_dir + sample_name + ".npy", grid, allow_pickle=True)
        with open(save_dir + "/labels.csv", "a", encoding="utf8") as file:
            file.write(sample_name+","+str(day_name)+"," + str(leadtime)+","+str(member)+"\n")
            file.close()
        l_grid.append((grid, sample_name)) # shape = [384, 2] (number of files, grid + sample_name)
        if args.verbose >= 3: print_progress_bar(idx, n_files)
    print("\n")
    return l_grid


def merge_variables(day_name, raw_data_dir, args):
    """Open the 4 files of day day_name (rr, u, v and t2m) and add them to list_mat

    Args:
        day_name (str): Name of the day
        raw_data_dir (str): Directory of the raw data
        args (argparse.Namespace): args of the program

    Returns:
        list[numpy.array]: a list of dimension [4, 660, 882, 24, 16], (n_variables, x, y, n_lead_time, n_members)
    """
    if args.verbose >= 2: print("Merging " + day_name + "...")
    list_mat = []
    for var in VAR_NAMES:
        filename = raw_data_dir + day_name + "_" + var + "lt1-24.npy"
        if args.verbose >= 3: print("opening", filename)
        mat = np.load(filename, allow_pickle=True)
        list_mat.append(mat)
    return list_mat

def split_by_lead_time_and_member(mat, args):
    """split the matrice mat by lead_time and members

    Args:
        mat (list[numpy.array]): a list of dimension [4, 717, 1121, 24, 16], (n_variables, x, y, n_lead_time, n_members)
        args (argparse.Namespace): args of the program

    Returns:
        list[numpy.array]: a list of dimension [384, 4, 717, 1121], (n_lead_time * n_members, n_variables, x, y)
    """
    if args.verbose >= 2: print("Splitting...")
    mat = np.array(mat)
    n_leadtime, n_members = mat.shape[3], mat.shape[4]
    data = [mat[:, :, :, i, j] for i in range(n_leadtime) for j in range(n_members)]
    return data

def crop(l_grid, save_dir, args):
    """Crop the grids according to INDEXES

    Args:
        l_grid (list[tuple[numpy.array, str]]): a list of shape [384, 2] with the tuple containing grids of shape [4, 717, 1121] and the sample name
        save_dir (str): Directory where the data is saved
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 2: print("Cropping...")
    save_dir_cropped = save_dir + "cropped/"
    make_save_dir(save_dir_cropped, args)

    n_grid  = len(l_grid) # should be 384
    for index, (grid, sample_name) in enumerate(l_grid):
        lb_index, rb_index, lu_index, ru_index = INDEXES
        grid = grid[:, lb_index:rb_index, lu_index:ru_index] # grid shape = [4, 717, 1121] (n_variables, x, y)
        np.save(save_dir_cropped + sample_name, grid, allow_pickle=True)
        if args.verbose >= 3: print_progress_bar(index, n_grid)


def merge_into_gigafiles(working_dir, datatype, args):
    """Merge the numerous little files in gigafiles to load patches at once and accelerate data computation like importance sampling or cropping

    Args:
        working_dir (str): Working directory
        datatype (str): "splitted" or "cropped"
        args (argparse.Namespace): args of the program
    """
    s_datatype = {"splitted", "cropped"}
    d_datatype_max_file_loaded = {"splitted": 1000, "cropped": 8000}
    if datatype not in s_datatype:
        raise ValueError("Datatype must be in " + str(s_datatype) + " you gave '" + datatype + "'.")
    dataframe = pd.read_csv(working_dir + "splitted/labels.csv")
    data_dir = working_dir + datatype
    save_dir = data_dir + "_giga/"
    data_dir += "/"
    make_save_dir(save_dir, args)
    with open(save_dir + "labels.csv", "w", encoding="utf8") as file:
        file.write("Name,Date,Leadtime,Member,Gigafile,Localindex\n")
        file.close()
    handle_patch(dataframe, data_dir, save_dir, d_datatype_max_file_loaded[datatype], args)

def handle_patch(dataframe, data_dir, save_dir, max_files_loaded, args):
    """Handle patch processing.

    Args:
        dataframe (pandas.DataFrame): Dataframe where are saved the name of the files.
        data_dir (str): Data directory
        save_dir (str): Directory where the data is saved
        max_files_loaded (int): Maximum of files loaded in a patch (to prevent overuse of RAM)
        args (argparse.Namespace): args of the program
    """
    n_files = len(dataframe)
    begin = 0
    n_patch = n_files // max_files_loaded + 1
    end = min(max_files_loaded, n_files)
    files_processed = 0
    patch_count = 0
    while files_processed < n_files:
        patch_count += 1
        if args.verbose >= 2: print("Patch", patch_count, "/", n_patch)
        files_processed += end - begin + 1
        giga = load(dataframe, data_dir, save_dir, begin, end, patch_count, args)
        np.save(save_dir + str(patch_count) + ".npy", giga, allow_pickle=True)
        del giga
        begin = end + 1
        end = min(begin + max_files_loaded, begin + n_files-files_processed)

def load(dataframe, data_dir, save_dir, beg, end, patch_count, args):
    """Load patch of files from their name in the dataframe and store the data in a list.

    Args:
        dataframe (pandas.DataFrame): the dataframe from which we can find the name of the datafiles
        data_dir (str): Data directory
        save_dir (str): Directory where the data is saved
        beg (int): from which index of the dataframe we want to load the datafiles...
        end (int): ... to where we want to stop
        patch_count (int): the patch number, allowing to save the gigafile number where is saved the numpy array
        args (argparse.Namespace): args of the program

    Returns:
        list: list of numpy arrays representing maps with 4 channels (rr, u, v, t2m)
    """
    n_tot = end - beg
    if args.verbose >= 2: print("Loading data,", n_tot, "files...")
    data = []
    start_time = time()
    for index in range(beg, end):
        row = dataframe.iloc[index]
        if args.verbose >= 3 and (index + 1) % (n_tot // args.refresh) == 0:
            print_progress(index-beg, n_tot, start_time)
        data.append(np.load(data_dir + row["Name"] + ".npy", allow_pickle=True))
        with open(save_dir + "labels.csv", "a", encoding="utf8") as file:
            file.write(row["Name"] + "," + row["Date"] + "," + str(row["Leadtime"]) + "," + str(row["Member"]) + "," + str(patch_count) + "," + str(index-beg) + "\n")
            file.close()
    return data


def crop_from_gigafiles(data_path_giga, save_dir,  args):
    """Crops from gigafile splitted.

    Args:
        data_path_giga (str): Data path to the directory of the gigafiles
        save_dir (str): Directory where data is saved
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 1: print("Cropping from gigafiles...")
    dataframe = pd.read_csv(data_path_giga + "labels.csv")
    data_group_gigafile = dataframe.groupby("Gigafile")
    n_gigafiles = sum(1 for entry in os.scandir(data_path_giga)) - 1
    start_time = time()
    for index, entry in enumerate(os.scandir(data_path_giga)):
        if args.verbose >= 2: 
            print("Processing Gigafile", index + 1, "of", n_gigafiles)
            if args.verbose >= 3 and (index + 1) % (n_gigafiles // args.refresh) == 0:
                print_progress(index, n_gigafiles, start_time)
        if entry.name != "labels.csv":
            l_grid = np.load(entry.path, allow_pickle=True)
            l_grid = [(l_grid[row.Localindex], row.Name) for row in data_group_gigafile.get_group(int(entry.name[:-4])).itertuples()]
            crop(l_grid, save_dir, args)
            print("\n")
    merge_into_gigafiles(save_dir, "cropped", args)

# def split(dataframe, DATA_DIR, SAVE_DIR):
#     n_patch = int(dataframe.iloc[-1]["Gigafile"])
#     start_time = time()
#     for gigafile_index in range(1, n_patch + 1):
#         print("\nLoading patch", gigafile_index, "/", n_patch, "...")
#         data = np.load(DATA_DIR + str(gigafile_index) + ".npy", allow_pickle=True)
#         print("Done.\n")
#         if (gigafile_index + 1) % (n_patch // 10) == 0:
#             print_progress(gigafile_index, n_patch, start_time)
#         for row in dataframe.itertuples():
#             if row.Gigafile == gigafile_index:
#                 np.save(SAVE_DIR + row.Name + ".npy", data[row.Localindex], allow_pickle=True)


if __name__ == "__main__":
    #### ARGPARSE ####
    parser = ArgumentParser()

    parser.add_argument("save_directory", type=str, default=None, help="Directory where data will be saved: 'pre_proc_' + save_directory")
    parser.add_argument("-l", "--load_directory", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/brochetc/float32_t2m_u_v_rr/", help="Data directory from which data is loaded")
    parser.add_argument("-c", "--crop_index", type=int, nargs="*", default=[120, 376, 540, 796] ,help="Crop index. If not specified, take the values : [120, 376, 540, 796] (SE_indexes). If no crop is wanted, pass 0 as an argument. Ex: -c 120 376 540 796 for SE_indexes; -c 0 for no crop")
    parser.add_argument("-s", "--save_path", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/", help="Path where data is saved")
    parser.add_argument("-r", "--refresh", type=int, default=10, help="Frequence at which progress is shown")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")

    args = parser.parse_args()


    #### GLOBAL ####
    VAR_NAMES = ["rr", "u", "v", "t2m"]
    INDEXES = args.crop_index # SE_INDEXES = (120, 376, 540, 796); FR_INDEXES = (20, 680, 150, 972)
    CROPPING = (len(INDEXES) == 4)


    #### PATH ####
    RAW_DATA_DIR = args.load_directory
    SAVE_DIR = args.save_path + "pre_proc_" + args.save_directory + "/"

    # rename_rr_files(RAW_DATA_DIR, args)
    # process_all(RAW_DATA_DIR, SAVE_DIR, CROPPING, args)
    # merge_into_gigafiles(SAVE_DIR, "splitted", args)
    if CROPPING:
        merge_into_gigafiles(SAVE_DIR, "cropped", args)
