import glob
import os
from argparse import ArgumentParser
from time import perf_counter

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
        os.rename(
            filename,
            raw_data_dir
            + filename[len(raw_data_dir) + 7 : len(raw_data_dir) + 27]
            + "_rrlt1-24.npy",
        )


def process_all(raw_data_dir, variable_name_list, save_dir, args):
    """transform the datafiles in a way more practical with one file for each day / lead time / member and with all 4 variables being the channels in the order (rr, u, v, t2mS)

    Args:
        raw_data_dir (str): Directory of the raw data
        save_dir (str): Directory where the data is saved
        args (argparse.Namespace): args of the program
    """
    save_dir_splitted = f"{save_dir}splitted/"
    make_save_dir(save_dir_splitted, args)
    with open(f"{save_dir_splitted}labels.csv", "w", encoding="utf8") as csv_file:
        csv_file.write(f"Name,Date,Leadtime,Member\n")

    variable_date_dict = {
        variable: {
            datafile.name[:21]: datafile
            for datafile in os.scandir(raw_data_dir)
            if f"{variable}lt1-24.npy" in datafile.name
        }
        for variable in variable_name_list
    }
    n_days = len(variable_date_dict["rr"])
    for variable, date_dict in variable_date_dict.items():
        if len(date_dict) != n_days:
            raise ValueError(
                f"Day number for variable {variable} is different than for variable rr"
            )
    start_time = perf_counter()
    for day_count, date in enumerate(variable_date_dict["rr"]):
        if (day_count + 1) % ((n_days // args.refresh) + 1) == 0:
            print_progress(day_count, n_days, start_time)
        if args.verbose >= 1:
            print(f"Day {date[:-1]}")
        array_at_date_set = [
            np.load(variable_date_dict[variable][date].path)
            for variable in variable_name_list
        ]
        if args.verbose >= 2:
            print(f"Merging...")
        merged_array = np.array(
            [array for array in array_at_date_set]
        )  # [nbr_var, x, y, leadtimes, members]
        if args.verbose >= 2:
            print(f"Splitting{' and cropping' * args.crop}...")
        splitted_by_members_arrays = np.array(
            [merged_array[:, :, :, :, member] for member in range(16)]
        )  # [members, nbr_var, x, y, leadtimes]
        for member, splitted_by_members_array in enumerate(
            splitted_by_members_arrays
        ):  # [nbr_var, x, y, leadtimes]
            splitted_by_leadtime_arrays = np.array(
                [splitted_by_members_array[:, :, :, leadtime] for leadtime in range(24)]
            )  # [leadtimes, nbr_var, x, y]
            for leadtime, splitted_by_leadtime_array in enumerate(
                splitted_by_leadtime_arrays
            ):  # [nbr_var, x, y]
                final_array = splitted_by_leadtime_array
                sample_name = f"_sample{day_count * 384 + member * 24 + leadtime}"
                np.save(f"{save_dir_splitted}{sample_name}.npy", final_array)
                with open(
                    f"{save_dir_splitted}labels.csv", "a", encoding="utf8"
                ) as file:
                    file.write(f"{sample_name},{date[:-1]},{leadtime},{member}\n")
                if args.crop:
                    final_array_cropped = crop(final_array, save_dir, sample_name, args)


def crop(array, save_dir, sample_name, args):
    """Crop the grids according to args.crop_indexes

    Args:
        l_grid (list[tuple[numpy.array, str]]): a list of shape [n_grid, 2] with the tuple containing grids of shape [4, 717, 1121] and the sample name
        save_dir (str): Directory where the data is saved
        args (argparse.Namespace): args of the program
    """
    index_string = "_".join([str(index) for index in args.crop_indexes])
    save_dir_cropped = f"{save_dir}cropped_{index_string}/"
    make_save_dir(save_dir_cropped, args)
    x_min, x_max, y_min, y_max = args.crop_indexes
    array = array[:, x_min:x_max, y_min:y_max]  # array shape = [nbr_var, x, y]
    np.save(f"{save_dir_cropped}{sample_name}", array)


def merge_into_gigafiles(data_dir, datatype, args):
    """Merge the numerous little files in gigafiles to load patches at once and accelerate data computation like importance sampling or cropping

    Args:
        working_dir (str): Working directory
        datatype (str): "splitted" or "cropped"
        args (argparse.Namespace): args of the program
    """
    s_datatype = {"splitted", "cropped"}
    d_datatype_max_file_loaded = {"splitted": 1000, "cropped": 8000}
    if datatype not in s_datatype:
        raise ValueError(
            "Datatype must be in " + str(s_datatype) + " you gave '" + datatype + "'."
        )
    dataframe = pd.read_csv(f"{data_dir}splitted/labels.csv")
    index_string = "_".join([str(index) for index in args.crop_indexes])
    if datatype == "cropped":
        data_dir = f"{data_dir}{datatype}_{index_string}"
    else:
        data_dir = f"{data_dir}{datatype}"
    save_dir = f"{data_dir}_giga/"
    data_dir += f"/"
    make_save_dir(save_dir, args)
    with open(f"{save_dir}labels.csv", "w", encoding="utf8") as file:
        file.write("Name,Date,Leadtime,Member,Gigafile,Localindex\n")
    handle_patch(
        dataframe, data_dir, save_dir, d_datatype_max_file_loaded[datatype], args
    )


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
    n_patch = n_files // max_files_loaded + 1
    begin = 0
    end = min(max_files_loaded, n_files)
    for patch in range(1, n_patch + 1):
        if args.verbose >= 2:
            print(f"Patch {patch}/{n_patch}")
        giga = load(dataframe, data_dir, save_dir, begin, end, patch, args)
        np.save(f"{save_dir}{patch}.npy", giga)
        del giga
        begin = end
        files_processed = patch * max_files_loaded
        end = min(begin + max_files_loaded, begin + n_files - files_processed)


def load(dataframe, data_dir, save_dir, beg, end, patch, args):
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
    if args.verbose >= 3:
        print(f"Loading data, {n_tot} files...")
    data = []
    start_time = perf_counter()
    for index in range(beg, end):
        row = dataframe.iloc[index]
        if args.verbose >= 3 and (index + 1) % (n_tot // args.refresh) == 0:
            print_progress(index - beg, n_tot, start_time)
        data.append(np.load(f"{data_dir}{row['Name']}.npy"))
        with open(f"{save_dir}labels.csv", "a", encoding="utf8") as file:
            file.write(
                f"{row['Name']},{row['Date']},{row['Leadtime']},{row['Member']},{patch},{index-beg}\n"
            )
    return data


def crop_from_gigafiles(data_path_giga, save_dir, args):
    """Crops from gigafile splitted.

    Args:
        data_path_giga (str): Data path to the directory of the gigafiles
        save_dir (str): Directory where data is saved
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 1:
        print("Cropping from gigafiles...")
    dataframe = pd.read_csv(data_path_giga + "labels.csv")
    data_group_gigafile = dataframe.groupby("Gigafile")
    n_gigafiles = sum(1 for entry in os.scandir(data_path_giga)) - 1
    start_time = perf_counter()
    for index, entry in enumerate(os.scandir(data_path_giga)):
        if args.verbose >= 2:
            print("Processing Gigafile", index + 1, "of", n_gigafiles)
            if args.verbose >= 3 and (index + 1) % (n_gigafiles // args.refresh) == 0:
                print_progress(index, n_gigafiles, start_time)
        if entry.name != "labels.csv":
            l_grid = np.load(entry.path, allow_pickle=True)
            l_grid = [
                (l_grid[row.Localindex], row.Name)
                for row in data_group_gigafile.get_group(
                    int(entry.name[:-4])
                ).itertuples()
            ]
            crop(l_grid, save_dir, args)
            print("\n")
    merge_into_gigafiles(save_dir, "cropped", args)


def split(data_dir, args):
    """Split gigafiles into small unit files.

    Args:
        data_path_giga (str): Data path to the directory of the gigafiles
        save_dir (str): Directory where data is saved
        args (argparse.Namespace): args of the program
    """
    if args.verbose >= 1:
        print("Splitting from gigafiles...")
    make_save_dir(f"{data_dir}/", args)
    with open(f"{data_dir}/labels.csv", "w", encoding="utf8") as file:
        file.write("Name,Date,Leadtime,Member,Gigafile,Localindex\n")
    dataframe = pd.read_csv(f"{data_dir}_giga/labels.csv")
    data_group_gigafile = dataframe.groupby("Gigafile")
    gigafiles_set = {
        gigafile for gigafile in os.scandir(data_path) if gigafile.name != "labels.csv"
    }
    n_gigafiles = len(gigafiles_set)
    start_time = perf_counter()
    for idx_gigafile, gigafile in enumerate(gigafiles_set):
        if args.verbose >= 2:
            print("Processing Gigafile", idx_gigafile + 1, "of", n_gigafiles)
            if (
                args.verbose >= 3
                and (idx_gigafile + 1) % ((n_gigafiles // args.refresh) + 1) == 0
            ):
                print_progress(idx_gigafile, n_gigafiles, start_time)
        l_grid = np.load(gigafile.path)
        for row in data_group_gigafile.get_group(int(gigafile.name[:-4])).itertuples():
            grid = l_grid[row.Localindex]
            with open(save_dir + "labels.csv", "a", encoding="utf8") as file:
                file.write(
                    f"{row.Name},{row.Date},{row.Leadtime},{row.Member},{gigafile.name},{row.Localindex}\n"
                )
            np.save(data_dir + row.Name + ".npy", grid)


if __name__ == "__main__":
    ## ARGPARSE ##
    parser = ArgumentParser()

    parser.add_argument(
        "save_directory",
        type=str,
        default=None,
        help="Directory where data will be saved: 'pre_proc_' + save_directory",
    )
    parser.add_argument(
        "-l",
        "--load_directory",
        type=str,
        default="/cnrm/recyf/NO_SAVE/Data/users/brochetc/float32_t2m_u_v_rr/",
        help="Data directory from which data is loaded",
    )
    parser.add_argument(
        "-c", "--crop", action="store_true", help="Crop while processing all"
    )
    parser.add_argument(
        "--crop_indexes",
        type=int,
        nargs="*",
        default=[120, 376, 540, 796],
        help="Crop index. If not specified, take the values : [120, 376, 540, 796] (SE_indexes). If no crop is wanted, pass 0 as an argument. Ex: --crop_indexes 120 376 540 796 for SE_indexes; --crop_indexes 0 for no crop",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/data_for_importance_sampling/",
        help="Path where data is saved",
    )
    parser.add_argument(
        "-r",
        "--refresh",
        type=int,
        default=5,
        help="Frequence at which progress is shown",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity"
    )

    parser.add_argument("--split_dir", type=str, help="When using split() on priam")

    args = parser.parse_args()

    ## GLOBAL ##
    variable_name_list = ["rr", "u", "v", "t2m"]
    INDEXES = (
        args.crop_indexes
    )  # SE_INDEXES = (120, 376, 540, 796); FR_INDEXES = (20, 680, 150, 972)
    CROPPING = len(INDEXES) == 4

    #### INIT PREPROC ####
    # nohup python3 -u pre_proc_for_is.py -vv -c 11-08 > pre_proc.out 2> pre_proc.err &
    # ## PATH ##
    RAW_DATA_DIR = args.load_directory
    SAVE_DIR = f"{args.save_path}pre_proc_{args.save_directory}/"

    # rename_rr_files(RAW_DATA_DIR, args)
    # process_all(RAW_DATA_DIR, variable_name_list, SAVE_DIR, args)
    # merge_into_gigafiles(SAVE_DIR, "splitted", args)
    if args.crop:
        merge_into_gigafiles(SAVE_DIR, "cropped", args)

    #### CROPPING FROM GIGA SPLITTED ####

    # ## PATH ##
    # DATA_PATH_GIGA = args.load_directory
    # SAVE_DIR = args.save_path + "pre_proc_" + args.save_directory + "/"

    # crop_from_gigafiles(DATA_PATH_GIGA, SAVE_DIR, args)

    #### SPLITTING FROM GIGAFILE ####
    ## PATH ##
    # DATA_PATH = args.split_dir
    # split(DATA_PATH, args)
