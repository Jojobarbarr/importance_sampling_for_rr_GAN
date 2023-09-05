from argparse import ArgumentParser

import matplotlib
import numpy as np
import pandas as pd
from called.utile import make_save_dir

matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796/", help="Directory where samples are.")
    parser.add_argument("-c", "--csv_dir", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/data_for_importance_sampling/pre_proc_11-08/splitted/", help="Directory where csv is.")
    sample_choice = parser.add_mutually_exclusive_group()
    sample_choice.add_argument("-s", "--sample_num", type=str, help="Sample name drawn, if not specified, choose a random one.")
    sample_choice.add_argument("-n", "--n_samples", type=int, default=1, help="Number of random samples if sample_num isn't specified")

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity, count")
    parser.add_argument("--save_dir", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/data_for_importance_sampling/visual/", help="Directory where figures are saved.")
    
    args = parser.parse_args()

    save_dir_img = args.save_dir
    make_save_dir(save_dir_img, args)
    raw_data_dir = args.data_dir
    csv_dir = args.csv_dir
    sample_name = f"_sample{args.sample_num}"
    if not args.sample_num:
        dataframe = pd.read_csv(f"{csv_dir}labels.csv")
        with open(f"{save_dir_img}labels.csv", "w", encoding="utf8") as file:
            file.write(f"Name,Date,Leadtime,Member\n")
        for _ in range(args.n_samples):
            row = dataframe.sample(1)
            with open(f"{save_dir_img}labels.csv", "a", encoding="utf8") as file:
                file.write(f"{row.Name.values[0]},{row.Date.values[0]},{row.Leadtime.values[0]},{row.Member.values[0]}\n")
            sample_name = row.Name.values[0]
            grid = np.load(f"{raw_data_dir}{sample_name}.npy")[0]
            plt.clf()
            fig, axes = plt.subplots()
            img = axes.imshow(grid, origin="lower")
            fig.colorbar(img)
            fig.savefig(f"{save_dir_img}{sample_name}.png")
            plt.close(fig)
    else:
        grid = np.load(f"{raw_data_dir}{sample_name}.npy")[0]
        plt.clf()
        fig, axes = plt.subplots()
        img = axes.imshow(grid, origin="lower")
        fig.colorbar(img)
        fig.savefig(f"{save_dir_img}{sample_name}.png")
        plt.close(fig)


