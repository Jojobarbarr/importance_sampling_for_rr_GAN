from argparse import ArgumentParser

from called.process_is import importance_sampling, compute_c
from stats import run_stat

# nohup python3 -u main.py -vv -r 5 -p 5 0.001 500 --n_instances 50 pre_proc_31-07-10h/cropped_giga/ 11-08-11h_default/ > output/default.txt 2> output/default.err &

DEFAULT_PARAM = [5, 0.001, 500]
#### ARGPARSE ####
parser = ArgumentParser()

parser.add_argument("directory", type=str, help="Data directory from which data is loaded")
parser.add_argument("save", type=str, help="Data directory where the data is saved")
parser.add_argument("--l_c", type=float, nargs="*", help="The initial points for fsolve. MUST BE CLOSE TO THE ROOT")
parser.add_argument("-r", "--refresh", type=int, default=10, help="Frequence at which progress is shown")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
parser.add_argument("-p", "--param", type=float, nargs="*", help=f"Importance sampling parameters. If not specified, take the values : {DEFAULT_PARAM}")
parser.add_argument("-t", "--threshold", type=float, default=0, help="Threshold for stats")
parser.add_argument("-b", "--progress_bar", action="store_true", help="Print the progress bar")
parser.add_argument("-o", "--rough", action="store_true", help="If used, importance sampling is done with a rough ladder filter.")

parser.add_argument("--stats_only", action="store_true", help="If used, only run_stat and save_mix are executed")
parser.add_argument("--main_path", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/", help="Base path")
parser.add_argument("--n_instances", type=int, default=1, help="Number of instances")
parser.add_argument("--ignore_c", action="store_true", help="Don't execute fsolve to find c")
parser.add_argument("--ravuri", action="store_true", help="Importance sampling with the same function as Ravuri et al.")

######################################
parser.add_argument("--force", action="store_true")
parser.add_argument("--old_param", type=str)
######################################

args = parser.parse_args()
if args.param is None:
    args.param = DEFAULT_PARAM[5, 0.001, 500]
    print(f"Default parameter list: {args.param}")
if args.l_c is None:
    args.l_c = [1, 1.25]



#### GLOBAL ####
## IMPORTANCE SAMPLING ##
S_RR, Q_MIN, M = args.param
if not args.stats_only and not args.ignore_c:
    C = compute_c(S_RR, Q_MIN, M, args.l_c)
    print(f"c = {C}")
else:
    C = 0
PARAMETERS = (S_RR, Q_MIN, M, C)
PARAMETERS_STR = f"{S_RR}_{Q_MIN}_{M}"

## STATS ##
VARIABLE= f"rr"
THRESHOLD = args.threshold
GRIDSHAPE = (256, 256)

VAR_NAMES = (f"rr", f"u", f"v", f"t2m")
VARIABLE= VAR_NAMES.index(VARIABLE)
THRESHOLD_STR = f"{THRESHOLD}"
#### PATH ####
MAIN_PATH = f"{args.main_path}"
DIRECTORY = f"{args.directory}"

#### IMPORTANCE SAMPLING ####
## PATH ##
CSV_DIR = f"{MAIN_PATH}{DIRECTORY}"
DATA_DIR = f"{MAIN_PATH}{DIRECTORY}"
SAVE_DIR = f"{MAIN_PATH}{args.save}{PARAMETERS_STR}/"

DIRS = (CSV_DIR, DATA_DIR, SAVE_DIR)

if not args.stats_only:
    importance_sampling(PARAMETERS, DIRS, GRIDSHAPE, VARIABLE, args)

#### STATS ####
## PATH ##
######################################
if args.force:
    PARAMETERS_STR = args.old_param.split(' ')
    PARAMETERS_STR = '_'.join(PARAMETERS_STR)
    CSV_DIR = f"{MAIN_PATH}{DIRECTORY}"
    DATA_DIR = f"{MAIN_PATH}{DIRECTORY}"
    SAVE_DIR = f"{MAIN_PATH}{args.save}{PARAMETERS_STR}/"
    DIRS = (DATA_DIR, SAVE_DIR)
    run_stat(DIRS, VARIABLE, GRIDSHAPE, args)
######################################
else:
    DIRS = (DATA_DIR, SAVE_DIR)


    run_stat(DIRS, VARIABLE, GRIDSHAPE, args)

    print(PARAMETERS_STR, "DONE.")

    # #### VISUALIZE ####
    # ## PATH ##
    # DATA_DIR = SAVE_DIR
    # SAVE_DIR = DATA_DIR + "mix/threshold_" + THRESHOLD_STR + "/" 
    # # make_save_dir(SAVE_DIR, args)

    # DIRS = (DATA_DIR, SAVE_DIR)

    # print("Bootstrapping...")

    # # save_mix(PARAMETERS_STR, THRESHOLD_STR, DIRS, args)
    # print("Bootstrapping done.")

    print(f"DONE")