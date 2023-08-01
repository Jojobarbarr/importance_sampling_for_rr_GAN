from argparse import ArgumentParser

from called.process_is import importance_sampling
from called.stats import run_stat, save_mix
from called.utile import make_save_dir, parse_float

#### ARGPARSE ####
parser = ArgumentParser()

parser.add_argument("directory", type=str, default=None, help="Data directory from which data is loaded")
parser.add_argument("save", type=str, default=None, help="Data directory where the data is saved")
parser.add_argument("-r", "--refresh", type=int, default=10, help="Frequence at which progress is shown")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
parser.add_argument("-p", "--param", type=float, nargs="*", help="Importance sampling parameters. If not specified, take the values : [0.01, 0.1, 1, 0, 1, 0]")
parser.add_argument("--main_path", type=str, default="/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/", help="Base path")
parser.add_argument("--n_instances", type=int, default=1, help="Number of instances")
parser.add_argument("--n_instances_pre", type=int, default=0, help="Number of instances already sampled")
parser.add_argument("--wind_importance", type=bool, default=False, help="True if you want to compute wind importance")

args = parser.parse_args()
if args.directory is None:
    raise ValueError("Must specified a directory with --d")
if args.param is None:
    args.param = [0.01, 0.1, 1, 0, 1, 0]
    print("Default parameter list :", args.param)

#### GLOBAL ####
## IMPORTANCE SAMPLING ##

Q_MIN, M, P, Q, S_RR, S_W = args.param

PARAMETERS = (Q_MIN, M, P, Q, S_RR, S_W)
PARAM_STRING = parse_float(str(Q_MIN) + "_" + str(M) + "_" + str(P) + "_" + str(Q) + "_" + str(S_RR) + "_" + str(S_W))

## STATS ##
VAR = "rr"
THRESHOLD = 0.1
GRIDSHAPE = [256, 256]

VAR_NAMES = ["rr", "u", "v", "t2m"]
VAR = VAR_NAMES.index(VAR)
THRESHOLD_STR = parse_float(str(THRESHOLD))
#### PATH ####
MAIN_PATH = args.main_path
DIRECTORY = args.directory

#### IMPORTANCE SAMPLING ####
# nohup python3 -u main.py -vv -r 5 -p -1 3 1 0 3 0 --n_instances 50 pre_proc_31-07-10h/cropped_giga/ 2-08-8h_256/ > output/-1_3_3.txt 2> output/-1_3_3.err &
## PATH ##
CSV_DIR = MAIN_PATH + DIRECTORY
DATA_DIR = MAIN_PATH + DIRECTORY
SAVE_DIR = MAIN_PATH + "importance_sampling/" + args.save + PARAM_STRING +"/"

DIRS = (CSV_DIR, DATA_DIR, SAVE_DIR)

importance_sampling(PARAMETERS, DIRS, args)

#### STATS ####
## PATH ##
CSV_DIR_IS = SAVE_DIR
SAVE_DIR = MAIN_PATH + "analysis/" + args.save + PARAM_STRING + "/"
DIRS = (CSV_DIR, CSV_DIR_IS, DATA_DIR, SAVE_DIR)

run_stat(VAR, GRIDSHAPE, DIRS, THRESHOLD, args)

print(PARAM_STRING, "DONE.")

#### VISUALIZE ####
## PATH ##
DATA_DIR = SAVE_DIR
SAVE_DIR = DATA_DIR + "mix/threshold_" + THRESHOLD_STR + "/" 
make_save_dir(SAVE_DIR, args)

DIRS = (DATA_DIR, SAVE_DIR)

print("Bootstrapping...")

save_mix(PARAM_STRING, THRESHOLD_STR, DIRS, args)
print("Bootstrapping done.")

