import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

D = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/analysis/27-07-16h30/"
arr = np.load(D + "-1-0_5-0_1-0_0-0_3-0_0-0/hist_-1-0_5-0_1-0_0-0_3-0_0-0.npy")
plt.clf()
plt.hist(arr, range = (20, 50), bins=100)
plt.savefig(D + "-1-0_5-0_1-0_0-0_3-0_0-0/hist_manip_1.png")

