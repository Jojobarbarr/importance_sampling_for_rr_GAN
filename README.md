# Pre-processing code for Météo France internship.
## Goal
The idea is to pre process a large dataset of AROME outputs. AROME is the model used by Météo France to simulate the atmosphere.
The code here is executed on a supercomputer in Météo France, and scripts to launch it are available.

## Use
From the beginning:
Open these 3 files and uncomment eveything in the main section then execute them one after the over

1) pre_proc_for_is.py <--- very long ~1 days
2) main.py <--- ~2-3h, up to 6 execution in parallel on sxgmap3 (~20Go RAM /execution)
                ~<1-2h, for stats_only (~20Go RAM /execution)
3) compare_hist_2_by_2.py <--- ~1min

If splitting already done, it's possible to call crop_from_gigafiles from pre_proc_for_is, it will be quickier than cropping from little files.

Once cropping is done, only main.py and compare_hist_2_by_2.py are useful.
It is then possible to use launch.sh to run importance sampling, just change the name of $dir and q_min, m and s_rr values.
To run stats only on an already importance sampled dataset, run launch_only_stats.sh with the correct values and name in $dir and $t.
To draw and save histograms between two sets of parameter, run compare.sh with the good name in $dir and the good value for threshold $t (-t option for main.py).
