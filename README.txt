From the beginning:
Open these 3 files and uncomment eveything in the main section then execute them one after the over

1) pre_proc_for_is.py <--- very long ~1-2 days
2) main.py <--- ~12h, up to 6 execution in parallel on sxgmap3 (~20Go RAM /execution)
3) compare_hist_2_by_2.py <--- ~1min

If splitting already done, it's possible to call crop_from_gigafiles from pre_proc_for_is, it will be quickier than cropping from little files.

Once cropping is done, only main.py and compare_hist_2_by_2.py are useful. 
