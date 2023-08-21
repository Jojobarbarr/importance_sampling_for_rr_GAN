nohup python3 -u main.py --stats_only -r 5 -p 5 0.001 500 -vvvv data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ /importance_sampling/16-08-12h/ > output/5_0.001_500.out 2> output/5_0.001_500.err &


nohup python3 -u main.py --stats_only -vvvv -r 5 -p 5 0.001 500 --old_param "-0-2 50-0 1-0 0-0 20-0 0-0" --force data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ del/output/importance_sampling/~NO/3-08-12h_256/ > output/OLD-0-2.out 2> output/OLD-0-2.err &
