dir="16-08-12h/"

srr=5
qmin=0.001
m=500

c1=1
c2=1.25

nohup python3 -u main.py --stats_only -vvv -r 5 -p $srr $qmin $m --l_c $c1 $c2 --n_instances 25 data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ $dir > output/$srr-$qmin-$m.txt 2> output/err/$srr-$qmin-$m.err &

srr=20
qmin=0.001
m=500

c1=6.5
c2=7

nohup python3 -u main.py --stats_only -vvv -r 5 -p $srr $qmin $m --l_c $c1 $c2 --n_instances 25 data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ $dir > output/$srr-$qmin-$m.txt 2> output/err/$srr-$qmin-$m.err &

srr=5
qmin=0.0001
m=500

c1=1
c2=1.25

nohup python3 -u main.py --stats_only -vvv -r 5 -p $srr $qmin $m --l_c $c1 $c2 --n_instances 25 data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ $dir > output/$srr-$qmin-$m.txt 2> output/err/$srr-$qmin-$m.err &

srr=5
qmin=0.001
m=1000

c1=1
c2=1.1

nohup python3 -u main.py --stats_only -vvv -r 5 -p $srr $qmin $m --l_c $c1 $c2 --n_instances 25 data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ $dir > output/$srr-$qmin-$m.txt 2> output/err/$srr-$qmin-$m.err &

srr=5
qmin=0.001
m=10

c1=3.9
c2=4

nohup python3 -u main.py --stats_only -vvv -r 5 -p $srr $qmin $m --l_c $c1 $c2 --n_instances 25 data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/ $dir > output/$srr-$qmin-$m.txt 2> output/err/$srr-$qmin-$m.err &

