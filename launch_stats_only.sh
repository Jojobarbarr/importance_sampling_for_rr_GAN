dir="2-08-15h_256/"
t=0

for qmin in 0
do
	for m in 0.001 0.01 0.1 0.5
	do
		for rr in 0.5 5
		do
			nohup python3 -u main.py -vv -r 5 --stats_only -p $qmin $m 1 0 $rr 0 --n_instances 50 -t $t pre_proc_31-07-10h/cropped_giga/ $dir > "output/$qmin-$m-$rr.txt" 2> "output/$qmin-$m-$rr.err" &
		done
	done
done
