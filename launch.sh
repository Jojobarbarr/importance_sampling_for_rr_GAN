dir="2-08-15h15_256/"

for qmin in -0.2 -0.001 0.01 0.2
do
	for m in 0.001 10
	do
		for rr in 0.5 10
		do
			nohup python3 -u main.py -vv -r 5 -p $qmin $m 1 0 $rr 0 --n_instances 50 -t 0 pre_proc_31-07-10h/cropped_giga/ $dir > "output/$qmin-$m-$rr.txt" 2> "output/$qmin-$m-$rr.err" &
		done
	done
done

