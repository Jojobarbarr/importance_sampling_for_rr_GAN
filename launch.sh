for qmin in 0
do
	for m in 0.1 3 5
	do
		for rr in 1 3 5
		do
			nohup python3 -u main.py -vv -r 5 -p $qmin $m 1 0 $rr 0 --n_instances 50 pre_proc_31-07-10h/cropped_giga/ 2-08-10h_256/ > "output/$qmin-$m-$rr.txt" 2> "output/$qmin-$m-$rr.err" &
		done
	done
done
