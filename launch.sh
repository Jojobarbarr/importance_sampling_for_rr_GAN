dir="7-08-14h/"

for qmin in 0.0001
do
	for m in 1 10 50 
	do
		for rr in 5 30
		do
			nohup python3 -u main.py -vv -r 5 -p $qmin $m 1 0 $rr 0 --n_instances 50 -t 0.2 pre_proc_31-07-10h/cropped_giga/ $dir > "output/$qmin-$m-$rr.txt" 2> "output/$qmin-$m-$rr.err" &
		done
	done
done

