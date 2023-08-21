dir="11-08-15h/"

for srr in 5 10
do
	for qmin in 0.001 0.00001
	do
		for m in 10 500 1000
		do
			nohup python3 -u main.py -vvv -r 5 -p $srr $qmin $m --n_instances 25 pre_proc_31-07-10h/cropped_giga/ 11-08-11h_default/ > output/$srr-$qmin-$m.txt 2> output/$srr-$qmin-$m.err &
		done
	done
done

