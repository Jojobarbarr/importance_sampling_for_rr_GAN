dir="2-08-10h_256/"
t=0.1

python3 compare_hist_2_by_2.py -vv -r 0 30 -t $t $dir &
python3 compare_hist_2_by_2.py -vv -r 20 50 -t $t $dir &
python3 compare_hist_2_by_2.py -vv -r 50 70 -t $t $dir &
python3 compare_hist_2_by_2.py -vv -r 0 4 -l -t $t $dir &
python3 compare_hist_2_by_2.py -vv -r 0 1.5 -ll -t $t $dir &
python3 compare_hist_2_by_2.py -vv -r -0.7 0.7 -l -s -t $t $dir &
python3 compare_hist_2_by_2.py -vv -r -0.7 0.9 -ll -s -t $t $dir &
