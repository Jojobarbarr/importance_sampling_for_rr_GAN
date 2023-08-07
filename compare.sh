dir="7-08-14h/"
t=0.2

python3 compare_hist_2_by_2.py -vv -r 0 20 -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r 4 20 -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r 20 50 -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r 50 70 -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r 0 4 -l -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r 0 1.5 -ll -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r -0.7 0.7 -l -s -t $t $dir > output/out.out &
python3 compare_hist_2_by_2.py -vv -r -0.7 0.9 -ll -s -t $t $dir > output/out.out &
