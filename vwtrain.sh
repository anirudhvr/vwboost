rm -f cache_train cache_test r_temp
vw -d $1 -e --Loss 1 -l 1280000 --initial_t 128000 --power_t 1 --cache_file cache_train -f r_temp --passes 100
