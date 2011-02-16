rm -f p_out cache_test
vw -d $1 -e --Loss 1 -t --cache_file cache_test -i r_temp -p p_out
