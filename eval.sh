
cleanup() {
    echo "Terminating all processes..."
    pkill -f eval_vln_r2r_6.py
    echo "All processes terminated."
}


trap cleanup EXIT SIGINT
#sleep 5h 
for i in {0,1,2,3,4,5,6,7}
do
    CUDA_VISIBLE_DEVICES=$((i)) python eval_vln_r2r_6.py --model_id $((i+1))  &
done

wait
