
cleanup() {
    echo "Terminating all processes..."
    pkill -f eval_vln_r2r_6.py
    echo "All processes terminated."
}



trap cleanup EXIT SIGINT SIGTERM

GPU_NUM=8
for i in $(seq 0 $((GPU_NUM-1)))
do
    CUDA_VISIBLE_DEVICES=$((i)) python eval_vln_r2r_6.py --model_id $((i+1))  &
done

wait
