# # 定义一个清理函数
# cleanup() {
#     echo "Terminating all processes..."
#     pkill -f eval_video_multi.py
#     echo "All processes terminated."
# }
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/LingXi/NVIDIA-Linux-x86_64-535.129.03
# # 捕获 EXIT 和 SIGINT 信号
# trap cleanup EXIT SIGINT
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENDPOINT=https://hf-mirror.com
# export HF_HOME=$HOME/.cache/hf_mirror
 #for i in {0,1}
 #do
 #    python eval_video_multi.py --model_id $((i+1)) --gpu_id $((i)) &
 #done

# wait

# 定义一个清理函数
cleanup() {
    echo "Terminating all processes..."
    pkill -f eval_vln_rxr_6.py
    pkill -f eval_vln_r2r_6.py
    echo "All processes terminated."
}

# 捕获 EXIT 和 SIGINT 信号
trap cleanup EXIT SIGINT
#sleep 5h 
for i in {0,1,2,3,4,5,6,7}
do
    CUDA_VISIBLE_DEVICES=$((i)) python eval_vln_rxr_6.py --model_id $((i+1))  &
    CUDA_VISIBLE_DEVICES=$((i)) python eval_vln_r2r_6.py --model_id $((i+1))  &
done

wait
