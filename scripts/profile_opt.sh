models="opt-6.7b opt-13b"
precisions="int8-fp16 fp16"
batch_sizes="1 2 4"
seq_lens="256 512 1024"

for seq_len in $seq_lens; do
    for batch_size in $batch_sizes; do
        for model in $models; do
            for precision in $precisions; do
                CUDA_VISIBLE_DEVICES=4 python profiling/profile_opt.py --model $model --precision $precision --seq-len $seq_len --batch-size $batch_size
            done
        done
    done
done