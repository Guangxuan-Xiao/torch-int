models="opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b"
precisions="int8-fp32 int8-fp16 fp16"
batch_sizes="1 2 4"
seq_lens="512 1024"

for seq_len in $seq_lens; do
    for batch_size in $batch_sizes; do
        for model in $models; do
            for precision in $precisions; do
                python profiling/profile_opt.py --model $model --precision $precision --seq-len $seq_len --batch-size $batch_size
            done
        done
    done
done