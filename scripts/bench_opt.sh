models="opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b" # opt-30b
precisions="fp16 int8-fp16"
batch_sizes="1 2 4 8 16 32"
seq_lens="512 256 1024 128"
output_file=log/opt_latency_memory.csv
touch $output_file

echo "model,precision,batch_size,seq_len,latency(ms),memory(MB)" >$output_file
for seq_len in $seq_lens; do
    for batch_size in $batch_sizes; do
        for model in $models; do
            for precision in $precisions; do
                # The output is like this:
                # Namespace(batch_size=2, model='opt-125m', precision='int8-fp32', seq_len=128)
                # Average inference time: 9.13 ms
                # Peak memory usage: 295.54 MB
                output=$(python benchmark/bench_opt.py --model $model --precision $precision --seq-len $seq_len --batch-size $batch_size)
                ms=$(echo "$output" | grep "Average inference time:" | awk '{print $4}')
                mem=$(echo "$output" | grep "Peak memory usage:" | awk '{print $4}')
                echo "$model,$precision,$batch_size,$seq_len,$ms,$mem" | tee -a $output_file
            done
        done
    done
done
