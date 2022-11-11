models="opt-13b opt-30b"
precisions="int8-fp16-dynamic-a-token int8-fp16-dynamic-a-tensor llm_int8_0"
batch_sizes="4"
seq_lens="256 512"
output_file=log/ablate_quantization_scheme.csv
touch $output_file

echo "model,precision,batch_size,seq_len,latency(ms),memory(MB)" >$output_file
for seq_len in $seq_lens; do
    for batch_size in $batch_sizes; do
        for model in $models; do
            for precision in $precisions; do
                output=$(CUDA_VISIBLE_DEVICES=0 python benchmark/bench_opt.py --model $model --precision $precision --seq-len $seq_len --batch-size $batch_size)
                ms=$(echo "$output" | grep "Average inference time:" | awk '{print $4}')
                mem=$(echo "$output" | grep "Peak memory usage:" | awk '{print $4}')
                echo "$model,$precision,$batch_size,$seq_len,$ms,$mem" | tee -a $output_file
            done
        done
    done
done
