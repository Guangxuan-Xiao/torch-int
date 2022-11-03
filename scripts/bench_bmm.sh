precisions="int8 fp16"
batch_sizes="1 4 8 16"
seq_lens="256 512 1024"
hidden_dims="12288"
output_file=log/bmm_latency.csv
touch $output_file

echo "precision,batch_size,seq_len,hidden_size,ms" >$output_file
for precision in $precisions; do
    for seq_len in $seq_lens; do
        for batch_size in $batch_sizes; do
            for hidden_dim in $hidden_dims; do
                echo "precision=$precision, seq_len=$seq_len, batch_size=$batch_size, hidden_dim=$hidden_dim"
                ms=$(python benchmark/bench_bmm.py --precision $precision --seq-len $seq_len --batch-size $batch_size --hidden-dim $hidden_dim | grep "Average inference time: " | cut -d " " -f 4)
                echo "$precision,$batch_size,$seq_len,$hidden_dim,$ms" | tee -a $output_file
            done
        done
    done
done
