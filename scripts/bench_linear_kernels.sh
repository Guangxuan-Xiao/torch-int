precisions="int8 fp16"
seq_lens="128 256 512 1024"
C1s="12288 49152"
C2s="12288 49152"
fns="linear_a8_w8_b32_o32 linear_a8_w8_b8_o8 linear_relu_a8_w8_b8_o8"

output_file=log/linear_kernels_latency.csv
touch $output_file
echo "precision,seq_len,C1,C2,fn,ms" >$output_file
for precision in $precisions; do
    for seq_len in $seq_lens; do
        for C1 in $C1s; do
            for C2 in $C2s; do
                for fn in $fns; do
                    ms=$(python benchmark/bench_linear_kernels.py --precision $precision --seq-len $seq_len --C1 $C1 --C2 $C2 --func $fn | grep "Average inference time: " | cut -d " " -f 4)
                    echo "$precision,$seq_len,$C1,$C2,$fn,$ms" >>$output_file
                done
            done
        done
    done
done
