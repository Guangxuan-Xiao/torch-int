precisions="int8 fp16"
seq_lens="128 256 512 1024"

output_file=log/linear_kernels_latency.csv
touch $output_file

# C1s="12288 49152"
# C2s="12288 49152"
# fns="linear_a8_w8_b32_o32 linear_a8_w8_b8_o8 linear_relu_a8_w8_b8_o8"
# for precision in $precisions; do
#     for seq_len in $seq_lens; do
#         for C1 in $C1s; do
#             for C2 in $C2s; do
#                 for fn in $fns; do
#                     ms=$(python benchmark/bench_linear_kernels.py --precision $precision --seq-len $seq_len --C1 $C1 --C2 $C2 --func $fn | grep "Average inference time: " | cut -d " " -f 4)
#                     echo "$precision,$seq_len,$C1,$C2,$fn,$ms" >>$output_file
#                 done
#             done
#         done
#     done
# done

setting1="12288:49152:linear_relu_a8_w8_b8_o8:fc1" # fc1
setting2="49152:12288:linear_a8_w8_b32_o32:fc2"    # fc2
setting3="12288:12288:linear_a8_w8_b8_o8:qkv"      # q, k, v
setting4="12288:12288:linear_a8_w8_b32_o32:out"    # out_proj
echo "precision,seq_len,C1,C2,fn,name,ms" >$output_file
for precision in $precisions; do
    for seq_len in $seq_lens; do
        for setting in $setting1 $setting2 $setting3 $setting4; do
            C1=$(echo $setting | cut -d ":" -f 1)
            C2=$(echo $setting | cut -d ":" -f 2)
            fn=$(echo $setting | cut -d ":" -f 3)
            name=$(echo $setting | cut -d ":" -f 4)
            echo "precision=$precision, seq_len=$seq_len, C1=$C1, C2=$C2, fn=$fn, name=$name"
            ms=$(python benchmark/bench_linear_kernels.py --precision $precision --seq-len $seq_len --C1 $C1 --C2 $C2 --func $fn | grep "Average inference time: " | cut -d " " -f 4)
            echo "$precision,$seq_len,$C1,$C2,$fn,$name,$ms" | tee -a $output_file
        done
    done
done
