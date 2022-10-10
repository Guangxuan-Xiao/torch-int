python tests/bench_quant_linear.py --precision int8 --seq-len 256 --C1 12288 --C2 12288
python tests/bench_quant_linear.py --precision int8 --seq-len 256 --C1 12288 --C2 49152
python tests/bench_quant_linear.py --precision int8 --seq-len 256 --C1 49152 --C2 12288
python tests/bench_quant_linear.py --precision int8 --seq-len 512 --C1 12288 --C2 12288
python tests/bench_quant_linear.py --precision int8 --seq-len 512 --C1 12288 --C2 49152
python tests/bench_quant_linear.py --precision int8 --seq-len 512 --C1 49152 --C2 12288

python tests/bench_quant_linear.py --precision int8 --seq-len 256 --C1 12288 --C2 12288 --act-quant per_tensor
python tests/bench_quant_linear.py --precision int8 --seq-len 256 --C1 12288 --C2 49152 --act-quant per_tensor
python tests/bench_quant_linear.py --precision int8 --seq-len 256 --C1 49152 --C2 12288 --act-quant per_tensor
python tests/bench_quant_linear.py --precision int8 --seq-len 512 --C1 12288 --C2 12288 --act-quant per_tensor
python tests/bench_quant_linear.py --precision int8 --seq-len 512 --C1 12288 --C2 49152 --act-quant per_tensor
python tests/bench_quant_linear.py --precision int8 --seq-len 512 --C1 49152 --C2 12288 --act-quant per_tensor
