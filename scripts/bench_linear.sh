python tests/bench_linear.py --precision int8 --seq-len 256 --C1 12288 --C2 12288
python tests/bench_linear.py --precision int8 --seq-len 256 --C1 12288 --C2 49152
python tests/bench_linear.py --precision int8 --seq-len 256 --C1 49152 --C2 12288
python tests/bench_linear.py --precision fp16 --seq-len 256 --C1 12288 --C2 12288
python tests/bench_linear.py --precision fp16 --seq-len 256 --C1 12288 --C2 49152
python tests/bench_linear.py --precision fp16 --seq-len 256 --C1 49152 --C2 12288


python tests/bench_linear.py --precision int8 --seq-len 512 --C1 12288 --C2 12288
python tests/bench_linear.py --precision int8 --seq-len 512 --C1 12288 --C2 49152
python tests/bench_linear.py --precision int8 --seq-len 512 --C1 49152 --C2 12288

python tests/bench_linear.py --precision fp16 --seq-len 512 --C1 12288 --C2 12288
python tests/bench_linear.py --precision fp16 --seq-len 512 --C1 12288 --C2 49152
python tests/bench_linear.py --precision fp16 --seq-len 512 --C1 49152 --C2 12288