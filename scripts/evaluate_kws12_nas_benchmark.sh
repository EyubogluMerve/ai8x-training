#!/bin/sh
python train.py --data '/data_ssd' --out-dir logs/test_results_v2/librispeech_var --model ai85kws20netnas --use-bias --dataset KWS_12_benchmark --confusion --evaluate --exp-load-weights-from logs/benchmark_v2/librispeech_var/2024.04.23-162837/qat_best.pth.tar --device MAX78000 "$@"
