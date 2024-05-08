#!/bin/sh
python train.py --data '/data_ssd' --out-dir logs/benchmark_v2/librispeech_var --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --qat-policy policies/qat_policy_late_kws20.yaml --compress policies/schedule_kws20.yaml --model ai85kws20netnas --use-bias --dataset KWS_12_benchmark --confusion --device MAX78000 "$@"
