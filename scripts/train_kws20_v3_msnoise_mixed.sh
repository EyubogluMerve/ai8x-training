#!/bin/sh
python train.py --out-dir logs/v3_original --data '/data_ssd' --print-freq 100 --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85kws20netv3 --dataset KWS_20_msnoise_mixed --confusion --device MAX78000 "$@"
