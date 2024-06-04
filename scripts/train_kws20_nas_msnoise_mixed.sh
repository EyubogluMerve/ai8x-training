#!/bin/sh
python train.py --data '/data_ssd' --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --print-freq 100 --deterministic --qat-policy policies/qat_policy_late_kws20.yaml --compress policies/schedule_kws20.yaml --model ai85kws20netnas --use-bias --dataset KWS_20_msnoise_mixed --confusion --device MAX78000 "$@"
