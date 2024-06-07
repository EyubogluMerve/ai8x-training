#!/bin/sh
python train.py --out-dir 'logs/kws35_dataset4' --data '/data_ssd' --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --qat-policy policies/qat_policy_late_kws20.yaml --compress policies/schedule_kws20.yaml --model ai85kws20netnas --use-bias --dataset KWS_35 --confusion --device MAX78000 "$@"
