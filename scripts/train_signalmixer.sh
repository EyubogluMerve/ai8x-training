#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --out-dir logs/signalmixer_kws_updated --qat-policy policies/qat_policy_late_kws20.yaml --compress policies/schedule_kws20.yaml --model ai85kws20netnas --use-bias --dataset signalmixer_all --confusion --device MAX78000 "$@" --data "/data_ssd"
