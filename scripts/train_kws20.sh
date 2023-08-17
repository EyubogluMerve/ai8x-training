#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85kws20net --data "/data_ssd/raw/KWS" --dataset KWS_20 --confusion --device MAX78000 "$@"
