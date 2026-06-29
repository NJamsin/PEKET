#!/bin/bash
export PATH="/home/liteul/anaconda3/envs/peket/bin:$PATH"
BANK_NUM=$1
START_TIME=$2
END_TIME=$3
TT=$4
LONGSLIDE=$5

L1_START=$((START_TIME + LONGSLIDE))
L1_END=$((END_TIME + LONGSLIDE))

export MPLCONFIGDIR="/tmp/matplotlib_noah_${BANK_NUM}_${START_TIME}_${LONGSLIDE}"
export ASTROPY_CACHE_DIR="/tmp/astropy_noah_${BANK_NUM}_${START_TIME}_${LONGSLIDE}"
mkdir -p $MPLCONFIGDIR
mkdir -p $ASTROPY_CACHE_DIR

mkdir -p /home/liteul/peket/test_injection/significance/out/bank_${BANK_NUM}

/home/liteul/anaconda3/envs/peket/bin/pycbc_multi_inspiral \
    -v \
    --instruments H1 L1 \
    --bank-file /home/liteul/peket/test_injection/inj_split/split_bank_${BANK_NUM}.hdf \
    --channel-name H1:GWOSC-4KHZ_R1_STRAIN L1:GWOSC-4KHZ_R1_STRAIN \
    --frame-cache H1:/home/liteul/peket/test_injection/data/background/inj_background_H1.lcf L1:/home/liteul/peket/test_injection/data/background/inj_background_L1.lcf \
    --gps-start-time H1:${START_TIME} L1:${L1_START} h1:${START_TIME} l1:${L1_START}\
    --gps-end-time H1:${END_TIME} L1:${L1_END} h1:${END_TIME} l1:${L1_END} \
    --ra 3.4461587900190143 \
    --dec -0.4080838827694336 \
    --trigger-time ${TT} \
    --low-frequency-cutoff 30.0 \
    --approximant TaylorF2 \
    --order 7 \
    --sample-rate H1:4096 L1:4096 h1:4096 l1:4096 \
    --pad-data H1:8 L1:8 h1:8 l1:8 \
    --segment-length H1:984 L1:984 h1:984 l1:984 \
    --segment-start-pad H1:8 L1:8 h1:8 l1:8 \
    --segment-end-pad H1:8 L1:8 h1:8 l1:8 \
    --psd-estimation H1:median L1:median h1:median l1:median \
    --psd-segment-length H1:16 L1:16 h1:16 l1:16 \
    --psd-segment-stride H1:8 L1:8 h1:8 l1:8 \
    --psd-inverse-length H1:16 L1:16 h1:16 l1:16 \
    --strain-high-pass H1:20 L1:20 h1:20 l1:20 \
    --autogating-threshold H1:50 L1:50 h1:50 l1:50 \
    --autogating-cluster H1:0.5 L1:0.5 h1:0.5 l1:0.5 \
    --autogating-width H1:0.25 L1:0.25 h1:0.25 l1:0.25 \
    --autogating-pad H1:0.25 L1:0.25 h1:0.25 l1:0.25 \
    --autogating-taper H1:0.25 L1:0.25 h1:0.25 l1:0.25 \
    --sngl-snr-threshold 4.5 \
    --coinc-threshold 4 \
    --chisq-bins 16 \
    --cluster-method window \
    --cluster-window 1.0 \
    --do-shortslides \
    --slide-shift 0.501 \
    --output /home/liteul/peket/test_injection/significance/out/bank_${BANK_NUM}/inj_bg_bank${BANK_NUM}_${START_TIME}-${END_TIME}_ls${LONGSLIDE}.hdf
