#!/bin/bash
# aux_queue.sh <python> <src_research_dir> <data_root> <out_root> <gpu> <arm:seed>...
P=$1; SRC=$2; D=$3; O=$4; G=$5; shift 5
cd $SRC
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G
for j in "$@"; do
  arm=${j%%:*}; s=${j##*:}; out=$O/lm/$arm/seed$s
  [ -f $out/results.json ] && continue
  mkdir -p $out
  echo "$(date -Is) START $arm seed$s gpu$G"
  $P -m scaffold_margin.train_lm --data $D/$arm --out $out --preset small --seq 1024 \
    --batch_tokens 131072 --micro_batch 16 --epochs 2 --lr 6e-4 --warmup_steps 200 --amp fp16 \
    --seed $s --data_seed $s --compile --save_ckpt --aux --eval_sets val,test > $out/train.log 2>&1 \
    && echo "$(date -Is) DONE $arm seed$s $(grep -h '\[eval\]' $out/train.log | tr '\n' ' ')" \
    || echo "$(date -Is) FAIL $arm seed$s"
done
echo "$(date -Is) QUEUE_END gpu$G"
