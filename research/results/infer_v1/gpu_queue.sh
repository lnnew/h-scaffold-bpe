#!/bin/bash
# usage: gpu_queue.sh <gpu> <slot 0|1>  -- runs every 2nd job of the fixed list
G=$1; SLOT=$2
cd /root/home/ubuntu/research/h-scaffold-bpe-infer-v1/research
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G
JOBS=()
for s in 0 1 2; do for arm in scaffold_canonical scaffold_dp bpe_canonical scaffold_longest scaffold_mix; do JOBS+=("$arm:$s"); done; done
i=0
for j in "${JOBS[@]}"; do
  if [ $((i % 2)) -eq $SLOT ]; then
    arm=${j%%:*}; s=${j##*:}; out=/root/home/ubuntu/research/runs/infer_v1/lm/$arm/seed$s
    if [ ! -f $out/results.json ]; then
      until [ -f /root/home/ubuntu/research/runs/infer_v1/data/$arm/meta.json ]; do sleep 60; done
      mkdir -p $out
      ev=val,test,test_canonical,test_longest,test_dp
      echo "$(date -Is) START $arm seed$s gpu$G"
      /root/home/ubuntu/research/h-scaffold-bpe/.venv/bin/python -m scaffold_margin.train_lm --data /root/home/ubuntu/research/runs/infer_v1/data/$arm --out $out --preset small --seq 1024         --batch_tokens 131072 --micro_batch 16 --epochs 2 --lr 6e-4 --warmup_steps 200         --seed $s --data_seed $s --compile --save_ckpt --eval_sets $ev > $out/train.log 2>&1         && echo "$(date -Is) DONE $arm seed$s $(grep -h "\[eval\]" $out/train.log | tr "\n" " ")"         || echo "$(date -Is) FAIL $arm seed$s"
    fi
  fi
  i=$((i+1))
done
echo "$(date -Is) QUEUE_END gpu$G"
