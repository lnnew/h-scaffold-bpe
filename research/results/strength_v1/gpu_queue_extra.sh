#!/bin/bash
# gpu_queue.sh <gpu> <slot> : every 2nd job of the fixed list (seed-major), bf16 H100, same schedule as infer_v1
G=$1; SLOT=$2; B=$HOME/jihyun_scaffold
cd $B/src-s1/research
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G
JOBS=(); for s in 3 4; do for c in 0 1 4 8; do JOBS+=("$c:$s"); done; done
i=0
for j in "${JOBS[@]}"; do
  if [ $((i % 2)) -eq $SLOT ]; then
    c=${j%%:*}; s=${j##*:}; out=$B/runs/strength/lm/c$c/seed$s
    if [ ! -f $out/results.json ]; then
      until [ -f $B/runs/strength/data/c$c/meta.json ]; do sleep 20; done
      mkdir -p $out; echo "$(date -Is) START c$c seed$s gpu$G"
      $B/venv/bin/python -m scaffold_margin.train_lm --data $B/runs/strength/data/c$c --out $out --preset small --seq 1024         --batch_tokens 131072 --micro_batch 64 --epochs 2 --lr 6e-4 --warmup_steps 200 --amp bf16         --seed $s --data_seed $s --compile --save_ckpt --eval_sets val,test > $out/train.log 2>&1         && echo "$(date -Is) DONE c$c seed$s $(grep -h "\[eval\]" $out/train.log | tr "\n" " ")"         || echo "$(date -Is) FAIL c$c seed$s"
    fi
  fi
  i=$((i+1))
done
echo "$(date -Is) QUEUE_END gpu$G"
