#!/bin/bash
# wait for this GPU current run, then run seeds 0 and 2 (quiche backup)
G=$1; ARM=$2
until [ -f /root/home/ubuntu/research/runs/infer_v2/lm/$ARM/seed1/results.json ] || grep -q FAIL /root/home/ubuntu/research/runs/infer_v2/logs/gpu$G.log; do sleep 30; done
exec /root/home/ubuntu/research/runs/infer_v2/aux_queue.sh /root/home/ubuntu/research/h-scaffold-bpe/.venv/bin/python /root/home/ubuntu/research/h-scaffold-bpe-infer-v2/research /root/home/ubuntu/research/runs/infer_v2/data /root/home/ubuntu/research/runs/infer_v2 $G $ARM:0 $ARM:2
