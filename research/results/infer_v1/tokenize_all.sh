set -e
cd /root/home/ubuntu/research
export PYTHONPATH=/root/home/ubuntu/research/h-scaffold-bpe-infer-v1/src:/root/home/ubuntu/research/h-scaffold-bpe-infer-v1/research/src
for job in bpe:canonical scaffold:canonical scaffold:longest scaffold:dp scaffold:mix; do
  tk=${job%%:*}; m=${job##*:}
  [ -f runs/infer_v1/data/${tk}_$m/meta.json ] && continue
  /root/home/ubuntu/research/h-scaffold-bpe/.venv/bin/python /root/home/ubuntu/research/h-scaffold-bpe-infer-v1/research/scripts/i_tokenize.py --tok /root/home/ubuntu/research/h-scaffold-bpe-infer-v1/artifacts/tokenizers/$tk-32768.json --text_dir data/text --method $m --out runs/infer_v1/data/${tk}_$m --workers 40
done
echo TOKENIZE_DONE
