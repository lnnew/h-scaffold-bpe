import json
import os
import select
import shutil
import subprocess
import time
from pathlib import Path

root = Path('/root/home/ubuntu/research')
source = root / 'h-scaffold-bpe-future-a-v5'
out = root / 'runs/future_a_v4'
python = root / 'h-scaffold-bpe/.venv/bin/python'
script = source / 'research/scripts/a_future_utility.py'
sha = '89e2da363a16edc00f454975136a99efe10ccc09'
marker = root / 'A_GPU_TAKEOVER.json'
backup = root / 'A_GPU_TAKEOVER_before_future_a_v4.json'
gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
owned = {line.split(',')[1].strip() for line in gpus.splitlines() if line.split(',')[0].strip() in {'0', '1'}}
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(line.split(',')[0].strip() in owned for line in apps.splitlines()):
    raise RuntimeError('GPU0/1 acquired another process; refusing to reassign')
if marker.exists():
    if backup.exists():
        raise RuntimeError('resource backup exists; refusing to overwrite prior evidence')
    shutil.copyfile(marker, backup)
state = dict(status='FUTURE_A_PREPARING', controller_pid=os.getpid(), source_sha=sha,
             source=str(source), run=str(out), owned_gpus=[0, 1])
def save():
    marker.write_text(json.dumps(state, indent=2))
save()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4')
try:
    for stage in ['prepare', 'run']:
        cmd = [str(python), '-u', str(script), '--stage', stage, '--out', str(out)]
        if stage == 'prepare':
            cmd += ['--root', str(root), '--teacher', str(root / 'runs/purpose_a_v1/teacher_pool40960/model.pt'),
                    '--source_sha', sha, '--score_cache', str(root / 'runs/future_a_v1'), '--training_cache', str(root / 'runs/future_a_v3')]
        child = subprocess.Popen(cmd, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        state.update(status='FUTURE_A_PREPARING' if stage == 'prepare' else 'FUTURE_A_TRAINING', child_pid=child.pid)
        save()
        last_progress = ''
        with (root / f'runs/future_a_v4_{stage}.log').open('w') as log:
            while child.poll() is None:
                ready, _, _ = select.select([child.stdout], [], [], 30)
                if ready:
                    line = child.stdout.readline()
                    if line:
                        log.write(line); log.flush(); print(line.rstrip(), flush=True)
                else:
                    progress = []
                    for gpu in [0, 1]:
                        p = out / f'gpu{gpu}_progress.json'
                        if p.exists():
                            item = json.loads(p.read_text()); item['gpu'] = gpu
                            log_path = out / f"runs/{item['method']}/seed{item['seed']}/log.jsonl"
                            if log_path.exists():
                                lines = log_path.read_text().splitlines()
                                if lines:
                                    item['step'] = json.loads(lines[-1])['step']
                            progress.append(item)
                    message = json.dumps(dict(stage=stage, progress=progress))
                    if message != last_progress:
                        print(message, flush=True); last_progress = message
            for line in child.stdout:
                log.write(line); print(line.rstrip(), flush=True)
        if child.returncode:
            raise RuntimeError(f'{stage} exited with {child.returncode}')
    state.update(status='FUTURE_A_COMPLETE', controller_pid=None, child_pid=None, returncode=0)
except Exception as error:
    state.update(status='FUTURE_A_FAILED', controller_pid=None, child_pid=None, error=str(error))
    raise
finally:
    save()
