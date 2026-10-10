import importlib.util
import json
from pathlib import Path
import sys
import tarfile

base = Path('/root/home/ubuntu/research')
root = base / 'runs/future_a_v4'
source = base / 'h-scaffold-bpe-future-a-v5'
assert json.loads((root / 'status.json').read_text())['status'] == 'COMPLETE'
sys.path[:0] = [str(source / 'src'), str(source / 'research/src'), str(source / 'research/scripts')]
spec = importlib.util.spec_from_file_location('future_controller', source / 'research/scripts/a_future_utility.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
contract = module.verify(root)
def digest(path):
    return module.file_hash(path)
checkpoints = []
contract_hash = digest(root / 'contract.json')
for method in contract['methods']:
    for seed in contract['seeds']:
        folder = root / f'runs/{method}/seed{seed}'
        identity = json.loads((folder / 'identity.json').read_text())
        assert identity == dict(contract_hash=contract_hash, method=method, seed=seed)
        model = folder / 'model.pt'
        assert model.stat().st_size > 1_000_000
        checkpoints.append(dict(method=method, seed=seed, path=str(model), bytes=model.stat().st_size,
                                sha256=digest(model)))
(root / 'CHECKPOINTS.json').write_text(json.dumps(dict(source_sha=contract['source_sha'],
        contract_hash=contract_hash, files=checkpoints), indent=2))
files = [root / name for name in ['contract.json', 'contract_draft.json', 'decision.json', 'status.json',
         'selection.json', 'EXPOSURE.json', 'CHECKPOINTS.json', 'pg19_manifest.json', 'pg19.jsonl',
         'score_samples.npz', 'eligibility_support.npy']]
files += sorted((root / 'tok').glob('*.json'))
for folder in sorted((root / 'runs').glob('*/seed*')):
    files += [folder / name for name in ['results.json', 'identity.json', 'log.jsonl']]
    files += sorted(folder.glob('eval_*.npz'))
with tarfile.open('/tmp/mlp-future-a-v4-results.tar.gz', 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(root)))
    archive.add(base / 'runs/future_a_v4_runner.py', arcname='RUNNER.py')
    archive.add(base / 'A_GPU_TAKEOVER.json', arcname='COMPLETION_MARKER.json')
print(json.dumps(dict(status='PASS', checkpoints=len(checkpoints), artifacts=len(files),
        archive='/tmp/mlp-future-a-v4-results.tar.gz', archive_sha256=digest(Path('/tmp/mlp-future-a-v4-results.tar.gz')))))
