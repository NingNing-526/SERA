"""One entry point for SERA validation-selected training and held-out testing."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
VARIANTS = ('full', 'wo_sem', 'wo_ret', 'wo_align', 'wo_aux', 'wo_tgf')

def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'train', 'test'])
    p.add_argument('--config', default=str(ROOT / 'configs/default.json'))
    p.add_argument('--run-dir', help='Test: directory produced by train; loads its saved configuration')
    p.add_argument('--dataset', choices=['mosi', 'mosei'])
    p.add_argument('--variant', choices=VARIANTS)
    for arg in ('bert-path', 'raw-data', 'semantic-features', 'output', 'device'):
        p.add_argument('--' + arg)
    for arg in ('seed', 'batch-size', 'epochs'):
        p.add_argument('--' + arg, type=int)
    p.add_argument('--top-k', nargs=3, type=int, metavar=('TEXT', 'AUDIO', 'VISION'))
    a = p.parse_args()
    if a.mode == 'test':
        if not a.run_dir:
            p.error('test requires --run-dir; its training configuration will be reused')
        if any(getattr(a, k) is not None for k in ('dataset','variant','seed','batch_size','epochs','top_k','output')):
            p.error('test must reuse training architecture/seed/batch settings')
        path = Path(a.run_dir).resolve() / 'config.json'
    else:
        path = Path(a.config).resolve()
    c = json.loads(path.read_text(encoding='utf-8'))
    for k in c:
        val = getattr(a, k, None)
        if val is not None:
            c[k] = val
    if a.mode == 'test':
        c['output'] = str(Path(a.run_dir).resolve())
    elif a.dataset or a.variant or a.seed:
        if not a.output:
            c['output'] = f"runs/{c['dataset']}_{c['variant']}_seed{c['seed']}"
        if a.dataset and not a.raw_data:
            c['raw_data'] = f"data/{c['dataset']}/unaligned_50.pkl"
        if a.dataset and not a.semantic_features:
            c['semantic_features'] = f"data/{c['dataset']}/gemma_semantic_tokens_v11_len256.pt"
    for k in ('bert_path','raw_data','semantic_features','output'):
        v = Path(c[k]).expanduser()
        c[k] = str((ROOT / v).resolve() if not v.is_absolute() else v.resolve())
    if c['dataset'] not in ('mosi','mosei') or c['variant'] not in VARIANTS:
        p.error('unsupported dataset or variant')
    if min(c['batch_size'], c['epochs'], *c['top_k']) < 1:
        p.error('batch size, epochs and Top-K must be positive')
    if not 0 <= c['seed'] < 2**32:
        p.error('seed must be in [0, 2**32)')
    return a, c

class Tee:
    def __init__(self, terminal, file): self.terminal, self.file = terminal, file
    def write(self, s): self.terminal.write(s); self.file.write(s); self.file.flush()
    def flush(self): self.terminal.flush(); self.file.flush()
    def isatty(self): return False

def main():
    a, c = parse()
    for key in ('bert_path', 'raw_data') + (() if c['variant'] == 'wo_sem' else ('semantic_features',)):
        if not Path(c[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {c[key]}")
    os.environ['SERA_SETTINGS'] = json.dumps(c)
    sys.path.insert(0, str(ROOT / 'src'))
    import config
    from utils import set_random_seed
    from engine.data import _build_eval_dataloader
    from engine.training import SERA_train
    from engine.evaluation import SERA_test
    set_random_seed(c['seed'])
    if a.mode == 'check':
        from model.sera import SERA
        import torch
        model = SERA(config=config).to(config.DEVICE).eval()
        for split in ('train','valid','test'):
            loader = _build_eval_dataloader(c['dataset'], split, c['batch_size'], config)
            with torch.no_grad():
                pred, loss, _, _ = model(next(iter(loader)), None, return_loss=True)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                raise RuntimeError(f'Nonfinite output: {split}')
            print(f'PASS {split}: samples={len(loader.dataset)}, prediction_shape={tuple(pred.shape)}')
            del loader
        return
    out = Path(c['output'])
    if a.mode == 'train' and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {out}; choose --output')
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('logs','results','checkpoints'): (out / folder).mkdir(exist_ok=True)
    if a.mode == 'train': (out / 'config.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
    status = out / f'{a.mode}.status'
    status.write_text('RUNNING\n', encoding='utf-8')
    oldout, olderr = sys.stdout, sys.stderr
    with (out / 'logs' / f'{a.mode}.log').open('a', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(oldout, log), Tee(olderr, log)
        try:
            print(json.dumps(c, indent=2))
            if a.mode == 'train':
                result = SERA_train(dataset=c['dataset'], check={'Loss': float('inf')}, config=config)
                name = 'validation'
            else:
                result = SERA_test(dataset=c['dataset'], check_list=['Loss'], config=config)['Loss']
                name = 'test'
                with (out/'results/test.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=['seed', *result.keys()])
                    w.writeheader(); w.writerow(dict(seed=c['seed'], **result))
            (out/f'results/{name}.json').write_text(json.dumps(result, indent=2, default=lambda x: x.item()), encoding='utf-8')
            status.write_text('COMPLETED\n', encoding='utf-8')
        except BaseException:
            traceback.print_exc(); status.write_text('FAILED\n', encoding='utf-8'); raise
        finally:
            sys.stdout, sys.stderr = oldout, olderr

if __name__ == '__main__':
    main()
