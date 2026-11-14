"""CPU end-to-end test with synthetic data and a tiny random BERT (not paper results)."""
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import os
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

ROOT = Path(__file__).resolve().parents[1]

def main():
    torch.manual_seed(12345)
    rng = np.random.default_rng(12345)
    with tempfile.TemporaryDirectory(prefix='sera_smoke_') as temp:
        root = Path(temp)
        bert = root / 'bert'
        bert.mkdir()
        (bert/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\nlike\nthis\nmovie\nbad\ngood\n', encoding='utf-8')
        BertTokenizer(str(bert/'vocab.txt')).save_pretrained(bert)
        BertModel(BertConfig(vocab_size=11, hidden_size=768, num_hidden_layers=1,
                            num_attention_heads=12, intermediate_size=128)).save_pretrained(bert)
        for dataset, adim, vdim in [('mosi',5,20),('mosei',74,35)]:
            raw, semantic = {}, {}
            for split, n in [('train',8),('valid',4),('test',4)]:
                ids = [f'{split}_{i}' for i in range(n)]
                raw[split] = dict(id=ids, raw_text=['i like this movie']*n,
                    audio=rng.normal(size=(n,375 if dataset == 'mosi' else 500,adim)).astype('float32'),
                    vision=rng.normal(size=(n,500,vdim)).astype('float32'),
                    text=rng.normal(size=(n,6,768)).astype('float32'),
                    audio_lengths=[5]*n, vision_lengths=[5]*n,
                    regression_labels=np.linspace(-2,2,n).astype('float32'))
                semantic[split] = {'ids':ids}
                for m in ('text','audio','vision'):
                    semantic[split][m] = torch.randn(n,20,32)
                    mask = torch.ones(n,20,dtype=torch.bool); mask[:,-2:] = False
                    semantic[split][m+'_mask'] = mask
            data_path, sem_path = root/f'{dataset}.pkl', root/f'{dataset}.pt'
            with data_path.open('wb') as f: pickle.dump(raw,f)
            torch.save(semantic,sem_path)
            c = json.loads((ROOT/'configs/default.json').read_text())
            c.update(dataset=dataset, bert_path=str(bert), raw_data=str(data_path),
                     semantic_features=str(sem_path), output=str(root/f'{dataset}_run'),
                     device='cpu', epochs=1, warmup_epochs=0)
            cfg = root/f'{dataset}.json'; cfg.write_text(json.dumps(c))
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            for args in [('check','--config',str(cfg)), ('train','--config',str(cfg)),
                         ('test','--run-dir',c['output'])]:
                subprocess.run([sys.executable,str(ROOT/'run.py'),*args],check=True,env=env)
            result = json.loads((Path(c['output'])/'results/test.json').read_text())
            assert all(np.isfinite(v) for v in result.values()), result
            assert (Path(c['output'])/'checkpoints/best_Loss_model.ckpt.sha256').is_file()
            print('SMOKE PASS:',dataset,flush=True)

if __name__ == '__main__': main()
