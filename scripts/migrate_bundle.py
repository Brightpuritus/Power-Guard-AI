"""Run from any directory after extracting the old data/model ZIP at project root."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]

def main():
    pairs=[('data.csv','data/raw/data.csv'),('cleaned','data/processed/cleaned'),
           ('eda_results','data/processed/eda_results'),('local_data','database/runtime')]
    pairs += [(p.name,'models/experiments/'+p.name) for p in ROOT.glob('results*') if p.is_dir()]
    pending=[]
    for old,new in pairs:
        src=(ROOT/old).resolve(); dst=(ROOT/new).resolve()
        if not src.is_relative_to(ROOT) or not dst.is_relative_to(ROOT):
            raise ValueError('Path is outside project')
        if src.exists():
            if dst.exists():raise FileExistsError(f'Destination already exists; no files moved: {dst}')
            pending.append((src,dst))
    for src,dst in pending:
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(src),str(dst))
        print(f'{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}')
    if not pending:print('No old data/model folders to migrate.')

if __name__=='__main__':main()
