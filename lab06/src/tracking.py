"""Keep the genuine MLflow FileStore usable when the lab folder is copied."""
from pathlib import Path
import yaml


def relocate_store(root):
    root = Path(root).resolve()
    if not root.exists():
        return
    for experiment in root.iterdir():
        if not experiment.is_dir() or not experiment.name.isdigit():
            continue
        meta = experiment / 'meta.yaml'
        if meta.exists():
            data = yaml.safe_load(meta.read_text(encoding='utf-8'))
            uri = experiment.as_uri()
            if data.get('artifact_location') != uri:
                data['artifact_location'] = uri
                meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        for run in experiment.iterdir():
            meta = run / 'meta.yaml'
            if run.is_dir() and meta.exists():
                data = yaml.safe_load(meta.read_text(encoding='utf-8'))
                uri = (run / 'artifacts').as_uri()
                if data.get('artifact_uri') != uri:
                    data['artifact_uri'] = uri
                    meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'results/mlruns'
    relocate_store(root)
    print('MLflow artifact locations now point to:', root)
