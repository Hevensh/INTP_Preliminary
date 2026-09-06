"""Convert a real Kaggle index to a small, relocatable Git manifest."""
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    sig = source['signature']
    roots = [PurePosixPath(p) for p in [*sig['train'], sig['val']]]
    result = dict(version=1, dataset='ambityga/imagenet100',
                  source='xiongwutao/intp-img-littletest version 18',
                  classes=sig['classes'], roots=[p.name for p in roots])
    for split in ('train', 'val'):
        rows = []
        for filename, label in source[split]:
            path = PurePosixPath(filename)
            matches = [(i,path.relative_to(root).as_posix()) for i,root in enumerate(roots)
                       if path.is_relative_to(root)]
            if len(matches) != 1:
                raise ValueError(filename)
            i,relative = matches[0]
            rows.append([i,relative,label])
        result[split] = rows
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(result,separators=(',',':')).encode(),mtime=0))
    print(len(result['train']),len(result['val']),args.output.stat().st_size)


if __name__ == '__main__':
    main()
