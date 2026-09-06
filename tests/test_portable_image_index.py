import gzip
import json
from pathlib import Path

import pytest

from experiments.imagenet100.data import ImageFolderSplits, _load_portable_index


def fixture_manifest(tmp_path):
    roots = (tmp_path/'train.X1', tmp_path/'val.X')
    for root in roots:
        (root/'n1').mkdir(parents=True)
        (root/'n1'/'a.jpg').touch()
    splits = ImageFolderSplits((roots[0],),roots[1],('n1',))
    payload = dict(version=1,classes=['n1'],roots=['train.X1','val.X'],
                   train=[[0,'n1/a.jpg',0]],val=[[1,'n1/a.jpg',0]])
    return splits,payload


def test_portable_manifest_relocates(tmp_path):
    splits,payload=fixture_manifest(tmp_path)
    path=tmp_path/'index.gz'
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    train,val=_load_portable_index(splits,path)
    assert train == [(str(splits.train[0]/'n1'/'a.jpg'),0)]
    assert val == [(str(splits.val/'n1'/'a.jpg'),0)]


@pytest.mark.parametrize('relative', ['../a.jpg','/n1/a.jpg','n2/a.jpg','n1/../../a.jpg'])
def test_portable_manifest_rejects_bad_paths(tmp_path,relative):
    splits,payload=fixture_manifest(tmp_path)
    payload['train'][0][1]=relative
    path=tmp_path/'index.gz'
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError):
        _load_portable_index(splits,path)


def test_portable_manifest_rejects_missing_sample(tmp_path):
    splits,payload=fixture_manifest(tmp_path)
    (splits.val/'n1'/'a.jpg').unlink()
    path=tmp_path/'index.gz'
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError):
        _load_portable_index(splits,path)


def test_loader_uses_bundle_without_scanning(tmp_path, monkeypatch):
    import experiments.imagenet100.data as data
    splits,payload=fixture_manifest(tmp_path)
    expected=([(str(splits.train[0]/'n1/a.jpg'),0)],
              [(str(splits.val/'n1/a.jpg'),0)])
    monkeypatch.setattr(data,'_load_portable_index',lambda *args: expected)
    def fail(*args):
        raise AssertionError('bundled index must avoid scanning')
    monkeypatch.setattr(data,'index_imagefolder_samples',fail)
    train,val,hit,path=data.load_or_index_imagefolder_samples(splits,cache_dir=tmp_path/'cache')
    assert (train,val)==expected and hit
