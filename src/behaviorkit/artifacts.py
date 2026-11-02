"""Approved observed digests. This catalog is not a publisher signature."""
from pathlib import Path
from .storage import sha256

APPROVED = {
    'rtmdet-tiny': {
        'checkpoint_sha256': '78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb',
        'url': 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
        'license': 'Apache-2.0', 'contract': 'rtmdet_decoded', 'recipe': 'scripts/prepare_models.py',
    },
    'd-fine-n': {
        'checkpoint_sha256': '41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c',
        'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth',
        'license': 'Apache-2.0', 'contract': 'dfine_deploy', 'recipe': 'scripts/prepare_models.py',
        'source_revision': '956d1709314c2c6a4df6f34de232054578a7449f',
    },
    'rf-detr-nano': {
        'checkpoint_sha256': 'd8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d',
        'url': 'https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth',
        'license': 'Apache-2.0', 'contract': 'rfdetr_raw', 'recipe': 'rfdetr==1.11.2',
        'qualification': 'independent exporter parity pending',
    },
}


def verify_checkpoint(name, path):
    if name not in APPROVED:
        raise ValueError('checkpoint is not in the approved artifact catalog')
    path = Path(path)
    if not path.is_file() or sha256(path) != APPROVED[name]['checkpoint_sha256']:
        raise ValueError(f'checkpoint digest mismatch for {name}; refused before loading')
    return path
