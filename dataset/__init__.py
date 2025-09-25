from .arctic import ArcticDataset
from .arctic_rigid import RigidArcticDataset
from .dexycb import DexYCBDataset
from .wild import WildDataset
from .ho3d_seg import HO3DDataset

def load_dataset(cfg, split='train', test_split='SDF', multi_batch=False):
    if 'wild' in cfg.name:
        return WildDataset(cfg, split)
    dataset_dict = {
        'ho3d' : HO3DDataset,
        'arctic': ArcticDataset,
        'arctic_rigid': RigidArcticDataset,
        'dexycb': DexYCBDataset,
    }
    if multi_batch:
        return dataset_dict[cfg.name](cfg, split, test_split, multi_batch)
    else:
        return dataset_dict[cfg.name](cfg, split, test_split)
