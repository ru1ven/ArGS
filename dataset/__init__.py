from .wild import WILDDataset
from .arctic import ArcticDataset
from .arctic_rigid import RigidArcticDataset


def load_dataset(cfg, split='train', test_split='SDF', multi_batch=False):
    
    dataset_dict = {
        
        'arctic': ArcticDataset,
        'arctic_rigid': RigidArcticDataset,
        'wild': WILDDataset
        
    }
    if multi_batch:
        return dataset_dict[cfg.name](cfg, split, test_split, multi_batch)
    else:
        return dataset_dict[cfg.name](cfg, split, test_split)
