from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .plantstereo_dataset import PlantStereoDataset
from .tsukuba_dataset import TsukubaDataset
from .instereo2k_dataset import Instereo2kDataset
from .middlebury14_dataset import MiddleburyDataset
from .eth3d_dataset import ETH3DDataset
from .apolloscape_dataset import ASDataset


__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "plantstereo": PlantStereoDataset,
    "tsukuba": TsukubaDataset,
    "instereo2k": Instereo2kDataset,
    "middlebury14": MiddleburyDataset,
    "eth3d": ETH3DDataset,
    "apollo": ASDataset
}

"""
--------------datapath--------------
sceneflow: '/media/wangqingyu/固态硬盘/SceneFlow'
kitti: '/home/wangqingyu/KITTI/201215'
middlebury14: '/media/wangqingyu/机械硬盘2/立体匹配公开数据集/01_middlebury数据集/middlebury2014'
plantstereo: '/home/wangqingyu/PlantStereo/PlantStereo2021'
tsukuba: '/media/wangqingyu/机械硬盘2/立体匹配公开数据集/06_NewTsukuba数据集'
instereo2k: '/media/wangqingyu/机械硬盘2/立体匹配公开数据集/05_InStereo2K数据集'
eth3d: '/media/wangqingyu/机械硬盘2/立体匹配公开数据集/04_Eth3D数据集'
apollo: '/home/wangqingyu/KITTI/201215'
"""
