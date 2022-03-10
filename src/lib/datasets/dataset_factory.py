from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.car import Car
from .dataset.fiveclass import Fiveclass
from .dataset.moreclass import Moreclass
from .dataset.person import Person
from .dataset.threeclass import Threeclass
from .dataset.twoclass import Twoclass
from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'moreclass': Moreclass,
  'car': Car,
  'threeclass': Threeclass,
  'twoclass': Twoclass,
  'person': Person,
  'fiveclass': Fiveclass
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
