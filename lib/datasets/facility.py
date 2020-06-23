import logging
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import spatial
from plyfile import PlyData

from lib.utils import read_txt, fast_hist, per_class_iu
from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type, cache
import lib.transforms as t


class FacilityVoxelizationDatasetBase:
  CLIP_SIZE = None
  CLIP_BOUND = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  NUM_LABELS = 8
  IGNORE_LABELS = ()
  INTENSITY = False

  IS_FULL_POINTCLOUD_EVAL = True

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.TrainVal: 'trainval.txt',
      DatasetPhase.Test: 'test.txt'
  }

  def test_pointcloud(self, pred_dir):
    print('Running full pointcloud evaluation.')
    # Join subcloud by their area and subcloud id.
    subcloud_dict = defaultdict(list)
    for i, data_path in enumerate(self.data_paths):
      area, subcloud = data_path.split(os.sep)
      subcloud, _ = os.path.splitext(subcloud)
      subcloud_id = '_'.join(subcloud.split('_')[:-1])
      subcloud_dict[(area, subcloud_id)].append(i)
    # Test independently for each subcloud.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    pred_list = sorted(os.listdir(pred_dir))
    hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
    for subcloud_idx, subcloud_list in enumerate(subcloud_dict.values()):
      print(f'Evaluating subcloud {subcloud_idx} / {len(subcloud_dict)}.')
      # Join all predictions and query pointclouds of split data.
      pred = np.zeros((0, 4))
      pointcloud = np.zeros((0, 7)) # CHANGED FROM 7 TO 8 BECAUSE I ASSUME 7 WAS 6 XYZRGB AND 1 LABEL AND NOW I ADDED INTENSITY
      for i in subcloud_list:
        pred = np.vstack((pred, np.load(os.path.join(pred_dir, pred_list[i]))))
        pointcloud = np.vstack((pointcloud, self.load_ply(i,FacilityVoxelizationDatasetBase.INTENSITY)[0]))
      # Deduplicate all query pointclouds of split data.
      pointcloud = np.array(list(set(tuple(l) for l in pointcloud.tolist())))
      # Run test for each subcloud.
      pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
      _, result = pred_tree.query(pointcloud[:, :3])
      ptc_pred = pred[result, 3].astype(int)
      ptc_gt = pointcloud[:, -1].astype(int)
      if self.IGNORE_LABELS is not None:
        ptc_pred = self.label2masked[ptc_pred]
        ptc_gt = self.label2masked[ptc_gt]
      hist += fast_hist(ptc_pred, ptc_gt, self.NUM_LABELS)
      # Print results.
      ious = []
      print('Per class IoU:')
      for i, iou in enumerate(per_class_iu(hist) * 100):
        result_str = ''
        if hist.sum(1)[i]:
          result_str += f'{iou}'
          ious.append(iou)
        else:
          result_str += 'N/A'  # Do not print if data not in ground truth.
        print(result_str)
      print(f'Average IoU: {np.nanmean(ious)}')

  def _augment_coords_to_feats(self, coords, feats, labels=None):
    # Center x,y
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = coords - coords_center
    feats = np.concatenate((feats, norm_coords), 1)
    return coords, feats, labels


class FacilityDataset(FacilityVoxelizationDatasetBase, VoxelizationDataset):

  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 0.05 = 5cm

  CLIP_BOUND = None  # [-N, N]
  TEST_CLIP_BOUND = None

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = \
      ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

  AUGMENT_COORDS_TO_FEATS = True
  NUM_IN_CHANNEL = 6

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               augment_data=True,
               elastic_distortion=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_root = config.facility_path
    if isinstance(self.DATA_PATH_FILE[phase], (list, tuple)):
      data_paths = []
      for split in self.DATA_PATH_FILE[phase]:
        data_paths += read_txt(os.path.join('splits/facility', split))
    else:
      data_paths = read_txt(os.path.join('splits/facility', self.DATA_PATH_FILE[phase]))

    logging.info('Loading {} {}: {}'.format(self.__class__.__name__, phase,
                                            self.DATA_PATH_FILE[phase]))

    VoxelizationDataset.__init__(
        self,
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  @cache
  def load_ply(self, index, intensity=False):
    filepath = self.data_root / self.data_paths[index]
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    if not intensity:
      feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    else:
      feats = np.array([data['red'], data['green'], data['blue'], data['intensity']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    return coords, feats, labels, None


class FacilityArea5Dataset(FacilityDataset):

    DATA_PATH_FILE = {
        DatasetPhase.Train: ['area1.txt', 'area2.txt', 'area3.txt','area4.txt', 'area6.txt'],
        DatasetPhase.TrainVal: ['area1.txt', 'area2.txt', 'area3.txt', 'area4.txt', 'area6.txt'],
        DatasetPhase.Val: 'area5.txt',
        DatasetPhase.Test: 'area5.txt',
    }


def test(config, intensity=False):
  """Test point cloud data loader.
  """
  from torch.utils.data import DataLoader
  from lib.utils import Timer
  import open3d as o3d

  def make_pcd(coords, feats):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords[:, :3].float().numpy())
    pcd.colors = o3d.utility.Vector3dVector(feats[:, :3].numpy() / 255)
    if intensity:
      pcd.intensities = o3d.utility.Vector3dVector(feats[:, 3:3].numpy())
    return pcd

  timer = Timer()
  DatasetClass = FacilityArea5Dataset
  transformations = [
      t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
      t.ChromaticAutoContrast(),
      t.ChromaticTranslation(config.data_aug_color_trans_ratio),
      t.ChromaticJitter(config.data_aug_color_jitter_std),
  ]

  dataset = DatasetClass(
      config,
      prevoxel_transform=t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS),
      input_transform=t.Compose(transformations),
      augment_data=True,
      cache=True,
      elastic_distortion=True)

  data_loader = DataLoader(
      dataset=dataset,
      collate_fn=t.cfl_collate_fn_factory(limit_numpoints=False),
      batch_size=1,
      shuffle=True)

  # Start from index 1
  iter = data_loader.__iter__()
  for i in range(100):
    timer.tic()
    coords, feats, labels = iter.next()
    pcd = make_pcd(coords, feats)
    o3d.visualization.draw_geometries([pcd])
    print(timer.toc())


if __name__ == '__main__':
  from config import get_config
  config = get_config()

  test(config)
