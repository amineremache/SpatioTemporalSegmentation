import glob
import numpy as np
import os

from tqdm import tqdm

from lib.utils import mkdir_p
from lib.pc_utils import save_point_cloud

import MinkowskiEngine as ME

data = '9_classes'
FACILITY_IN_PATH = '/home/ubuntu/data/storengy/raw/{0}'.format(data)
FACILITY_OUT_PATH = '/home/ubuntu/data/storengy/ply/ply_{0}'.format(data)


class FacilityDatasetConverter:

  INTENSITY = False
  CLASSES = ['ACTUATOR', 'BOX', 'CABLE', 'FLOOR', 'GAUGE','PIPESUPPORT','PIPE', 'STRUCTURE', 'VALVE']
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      f.readline()
      pointcloud = [l.split(',') for l in f if l != '\n']
    # Load point cloud to named numpy array.
    pointcloud = np.array(pointcloud).astype(np.float32)
    if FacilityDatasetConverter.INTENSITY:
      assert pointcloud.shape[1] == 7
      xyz = pointcloud[:, :3].astype(np.float32)
      rgbi = pointcloud[:, 3:7].astype(np.uint8)
      #i = pointcloud[:, 6].astype(np.uint8)
      return xyz, rgbi
    else:
      xyz = pointcloud[:, :3].astype(np.float32)
      rgb = pointcloud[:, 3:6].astype(np.uint8)
      return xyz, rgb

  @classmethod
  def convert_to_ply(cls, root_path, out_path):
    """Convert FacilityDataset to PLY format that is compatible with
    Synthia dataset. Assumes file structure as given by the dataset.
    Outputs the processed PLY files to `FACILITY_OUT_PATH`.
    """

    txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
    for txtfile in tqdm(txtfiles):
      file_sp = os.path.normpath(txtfile).split(os.path.sep)
      target_path = os.path.join(out_path, file_sp[-3])
      out_file = os.path.join(target_path, file_sp[-2] + '.ply')

      if os.path.exists(out_file):
        print(out_file, ' exists')
        continue

      annotation, _ = os.path.split(txtfile)
      subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt')) 
      coords, feats, labels =  [], [], []
      for inst, subcloud in enumerate(subclouds):
        # Read ply file and parse its rgb values.
        xyz, rgbi = cls.read_txt(subcloud)
        _, annotation_subfile = os.path.split(subcloud)
        clsidx = cls.CLASSES.index(annotation_subfile.split('_')[0])

        coords.append(xyz)
        feats.append(rgbi)
        labels.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)

      if len(coords) == 0:
        print(txtfile, ' has 0 files.')
      else:
        # Concat
        coords = np.concatenate(coords, 0)
        feats = np.concatenate(feats, 0)
        labels = np.concatenate(labels, 0)
        inds, collabels = ME.utils.sparse_quantize(
            coords,
            feats,
            labels,
            return_index=True,
            ignore_label=255,
            quantization_size=0.0001 # 0.01 = 1cm
        )
        pointcloud = np.concatenate((coords[inds], feats[inds], collabels[:, None]), axis=1)

        # Write ply file.
        mkdir_p(target_path)
        save_point_cloud(pointcloud, out_file, with_label=True, verbose=True, intensity=FacilityDatasetConverter.INTENSITY)


def generate_splits(facility_out_path):
  """Takes preprocessed out path and generate txt files"""
  split_path = './splits/facility'
  mkdir_p(split_path)
  for i in range(1, 7):
    curr_path = os.path.join(facility_out_path, f'Area_{i}')
    files = glob.glob(os.path.join(curr_path, '*.ply'))
    files = [os.path.relpath(full_path, facility_out_path) for full_path in files]
    out_txt = os.path.join(split_path, f'area{i}.txt')
    with open(out_txt, 'w') as f:
      f.write('\n'.join(files))


if __name__ == '__main__':
  FacilityDatasetConverter.convert_to_ply(FACILITY_IN_PATH, FACILITY_OUT_PATH)
  generate_splits(FACILITY_OUT_PATH)
