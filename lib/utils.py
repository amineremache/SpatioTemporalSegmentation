import json
import logging
import os
import errno
import time

import numpy as np
import torch

from lib.pc_utils import colorize_pointcloud, save_point_cloud

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F


def load_state_with_same_shape(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights


def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None):
  mkdir_p(config.log_dir)
  if config.overwrite_weights:
    if postfix is not None:
      filename = f"checkpoint_{config.wrapper_type}{config.model}{postfix}.pth"
    else:
      filename = f"checkpoint_{config.wrapper_type}{config.model}.pth"
  else:
    filename = f"checkpoint_{config.wrapper_type}{config.model}_iter_{iteration}.pth"
  checkpoint_file = config.log_dir + '/' + filename
  state = {
      'iteration': iteration,
      'epoch': epoch,
      'arch': config.model,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
  }
  if best_val is not None:
    state['best_val'] = best_val
    state['best_val_iter'] = best_val_iter
  json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
  torch.save(state, checkpoint_file)
  logging.info(f"Checkpoint saved to {checkpoint_file}")
  # Delete symlink if it exists
  if os.path.exists(f'{config.log_dir}/weights.pth'):
    os.remove(f'{config.log_dir}/weights.pth')
  # Create symlink
  os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def precision_at_one(pred, target, ignore_label=255):
  """Computes the precision@k for the specified values of k"""
  # batch_size = target.size(0) * target.size(1) * target.size(2)
  pred = pred.view(1, -1)
  target = target.view(1, -1)
  correct = pred.eq(target)
  correct = correct[target != ignore_label]
  correct = correct.view(-1)
  if correct.nelement():
    return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
  else:
    return float('nan')


def fast_hist(pred, label, n):
  k = (label >= 0) & (label < n)
  return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class WithTimer(object):
  """Timer for with statement."""

  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    out_str = 'Elapsed: %s' % (time.time() - self.tstart)
    if self.name:
      logging.info('[{self.name}]')
    logging.info(out_str)


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.averate_time = 0

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff


class ExpTimer(Timer):
  """ Exponential Moving Average Timer """

  def __init__(self, alpha=0.5):
    super(ExpTimer, self).__init__()
    self.alpha = alpha

  def toc(self):
    self.diff = time.time() - self.start_time
    self.average_time = self.alpha * self.diff + \
        (1 - self.alpha) * self.average_time
    return self.average_time


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def debug_on():
  import sys
  import pdb
  import functools
  import traceback

  def decorator(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      try:
        return f(*args, **kwargs)
      except Exception:
        info = sys.exc_info()
        traceback.print_exception(*info)
        pdb.post_mortem(info[2])

    return wrapper

  return decorator


def get_prediction(dataset, output, target):
  return output.max(1)[1]


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
  return torch.device('cuda' if is_cuda else 'cpu')


class HashTimeBatch(object):

  def __init__(self, prime=5279):
    self.prime = prime

  def __call__(self, time, batch):
    return self.hash(time, batch)

  def hash(self, time, batch):
    return self.prime * batch + time

  def dehash(self, key):
    time = key % self.prime
    batch = key / self.prime
    return time, batch


def save_rotation_pred(iteration, pred, dataset, save_pred_dir):
  """Save prediction results in original pointcloud scale."""
  decode_label_map = {}
  for k, v in dataset.label_map.items():
    decode_label_map[v] = k
  pred = np.array([decode_label_map[x] for x in pred], dtype=np.int)
  out_rotation_txt = dataset.get_output_id(iteration) + '.txt'
  out_rotation_path = save_pred_dir + '/' + out_rotation_txt
  np.savetxt(out_rotation_path, pred, fmt='%i')


def save_predictions(coords, upsampled_pred, transformation, dataset, config, iteration,
                     save_pred_dir):
  """Save prediction results in original pointcloud scale."""
  from lib.dataset import OnlineVoxelizationDatasetBase
  if dataset.IS_ONLINE_VOXELIZATION:
    assert transformation is not None, 'Need transformation matrix.'
  iter_size = coords[:, -1].max() + 1  # Normally batch_size, may be smaller at the end.
  if dataset.IS_TEMPORAL:  # Iterate over temporal dilation.
    iter_size *= config.temporal_numseq
  for i in range(iter_size):
    # Get current pointcloud filtering mask.
    if dataset.IS_TEMPORAL:
      j = i % config.temporal_numseq
      i = i // config.temporal_numseq
    batch_mask = coords[:, -1].numpy() == i
    if dataset.IS_TEMPORAL:
      batch_mask = np.logical_and(batch_mask, coords[:, -2].numpy() == j)
    # Calculate original coordinates.
    coords_original = coords[:, :3].numpy()[batch_mask] + 0.5
    if dataset.IS_ONLINE_VOXELIZATION:
      # Undo voxelizer transformation.
      curr_transformation = transformation[i, :16].numpy().reshape(4, 4)
      xyz = np.hstack((coords_original, np.ones((batch_mask.sum(), 1))))
      orig_coords = (np.linalg.inv(curr_transformation) @ xyz.T).T
    else:
      orig_coords = coords_original
    orig_pred = upsampled_pred[batch_mask]
    # Undo ignore label masking to fit original dataset label.
    if dataset.IGNORE_LABELS is not None:
      if isinstance(dataset, OnlineVoxelizationDatasetBase):
        label2masked = dataset.label2masked
        maskedmax = label2masked[label2masked < 255].max() + 1
        masked2label = [label2masked.tolist().index(i) for i in range(maskedmax)]
        orig_pred = np.take(masked2label, orig_pred)
      else:
        decode_label_map = {}
        for k, v in dataset.label_map.items():
          decode_label_map[v] = k
        orig_pred = np.array([decode_label_map[x] for x in orig_pred], dtype=np.int)
    # Determine full path of the destination.
    full_pred = np.hstack((orig_coords[:, :3], np.expand_dims(orig_pred, 1)))
    filename = 'pred_%04d_%02d.npy' % (iteration, i)
    if dataset.IS_TEMPORAL:
      filename = 'pred_%04d_%02d_%02d.npy' % (iteration, i, j)
    # Save final prediction as npy format.
    np.save(os.path.join(save_pred_dir, filename), full_pred)


def visualize_results(coords, input, target, upsampled_pred, config, iteration):
  # Get filter for valid predictions in the first batch.
  target_batch = coords[:, 3].numpy() == 0
  input_xyz = coords[:, :3].numpy()
  target_valid = target.numpy() != 255
  target_pred = np.logical_and(target_batch, target_valid)
  target_nonpred = np.logical_and(target_batch, ~target_valid)
  ptc_nonpred = np.hstack((input_xyz[target_nonpred], np.zeros((np.sum(target_nonpred), 3))))
  # Unwrap file index if tested with rotation.
  file_iter = iteration
  if config.test_rotation >= 1:
    file_iter = iteration // config.test_rotation
  # Create directory to save visualization results.
  os.makedirs(config.visualize_path, exist_ok=True)
  # Label visualization in RGB.
  xyzlabel = colorize_pointcloud(input_xyz[target_pred], upsampled_pred[target_pred])
  xyzlabel = np.vstack((xyzlabel, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'pred', '%04d.ply' % file_iter])
  save_point_cloud(xyzlabel, os.path.join(config.visualize_path, filename), verbose=False)
  # RGB input values visualization.
  xyzrgb = np.hstack((input_xyz[target_batch], input[:, :3].cpu().numpy()[target_batch]))
  filename = '_'.join([config.dataset, config.model, 'rgb', '%04d.ply' % file_iter])
  save_point_cloud(xyzrgb, os.path.join(config.visualize_path, filename), verbose=False)
  # Ground-truth visualization in RGB.
  xyzgt = colorize_pointcloud(input_xyz[target_pred], target.numpy()[target_pred])
  xyzgt = np.vstack((xyzgt, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'gt', '%04d.ply' % file_iter])
  save_point_cloud(xyzgt, os.path.join(config.visualize_path, filename), verbose=False)


def permute_pointcloud(input_coords, pointcloud, transformation, label_map,
                       voxel_output, voxel_pred):
  """Get permutation from pointcloud to input voxel coords."""
  def _hash_coords(coords, coords_min, coords_dim):
    return np.ravel_multi_index((coords - coords_min).T, coords_dim)
  # Validate input.
  input_batch_size = input_coords[:, -1].max().item()
  pointcloud_batch_size = pointcloud[:, -1].max().int().item()
  transformation_batch_size = transformation[:, -1].max().int().item()
  assert input_batch_size == pointcloud_batch_size == transformation_batch_size
  pointcloud_permutation, pointcloud_target = [], []
  # Process each batch.
  for i in range(input_batch_size + 1):
    # Filter batch from the data.
    input_coords_mask_b = input_coords[:, -1] == i
    input_coords_b = (input_coords[input_coords_mask_b])[:, :-1].numpy()
    pointcloud_b = pointcloud[pointcloud[:, -1] == i, :-1].numpy()
    transformation_b = transformation[i, :-1].reshape(4, 4).numpy()
    # Transform original pointcloud to voxel space.
    original_coords1 = np.hstack((pointcloud_b[:, :3], np.ones((pointcloud_b.shape[0], 1))))
    original_vcoords = np.floor(original_coords1 @ transformation_b.T)[:, :3].astype(int)
    # Hash input and voxel coordinates to flat coordinate.
    vcoords_all = np.vstack((input_coords_b, original_vcoords))
    vcoords_min = vcoords_all.min(0)
    vcoords_dims = vcoords_all.max(0) - vcoords_all.min(0) + 1
    input_coords_key = _hash_coords(input_coords_b, vcoords_min, vcoords_dims)
    original_vcoords_key = _hash_coords(original_vcoords, vcoords_min, vcoords_dims)
    # Query voxel predictions from original pointcloud.
    key_to_idx = dict(zip(input_coords_key, range(len(input_coords_key))))
    pointcloud_permutation.append(
        np.array([key_to_idx.get(i, -1) for i in original_vcoords_key]))
    pointcloud_target.append(pointcloud_b[:, -1].astype(int))
  pointcloud_permutation = np.concatenate(pointcloud_permutation)
  # Prepare pointcloud permutation array.
  pointcloud_permutation = torch.from_numpy(pointcloud_permutation)
  permutation_mask = pointcloud_permutation >= 0
  permutation_valid = pointcloud_permutation[permutation_mask]
  # Permuate voxel output to pointcloud.
  pointcloud_output = torch.zeros(pointcloud.shape[0], voxel_output.shape[1]).to(voxel_output)
  pointcloud_output[permutation_mask] = voxel_output[permutation_valid]
  # Permuate voxel prediction to pointcloud.
  # NOTE: Invalid points (points found in pointcloud but not in the voxel) are mapped to 0.
  pointcloud_pred = torch.ones(pointcloud.shape[0]).int().to(voxel_pred) * 0
  pointcloud_pred[permutation_mask] = voxel_pred[permutation_valid]
  # Map pointcloud target to respect dataset IGNORE_LABELS
  pointcloud_target = torch.from_numpy(
      np.array([label_map[i] for i in np.concatenate(pointcloud_target)])).int()
  return pointcloud_output, pointcloud_pred, pointcloud_target




def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.
    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]

    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    print("before one_hot")
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    print("shape of one_hot : ",target_one_hot.shape)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)