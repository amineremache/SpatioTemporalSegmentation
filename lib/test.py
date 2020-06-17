import logging
import os
import shutil
import tempfile
import warnings
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device, save_predictions, visualize_results, \
    permute_pointcloud, save_rotation_pred

from MinkowskiEngine import SparseTensor

def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               has_gt=False,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    acc = hist.diagonal() / hist.sum(1) * 100
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
        "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
            loss=losses, top1=scores, mIOU=np.nanmean(ious),
            mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
    if class_names is not None:
      debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
    debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  if num_class == 2:
    num_class += 1
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, None)


def test(model, data_loader, config, transform_data_fn=None, has_gt=True, validation=None,epoch=None):
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  alpha, gamma, eps  = 1, 2, 1e-6 # Focal Loss parameters
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  if not config.is_train:
    checkpoint_fn = config.resume + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      model.load_state_dict(state['state_dict'])
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))
  if validation:
    logging.info('===> Start validating')
  else:
    logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  all_preds = []
  all_labels = []
  batch_losses = []

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.save_prediction or config.test_original_pointcloud:
    if config.save_prediction:
      save_pred_dir = config.save_pred_dir
      os.makedirs(save_pred_dir, exist_ok=True)
    else:
      save_pred_dir = tempfile.mkdtemp()
    if os.listdir(save_pred_dir):
      raise ValueError(f'Directory {save_pred_dir} not empty. '
                       'Please remove the existing prediction.')

  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      if config.return_transformation:
        coords, input, target, transformation = data_iter.next()
      else:
        coords, input, target = data_iter.next()
        transformation = None
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      if config.wrapper_type != 'None':
        color = input[:, :3].int()
      if config.normalize_color:
        input[:, :3] = input[:, :3] / 255. - 0.5
      sinput = SparseTensor(input, coords).to(device)

      # Feed forward
      inputs = (sinput,) if config.wrapper_type == 'None' else (sinput, coords, color)
      soutput = model(*inputs)
      output = soutput.F

      pred = get_prediction(dataset, output, target).int()
      iter_time = iter_timer.toc(False)

      all_preds.append(pred.cpu().detach().numpy())
      all_labels.append(target.cpu().detach().numpy())

      if config.save_prediction or config.test_original_pointcloud:
        save_predictions(coords, pred, transformation, dataset, config, iteration, save_pred_dir)

      if has_gt:
        if config.evaluate_original_pointcloud:
          raise NotImplementedError('pointcloud')
          output, pred, target = permute_pointcloud(coords, pointcloud, transformation,
                                                    dataset.label_map, output, pred)

        target_np = target.numpy()
        num_sample = target_np.shape[0]
        target = target.to(device)
        # focal loss
        input_soft = nn.functional.softmax(output, dim=1) + eps
        weight = torch.pow(-input_soft + 1., gamma)
        focal_loss = (-alpha * weight * torch.log(input_soft)).mean()
        with open(config.log_dir+"/val_loss_vox_{0}.txt".format(data_loader.dataset.VOXEL_SIZE),'a') as val_loss_file:
          val_loss_file.writelines('{0}, {1}\n'.format(focal_loss,epoch))
        val_loss_file.close()

        batch_losses.append(focal_loss)

        losses.update(float(focal_loss), num_sample)
        scores.update(precision_at_one(pred, target), num_sample)
        hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
        ious = per_class_iu(hist) * 100

        prob = torch.nn.functional.softmax(output, dim=1)
        ap = average_precision(prob.cpu().detach().numpy(), target_np)
        aps = np.vstack((aps, ap))
        # Due to heavy bias in class, there exists class with no test label at all
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          ap_class = np.nanmean(aps, 0) * 100.

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_labels)
        to_ignore = [i for i in range(len(targets)) if targets[i] == 255]
        preds_trunc = [preds[i] for i in range(len(preds)) if i not in to_ignore]
        targets_trunc = [targets[i] for i in range(len(targets)) if i not in to_ignore]
        cm = confusion_matrix(targets_trunc,preds_trunc,normalize='true')
        np.savetxt(config.log_dir+'/confusion_matrix_epoch_{0}_vox_{1}.txt'.format(epoch,data_loader.dataset.VOXEL_SIZE),cm)

        reordered_ious = dataset.reorder_result(ious)
        reordered_ap_class = dataset.reorder_result(ap_class)
        class_names = dataset.get_classnames()
        print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            has_gt,
            losses,
            scores,
            reordered_ious,
            hist,
            reordered_ap_class,
            class_names=class_names)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  reordered_ious = dataset.reorder_result(ious)
  reordered_ap_class = dataset.reorder_result(ap_class)
  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      data_time,
      iter_time,
      has_gt,
      losses,
      scores,
      reordered_ious,
      hist,
      reordered_ap_class,
      class_names=class_names)

  if not config.is_train:
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_labels)
    to_ignore = [i for i in range(len(targets)) if targets[i] == 255]
    preds_trunc = [preds[i] for i in range(len(preds)) if i not in to_ignore]
    targets_trunc = [targets[i] for i in range(len(targets)) if i not in to_ignore]
    cm = confusion_matrix(targets_trunc,preds_trunc,normalize='true')
    np.savetxt(config.log_dir+'/confusion_matrix_test_vox_{0}.txt'.format(data_loader.dataset.VOXEL_SIZE),cm)

  if config.test_original_pointcloud:
    logging.info('===> Start testing on original pointcloud space.')
    dataset.test_pointcloud(save_pred_dir)
  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  if validation:
    return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100, batch_losses

  else:
    return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100
