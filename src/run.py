import glob
import os
import sys
import traceback
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from scipy.optimize.zeros import results_c
import torch.nn as nn
import torch.optim
import torch.utils.data.sampler as sampler
import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from lr_schedule import LinearRampUpExponentialDecay

import dataset
import dataset_utils
import plateau_function as pf
import plots
import single_timestamps
import tsn_models
import update
from softmax_scores import get_softmax_scores_for_untrimmed_videos, split_task_outputs
from stat_meter import StatMeter
from tsn_transforms import *


def build_tsn_train_transform(model, model_name, mode):
    """
    Builds a Pytorch transform for training.

    This code comes from TSN authors': https://github.com/yjxiong/tsn-pytorch

    :param model: pytorch model
    :param model_name: model name, as string
    :param mode: modality (rgb or flow)
    :return: Pytorch transform
    """
    if mode != 'RGBDiff':
        normalize = GroupNormalize(model.input_mean, model.input_std)
    else:
        normalize = IdentityTransform()

    train_augmentation = model.get_augmentation()

    return torchvision.transforms.Compose([
        train_augmentation,
        Stack(roll=model_name == 'tsn_bni'),
        ToTorchFormatTensor(model_name != 'tsn_bni'),
        normalize])


def build_tsn_test_transform(model, model_name, mode, n_crops):
    """
    Builds a Pytorch transform for training.

    This code comes from TSN authors': https://github.com/yjxiong/tsn-pytorch

    :param model: Pytorch model
    :param model_name: model name
    :param mode: image modality (rgb or flow)
    :param n_crops: number of crops, must be either 1 (centre crop) or 10 (centre and corner crops, plus flipped images)
    :return: Pytorch transform
    """
    if mode != 'RGBDiff':
        normalize = GroupNormalize(model.input_mean, model.input_std)
    else:
        normalize = IdentityTransform()

    if n_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(model.scale_size),
            GroupCenterCrop(model.crop_size)
        ])
    elif n_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(model.input_size, model.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(n_crops))

    return torchvision.transforms.Compose([
        cropping,
        Stack(roll=model_name == 'tsn_bni'),
        ToTorchFormatTensor(div=model_name != 'tsn_bni'),
        normalize])


def load_annotations(settings):
    """
    Loads train and test annotations. Paths to annotations are specified inside the yaml file

    :param settings: yaml settings, as dictionary
    :return: (train_df, test_df, class_labels, class_map, video_lengths). Train_df and test_df are pandas data frames,
     whereas class_labels is an ordered dictionary whose keys are integer (class_index column in the csv) and values are
     the corresponding class descriptive string (class column in the csv). class_map is the opposite of class_labels,
     i.e. keys are strings and values are integers. video_lengths is dictionary whose keys are video ids and values are
     the respective number of frames
    """
    annotations_path = settings['annotation_path']
    suffix = '_{}'.format(settings['points_set']) if 'points_set' in settings else ''
    train_df_path = os.path.join(annotations_path, 'train_df{}.csv'.format(suffix))
    test_df_path = os.path.join(annotations_path, 'test_df.csv')

    print('-> Loading dataframe from {}'.format(train_df_path))

    (train_df, test_df, class_labels, class_map) = dataset_utils.parse_annotations(train_df_path, test_df_path)
    video_lengths_df = pd.read_csv(os.path.join(annotations_path, 'video_lengths.csv'))
    video_lengths = {}

    if settings['mode'] == 'flow':
        shift = 1
    else:
        shift = 0

    video_lengths_df.frames -= shift

    for row in video_lengths_df.itertuples():
        video_lengths[row.video] = int(row.frames)

    return train_df, test_df, class_labels, class_map, video_lengths


def load_data(train_df, test_df, class_labels, class_map, video_lengths, train_sampling_strategy, settings):
    """
    Sets Pytorch data loaders and datasets

    :param train_df: training dataframe
    :param test_df: test dataframe
    :param class_labels: ordered dictionary whose keys are integer and values are the corresponding class string
    :param class_map: the inverse of class_labels
    :param video_lengths: dictionary containing videos' lengths
    :param train_sampling_strategy: the strategy to be used to sample training frames.
     Should be either `random_tsn`, for gt baseline or `samples_from_plateaus`, for ts baseline
    :param settings: yaml settings, as dictionary
    :return: (untrimmed_data_loader, train_data_loader, test_data_loader, class_labels, plateaus_per_video)
    """
    class_list = list(class_labels.values())
    untrimmed_data_loader = set_untrimmed_data_loader(train_df, class_list, video_lengths, settings)
    plateaus_per_video = single_timestamps.initialise_plateaus_for_dataset(train_df, settings, video_lengths)

    (train_data_loader, test_data_loader) = set_data_loader(train_df, test_df, train_sampling_strategy, class_list,
                                                            video_lengths, settings)

    train_data_loader.dataset.set_class_labels(class_labels)
    train_data_loader.dataset.set_class_map(class_map)
    test_data_loader.dataset.set_class_labels(class_labels)
    test_data_loader.dataset.set_class_map(class_map)

    return untrimmed_data_loader, train_data_loader, test_data_loader, class_labels, plateaus_per_video


def set_untrimmed_data_loader(train_df, classes, video_lengths, settings):
    """
    Sets the data loader for the untrimmed training videos

    :param train_df: train dataframe
    :param classes: list of class indexes (integers)
    :param video_lengths: dictionary containing videos' lengths
    :param settings: yaml settings, as dictionary
    :return: untrimmed_data_loader
    """
    frames_path = settings['frames_path']
    model_name = settings['model_name']
    model = settings['model'].module
    mode = settings['mode']
    dataset_name = settings['dataset_name']
    batch_size = settings['training']['batch_size_untrimmed']
    num_workers = settings['training']['num_workers_untrimmed']
    stride = settings['fit']['softmax_stride']
    n_samples = 1  # must be always 1 regardless of mode
    frame1_indexed = True
    sampling_strategy = 'uniform'

    print('-> Loading data from {} with {} workers'.format(frames_path, num_workers))

    if model_name == 'tsn_bni':
        n_crops = 1  # just one crop for the softmax scores
        test_transforms = build_tsn_test_transform(model, model_name, mode, n_crops)
    else:
        raise Exception('Unrecognised model name: {}'.format(model_name))

    untrimmed_dataset = dataset.Dataset(dataset_name, train_df, classes, frames_path, n_samples, mode,
                                        sampling_strategy, model_name, video_lengths,
                                        softmax_stride=stride, transforms=test_transforms, untrimmed=True,
                                        stack_size=settings['of_stack_size'], frames_are1_indexed=frame1_indexed)

    untrimmed_data_loader = torch.utils.data.DataLoader(untrimmed_dataset, batch_size=batch_size, shuffle=False,
                                                        num_workers=num_workers, pin_memory=True)

    return untrimmed_data_loader


def set_data_loader(train_df, test_df, train_sampling_strategy, classes, video_lengths, settings):
    """
    Sets the data loaders for training and testing

    :param train_df: train dataframe
    :param test_df: test dataframe
    :param train_sampling_strategy: the strategy to be used to sample training frames.
     Should be either `random_tsn`, for gt baseline or `samples_from_plateaus`, for ts baseline
    :param classes: list of class indices (integers)
    :param video_lengths: dictionary of videos' lengths
    :param settings: yaml settings, as dictionary
    :return: (train_data_loader, test_data_loader)
    """
    frames_folder = settings['frames_path']
    model_name = settings['model_name']
    model = settings['model'].module
    mode = settings['mode']
    n_test_crops = settings['testing']['n_crops']
    dataset_name = settings['dataset_name']
    n_testing_samples = settings['testing']['n_samples']
    batch_size_train = settings['training']['batch_size']
    batch_size_test = settings['testing']['batch_size']
    num_workers_train = settings['training']['num_workers']
    num_workers_test = settings['testing']['num_workers']
    test_sampling_strategy = 'uniform'

    print('-> Loading data from {} with {} workers'.format(frames_folder, num_workers_train))

    frame1_indexed = True

    if model_name == 'tsn_bni':
        train_transforms = build_tsn_train_transform(model, model_name, mode)
        test_transforms = build_tsn_test_transform(model, model_name, mode, n_test_crops)
        n_training_samples = settings['training']['n_samples']
    else:
        raise Exception('Unrecognised model name: {}'.format(model_name))

    training_set = dataset.Dataset(dataset_name, train_df, classes, frames_folder, n_training_samples, mode,
                                   train_sampling_strategy, model_name, video_lengths,
                                   transforms=train_transforms, untrimmed=False, stack_size=settings['of_stack_size'],
                                   frames_are1_indexed=frame1_indexed)

    test_set = dataset.Dataset(dataset_name, test_df, classes, frames_folder, n_testing_samples, mode,
                               test_sampling_strategy, model_name, video_lengths,
                               transforms=test_transforms, untrimmed=False, stack_size=settings['of_stack_size'],
                               frames_are1_indexed=frame1_indexed)

    indexes = list(range(len(training_set)))
    settings['indexes'] = indexes
    random.seed()
    random.shuffle(indexes)
    batch_sampler = sampler.BatchSampler(indexes, batch_size=settings['training']['batch_size'], drop_last=False)
    settings['batch_sampler'] = batch_sampler

    train_data_loader = torch.utils.data.DataLoader(training_set, batch_sampler = batch_sampler,
                                                    num_workers=num_workers_train, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False,
                                                   num_workers=num_workers_test, pin_memory=True)

    return train_data_loader, test_data_loader


def setup_tsn_bn_inception(n_classes, num_segments, mode='flow', stack_size=5, checkpoint_state=None,
                           load_kinetics_weights=False):
    """
    Sets tsn model with batch normalisation and inception architecture

    :param n_classes: number of classes
    :param num_segments: number of segments
    :param mode: image modality, must be either `rgb` or `flow`
    :param stack_size: optical flow stack size
    :param checkpoint_state: optional checkpoint dictionary
    :param load_kinetics_weights: if True will load the kinetics pre-trained weights from folder `../models`
    :return: the tsn Pytorch model
    """
    if mode not in ['flow', 'rgb']:
        raise Exception('Modality not implemented yet'.format(mode))

    print('-> Setting up TSN BN Inception')
    print('=' * 80)

    modality = 'Flow' if mode == 'flow' else 'RGB'
    consensus = 'avg'
    base_model = 'BNInception'
    new_length = 1 if mode == 'rgb' else stack_size

    bn_inception_init_path = './models/bn_inception.pth'  # these are the weights for the base model

    tsn_bni = tsn_models.TSN(
        n_classes,
        num_segments,
        modality,
        base_model=base_model,
        new_length=new_length,
        consensus_type=consensus,
        before_softmax=True,
        dropout=0.7,
        partial_bn=True,
        init_state_path=bn_inception_init_path)

    if checkpoint_state is not None:
        state = checkpoint_state['state_dict'] if 'state_dict' in checkpoint_state else checkpoint_state
        tsn_bni.load_state_dict(state)
    elif load_kinetics_weights:
        kinetics_state_path = './models/kinetics_tsn_{}.pth'.format(mode)
        print('-> Loading kinetics {} weights from {}'.format(mode.upper(), kinetics_state_path))
        tsn_bni.load_state_dict(torch.load(kinetics_state_path), strict=False)

    tsn_bni.cuda()
    tsn_bni = nn.DataParallel(tsn_bni) 
    num_param = sum([param.numel() for param in tsn_bni.parameters() if param.requires_grad])

    print('-> TSN BN Inception is ready')
    print(f'-> The model has {(num_param/1e6):.2f}M parameters in total')
    print('=' * 80)

    return tsn_bni


def calculate_accuracy(output, target, top_k=(1,)):
    """
    Calculates top k accuracy. I copied this bit from somewhere but I don't remember the source :(

    :param output: classification output, should be 1xN array of scores, one for each of the N classes
    :param target: the ground truth class
    :param top_k: tuple specifying top k values (e.g. 1, 5, etc.)
    :return: (accuracy_k, correct_k, pred_k)
    """
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred_k = output.topk(max_k, 1, True, True)
    pred_k = pred_k.t()
    correct = pred_k.eq(target.view(1, -1).expand_as(pred_k))

    accuracy_k = []
    correct_k = []

    for k in top_k:
        ck = correct[:k].sum()
        pk = (ck / batch_size) * 100
        accuracy_k.append(pk)
        correct_k.append(ck)

    return accuracy_k, correct_k, pred_k


def calculate_accuracy_and_loss_for_action(action_output, action_label, criterion, test_results, results,
                                           action_info, action_index, predicted_all, gt_labels):
    """
    Calculates accuracy and loss for an action instance

    :param action_output: classification output
    :param action_label: ground truth label
    :param criterion: Pytorch criterion object
    :param losses: StatMeter object storing loss values
    :param top1: StatMeter object storing top 1 values
    :param top5: StatMeter object storing top 5 values
    :param results: list of dictionaries storing result output
    :param action_info: action's info dictionary, coming from the dataset
    :param action_index: index of the action in the dataset/data loader
    :param predicted_all: global predictions (all actions), for the confusion matrix
    :param gt_labels: global ground truth labels (all actions), for the confusion matrix
    :return: nothing
    """
    if settings['num_tasks'] == 2:
        output = split_task_outputs(action_output, settings['num_classes'])
        tasks = {
            task: {
                "output": output[task],
                "preds": output[task].topk(5, -1)[1],
                "labels": action_label[f"{task}_class"],
                "weight": 1,
            }
            for task in ["noun", "verb"]
        }
        n_tasks = len(tasks)
        loss = 0.
        corrects, predicts = [], []
        for task, d in tasks.items():
            task_loss = criterion(d["output"], d["labels"])
            loss += d["weight"] * task_loss
            test_results[f'{task}_loss'].update(d["weight"] * task_loss.item(), 1)

            (_, correct, predicted) = calculate_accuracy(d["output"], d["labels"], top_k=(1, 5))
            test_results[f"{task}_top1"].update(correct[0], 1)
            test_results[f"{task}_top5"].update(correct[1], 1)
            # adding global predictions for confusion matrix
            corrects.append([x.cpu().numpy() for x in correct])
            predicts.append(predicted.cpu().numpy())
            predicted_all[task].append(predicted.cpu().numpy()[0][0])
            gt_labels[task].append(d["labels"].cpu().numpy()[0])

        noun_preds = (tasks['noun']['preds'] == tasks['noun']['labels'].unsqueeze(-1))
        verb_preds = (tasks['verb']['preds'] == tasks['verb']['labels'].unsqueeze(-1))
        action_preds = noun_preds & verb_preds
        test_results["action_top1"].update(action_preds[:,0].sum().item(), 1)
        test_results["action_top5"].update(action_preds.sum().item(), 1)
        test_results["action_loss"].update(loss.item() / n_tasks, 1)

        info = {
            'start_frame': action_info['start_frame'][action_index],
            'stop_frame': action_info['stop_frame'][action_index],
            'video_id': action_info['video_id'][action_index],
            'noun_class': eval(action_info['class'][action_index])[0],
            'verb_class': eval(action_info['class'][action_index])[1],
            'noun_class_index': eval(action_info['class_index'][action_index])[0],
            'verb_class_index': eval(action_info['class_index'][action_index])[1],
            'frame_samples': action_info['frame_samples'][action_index].tolist(),
            'noun_correct': corrects[0][0],
            'verb_correct': corrects[1][0],
            'noun_predicted': predicts[0].transpose().tolist()[0],
            'verb_predicted': predicts[1].transpose().tolist()[0],
        }
    else:
        with torch.no_grad():
            loss = criterion(action_output, action_label)

            # measure accuracy and record loss
            (_, correct, predicted) = calculate_accuracy(action_output, action_label, top_k=(1, 5))
            test_results['losses'].update(loss.item(), 1)
            test_results['top1'].update(correct[0], 1)
            test_results['top5'].update(correct[1], 1)

            # adding global predictions for confusion matrix
            predicted_all.append(predicted.cpu().numpy()[0][0])
            gt_labels.append(action_label.cpu().numpy()[0])

            info = {
                'start_frame': action_info['start_frame'][action_index],
                'stop_frame': action_info['stop_frame'][action_index],
                'video_id': action_info['video_id'][action_index],
                'class': action_info['class'][action_index],
                'class_index': action_info['class_index'][action_index],
                'frame_samples': action_info['frame_samples'][action_index].tolist(),
                'correct': correct[0],
                'predicted': predicted.t().tolist()[0]
            }

    results.append(info)


def rank_and_select_frames(batch_indices, settings):
    """
    Rank frames according to their classification scores, for each class separately. Will also select the first top-h
    frames for each class

    :param batch_indices: list of lists, where the inner lists contain the shuffled batch indices
    :param settings: settings dictionary
    :return: the top-h frames, for each class, as a list of dictionaries containing classification info
    """
    train_data_loader = settings['train_data_loader']
    num_crops = 1  # always one for training
    n_frames = settings['training']['n_samples']
    n_classes = settings['n_classes']
    model = settings['model']
    h = settings['training']['h']

    assert h > 0 and h <= 1, 'cv percentage must be between 0 and 1!'
    assert num_crops == 1, 'num crops must be 1 here'

    bar = tqdm.tqdm(total=len(train_data_loader), desc='-> Ranking training frames...', file=sys.stdout)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_iteration, (input_tuple) in enumerate(train_data_loader):
            (input_batch, label_batch, videos_info) = input_tuple
            batch_size = input_batch.shape[0]

            input_var = input_batch.float().cuda(non_blocking=True)
            output_batch = model(input_var, do_consensus=False)

            for i in range(batch_size):
                label = label_batch[i]
                g_id = videos_info['g_id'][i]
                centre_points = videos_info['centre_points'][i]
                batch_index = batch_indices[batch_iteration][i]

                for n in range(n_frames):
                    p = output_batch[i, n, label]
                    predictions.append(
                        {
                            'g_id': g_id,
                            'label': label,
                            'prediction': p,
                            'point': centre_points[n],
                            'batch_index': batch_index
                        }
                    )

            bar.update()

    bar.close()
    selected_frames = []
    model.train()

    for c in range(n_classes):
        class_predictions = sorted([cp for cp in predictions if cp['label'] == c], key=lambda x: x['prediction'],
                                   reverse=True)
        n_train_samples_class = max(1, int(round(len(class_predictions) * h)))
        selected_frames.extend(class_predictions[:n_train_samples_class])

    return selected_frames


def train_loop(train_loader, model, criterion, optimizer, epoch, optimizer_step_frequency, print_freq=1):
    """
    Trains the model on the whole training set

    :param train_loader: Pytorch train data loader
    :param model: Pytorch model
    :param criterion: Pytorch criterion object
    :param optimizer: Pytorch optimizer object
    :param epoch: training epoch (integer)
    :param optimizer_step_frequency: how often we call optimizer.step() and optimizer.zero_grad(). This is to allow
     bigger batches to be run on small gpus
    :param print_freq: how often you want to print stuff
    :return: (losses, top1, top5), all of them as StatMeter objects
    """
    results = {}
    if settings['num_tasks'] == 2:
        results['noun_loss'] = StatMeter()
        results['noun_top1'] = StatMeter()
        results['noun_top5'] = StatMeter()
        results['verb_loss'] = StatMeter()
        results['verb_top1'] = StatMeter()
        results['verb_top5'] = StatMeter()
        results['action_top1'] = StatMeter()
        results['action_top5'] = StatMeter()
        results['action_loss'] = StatMeter()
    else:
        results['losses'] = StatMeter()
        results['top1'] = StatMeter()
        results['top5'] = StatMeter()
    n_iterations = len(train_loader)

    # switch to train mode
    model.train()

    optimizer.zero_grad()  # set gradients to zero before starting the training loop

    for batch_iteration, (input_tuple) in enumerate(train_loader):
        train_batch(input_tuple, criterion, model, results)

        did_optimization_step = False

        # this is to allow bigger batch to be run on smaller gpus
        if (batch_iteration + 1) % optimizer_step_frequency == 0 or (batch_iteration + 1) == n_iterations:
            optimizer.step()
            optimizer.zero_grad()
            did_optimization_step = True

        if batch_iteration % print_freq == 0:
            if settings['num_tasks'] == 2:
                    print('Training Epoch: [{0}][{1}/{2}]\t'
                    'Action_Loss {loss.val:.4f} ({loss.total:.4f})\t'
                    'Noun_Loss {noun_loss.val:.4f} ({noun_loss.total:.4f})\t'
                    'Verb_Loss {verb_loss.val:.4f} ({verb_loss.total:.4f})\t'
                    'Action_Acc@1 {action_top1.total:.3f}\t'
                    'Action_Acc@5 {action_top5.total:.3f}\t'
                    'Noun_Acc@1 {noun_top1.total:.3f}\t'
                    'Noun_Acc@5 {noun_top5.total:.3f}\t'
                    'Verb_Acc@1 {verb_top1.total:.3f}\t'
                    'Verb_Acc@5 {verb_top5.total:.3f}\t'.format(epoch+1, batch_iteration + 1, n_iterations, loss=results['action_loss'], noun_loss=results['noun_loss'], verb_loss=results['verb_loss'], action_top1=results['action_top1'], action_top5=results['action_top5'],
                    noun_top1=results['noun_top1'], noun_top5=results['noun_top5'], verb_top1=results['verb_top1'], verb_top5=results['verb_top5']))
            else:
                print('Training Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.total:.4f})\t'
                  'Acc@1 {top1.total:.3f}\t'
                  'Acc@5 {top5.total:.3f}'.format(epoch+1, batch_iteration + 1, n_iterations, loss=results['losses'], top1=results['top1'],
                                                  top5=results['top5']))

            if optimizer_step_frequency > 1 and did_optimization_step:
                print('Performed optimiser step and zero grad')

    if settings['training']['learning_rate_decay']:
        settings['lr_scheduler'].step()
    
    results['learning_rate'] = settings['optimizer'].param_groups[0]['lr']

    print('\n')
    print('-' * 80)
    if settings['num_tasks'] == 2:
        print('--> Total Training: Action_Acc@1 {action_top1.total:.3f}\tAction_Acc@5 {action_top5.total:.3f}\tNoun_Acc@1 {noun_top1.total:.3f}\tNoun_Acc@5 {noun_top5.total:.3f}\tVerb_Acc@1 {verb_top1.total:.3f}\tVerb_Acc@5 {verb_top5.total:.3f}\t'.format(action_top1=results['action_top1'], action_top5=results['action_top5'], noun_top1=results['noun_top1'], noun_top5=results['noun_top5'], verb_top1=results['verb_top1'], verb_top5=results['verb_top5']))
    else:
        print('--> Total Training: Acc@1 {top1.total:.3f}\tAcc@5 {top5.total:.3f}'.format(top1=results['top1'], top5=results['top5']))
    print('-' * 80)

    return results


def train_batch(input_tuple, criterion, model, results):
    """
    Trains a single mini-batch. This function performs loss.backwards BUT NOT optimizer.step() NEITHER
    optimizer.zero_grad()
 
    :param input_tuple: input tuple coming from the data loader
    :param criterion: Pytorch criterion object
    :param model: Pytorch model
    :param losses: StatMeter storing loss values
    :param top1: StatMeter storing top1 values
    :param top5: StatMeter storing top5 values
    :return:
    """
    (input, label, videos_info) = input_tuple

    input = input.float().cuda(non_blocking=True)
    if settings['num_tasks'] == 2:
        label = {k:v.long().cuda(non_blocking=True) for k,v in label.items()}
    else:
        label = label.long().cuda(non_blocking=True)
    
    output = model(input)

    if settings['num_tasks'] == 2:
        output = split_task_outputs(output, settings['num_classes'])
        tasks = {
            task: {
                "output": output[task],
                "preds": output[task].topk(5, -1)[1],
                "labels": label[f"{task}_class"],
                "weight": 1,
            }
            for task in ["noun", "verb"]
        }
        n_tasks = len(tasks)
        loss = 0.
        for task, d in tasks.items():
            task_loss = criterion(d["output"], d["labels"])
            loss += d["weight"] * task_loss
            results[f'{task}_loss'].update(d["weight"] * task_loss.item(), input.size(0))

            (_, correct, _) = calculate_accuracy(d["output"], d["labels"], top_k=(1, 5))
            results[f"{task}_top1"].update(correct[0], input.size(0))
            results[f"{task}_top5"].update(correct[1], input.size(0))

        loss.backward()
        with torch.no_grad():
            noun_preds = (tasks['noun']['preds'] == tasks['noun']['labels'].unsqueeze(-1))
            verb_preds = (tasks['verb']['preds'] == tasks['verb']['labels'].unsqueeze(-1))
            action_preds = noun_preds & verb_preds
            results["action_top1"].update(action_preds[:,0].sum().item(), input.size(0))
            results["action_top5"].update(action_preds.sum().item(), input.size(0))
            results["action_loss"].update(loss.item() / n_tasks, input.size(0))

    else:    
        loss = criterion(output, label)
        loss.backward()

        # measure accuracy and record loss
        with torch.no_grad():
            (_, correct, _) = calculate_accuracy(output, label, top_k=(1, 5))
            
            results['losses'].update(loss.item(), input.size(0))
            results['top1'].update(correct[0], input.size(0))
            results['top5'].update(correct[1], input.size(0))


def train_with_ts(settings, train_dict, softmax_scores, plateaus_per_video, epoch, update_plateaus):
    """
    Trains the model using single timestamp supervision

    :param settings: settings dictionary
    :param train_dict: training dictionary, containing loads of info which will be saved to disk
    :param softmax_scores: softmax scores of the untrimmed training videos
    :param plateaus_per_video: dictionary whose keys are video ids and values are list of plateau objects
    :param epoch: epoch number
    :param update_plateaus: if True will update the plateaus
    :return: nothing
    """
    model = settings['model']
    criterion = settings['criterion']
    optimizer = settings['optimizer']
    train_data_loader = settings['train_data_loader']
    export_path = settings['export_path']
    writer = settings['log_writer']
    optimizer_step_frequency = settings['optimizer_step_frequency']
    cl_frequency = settings['training']['cl_frequency']
    cl_training = settings['training']['cl_training']
    tdd = train_data_loader.dataset.data
    mode = settings['mode']
    stack_size = settings['of_stack_size']
    n_points = settings['training']['n_samples']
    all_plateaus = single_timestamps.get_all_plateaus_in_dataset(plateaus_per_video)

    indexes = settings['indexes']
    batch_sampler = settings['batch_sampler']
    batch_indexes = list(batch_sampler)
    selected_plateaus = None

    if epoch % cl_frequency == 0 or not train_data_loader.dataset.samples_from_plateaus or \
            update_plateaus or not cl_training:
        # first sample N points from each plateau
        sampled_points = single_timestamps.sample_points_from_plateaus(all_plateaus, mode, stack_size=stack_size,
                                                                       n_samples=n_points)
        # train_data_loader.batch_sampler = batch_sampler
        train_data_loader.dataset.set_plateaus_samples(sampled_points)

        if update_plateaus:
            total_updated = single_timestamps.update_plateaus_in_dataset(plateaus_per_video, softmax_scores, train_dict,
                                                                         settings)

            writer.add_scalar('train/updated_percentage', total_updated / len(indexes), epoch)

            # slowly increasing h as we update
            if total_updated > 0 and cl_training:
                new_h = settings['training']['h'] + settings['training']['h_step']
                new_h = min(1, new_h)

                if new_h < 1:
                    print('-> Increasing CL h from {} to {}'.format(settings['training']['h'], new_h))
                else:
                    print('-> Reached CL h=1, using all training samples')

                settings['training']['h'] = new_h

            writer.add_scalar('train/h', settings['training']['h'], epoch)

        if cl_training and settings['training']['h'] < 1:
            # ranking and selecting the individual frames (scores are not averaged)
            selected_frames = rank_and_select_frames(batch_indexes, settings)
            selected_plateaus = set([sf['g_id'] for sf in selected_frames])
            selected_indexes = list(set([sf['batch_index'] for sf in selected_frames]))

            percentage_used_plateaus = len(selected_plateaus) / len(indexes)
            writer.add_scalar('train/used_plateau_percentage', percentage_used_plateaus, epoch)
            print('-> Using {:0.2f}% ({}/{}) training plateaus'.format(percentage_used_plateaus, len(selected_plateaus),
                                                                       len(indexes)))

            # now duplicate the frames for each selected plateau if they don't have exactly n_points each
            for sp in selected_plateaus:
                frames_in_plateau = [sf for sf in selected_frames if sf['g_id'] == sp]
                g = all_plateaus[sp]
                missing_points = n_points - len(frames_in_plateau)

                if missing_points > 0:
                    plateau_points = [sf['point'] for sf in frames_in_plateau]

                    assert all([p in sampled_points[g.video][g.id] for p in plateau_points]), \
                        'Where did that point come from?'

                    added_points = []

                    for m in range(missing_points):
                        added_points.append(random.choice(plateau_points))

                    plateau_points.extend(added_points)
                    sampled_points[g.video][g.id] = np.array(plateau_points, dtype=np.int32)

            # setting plateaus and corresponding points
            cl_batch_sampler = sampler.BatchSampler(selected_indexes, batch_size=settings['training']['batch_size'],
                                                    drop_last=False)
            train_data_loader.batch_sampler = cl_batch_sampler
            train_data_loader.dataset.set_plateaus_samples(sampled_points)
    else:
        if cl_training:
            selected_plateaus = [tdd[idx][0]['g_id'] for idx in indexes]

        sampled_points = train_data_loader.dataset.samples_from_plateaus  # using the previously sampled points

    single_timestamps.add_sampled_points_to_train_info(train_dict, sampled_points, n_points, all_plateaus)
    save_training_info(path_for_train_dicts(export_path), train_dict, epoch, used_for_training=selected_plateaus)

    train_results = train_loop(train_data_loader, model, criterion, optimizer, epoch,
                                                      optimizer_step_frequency)

    step = epoch
    write_tensorboard_accuracy_loss_logs(writer, 'train', train_results, step)
    save_epoch_info(settings, train_results, epoch, 'train')


def train_with_gt(settings, epoch):
    """
    Trains the model using full temporal supervision

    :param settings: settings dictionary
    :param epoch: epoch number
    :return: nothing
    """
    train_data_loader = settings['train_data_loader']
    model = settings['model']
    criterion = settings['criterion']
    optimizer = settings['optimizer']
    writer = settings['log_writer']
    optimizer_step_frequency = settings['optimizer_step_frequency']

    (loss_train, top1_train, top5_train) = train_loop(train_data_loader, model, criterion, optimizer, epoch,
                                                      optimizer_step_frequency)

    write_tensorboard_accuracy_loss_logs(writer, 'train', loss_train, top1_train, top5_train, epoch)
    save_epoch_info(settings, loss_train, top1_train, top5_train, epoch, 'train')


def test(epoch, settings, print_freq=1):
    """
    Tests the model

    :param epoch: epoch number
    :param settings: settings dictionary
    :param print_freq: how often you want to print stuff
    :return: nothing
    """
    writer = settings['log_writer']
    test_data_loader = settings['test_data_loader'] # this is used for testing
    export_path = settings['export_path']
    test_results = {}
    if settings['num_tasks'] == 2:
        test_results['noun_loss'] = StatMeter()
        test_results['noun_top1'] = StatMeter()
        test_results['noun_top5'] = StatMeter()
        test_results['verb_loss'] = StatMeter()
        test_results['verb_top1'] = StatMeter()
        test_results['verb_top5'] = StatMeter()
        test_results['action_top1'] = StatMeter()
        test_results['action_top5'] = StatMeter()
        test_results['action_loss'] = StatMeter()
    else:
        test_results['losses'] = StatMeter()
        test_results['top1'] = StatMeter()
        test_results['top5'] = StatMeter()
    results = []
    model = settings['model']
    val_loader = settings['test_data_loader']
    criterion = settings['criterion']

    # switch to evaluate mode
    model.eval()
    if settings['num_tasks'] == 2:
        gt_labels = {'noun':[], 'verb':[]}
        predicted_all = {'noun':[], 'verb':[]}
    else:
        gt_labels = []
        predicted_all = []
    model_name = val_loader.dataset.model_name
    with torch.no_grad():
        for i, (input, label, action_info) in enumerate(val_loader):
            input = input.float().cuda(non_blocking=True)
            if settings['num_tasks'] == 2:
                label = {k:v.long().cuda(non_blocking=True) for k,v in label.items()}
            else:
                label = label.long().cuda(non_blocking=True)

            if model_name == 'tsn_bni':
                action_label = label
                action_output = classify_action_with_tsn_bni(input, settings)
                calculate_accuracy_and_loss_for_action(action_output, action_label, criterion, test_results, results,
                                                    action_info, 0, predicted_all, gt_labels)
            else:
                raise Exception('Unrecognised model: {}'.format(val_loader.dataset.model_name))

            if i % print_freq == 0:
                if settings['num_tasks'] == 2:
                     print('Test: [{0}/{1}]\t'
                    'Action_Loss {loss.val:.4f} ({loss.total:.4f})\t'
                    'Noun_Loss {noun_loss.val:.4f} ({noun_loss.total:.4f})\t'
                    'Verb_Loss {verb_loss.val:.4f} ({verb_loss.total:.4f})\t'
                    'Action_Acc@1 {action_top1.total:.3f}\t'
                    'Action_Acc@5 {action_top5.total:.3f}\t'
                    'Noun_Acc@1 {noun_top1.total:.3f}\t'
                    'Noun_Acc@5 {noun_top5.total:.3f}\t'
                    'Verb_Acc@1 {verb_top1.total:.3f}\t'
                    'Verb_Acc@5 {verb_top5.total:.3f}\t'.format(i+1, len(val_loader), loss=test_results['action_loss'], noun_loss=test_results['noun_loss'], verb_loss=test_results['verb_loss'], action_top1=test_results['action_top1'], action_top5=test_results['action_top5'],
                    noun_top1=test_results['noun_top1'], noun_top5=test_results['noun_top5'], verb_top1=test_results['verb_top1'], verb_top5=test_results['verb_top5']))
                else:
                    print('Test: [{0}/{1}]\t'
                    'Loss {loss.total:.4f}\t'
                    'Acc@1 {top1.total:.3f}\t'
                    'Acc@5 {top5.total:.3f}'.format(i+1, len(val_loader), loss=test_results['losses'], top1=test_results['top1'], top5=test_results['top5']))

    print('\n')
    print('-' * 80)
    if settings['num_tasks'] == 2:
        print('--> Total Testing: Action_Acc@1 {action_top1.total:.3f}\tAction_Acc@5 {action_top5.total:.3f}\tNoun_Acc@1 {noun_top1.total:.3f}\tNoun_Acc@5 {noun_top5.total:.3f}\tVerb_Acc@1 {verb_top1.total:.3f}\tVerb_Acc@5 {verb_top5.total:.3f}\t'.format(action_top1=test_results['action_top1'], action_top5=test_results['action_top5'], noun_top1=test_results['noun_top1'], noun_top5=test_results['noun_top5'], verb_top1=test_results['verb_top1'], verb_top5=test_results['verb_top5']))
    else:
        print('--> Total Testing: Acc@1 {top1.total:.3f}\tAcc@5 {top5.total:.3f}'.format(top1=test_results['top1'], top5=test_results['top5']))
    print('-' * 80)

    if settings['num_tasks'] == 2:
        noun_cm = confusion_matrix(np.array(gt_labels['noun']), np.array(predicted_all['noun']))
        verb_cm = confusion_matrix(np.array(gt_labels['verb']), np.array(predicted_all['verb']))
        results = pd.DataFrame(results)
        save_epoch_info(settings, test_results, epoch, 'test')

        write_tensorboard_accuracy_loss_logs(writer, 'test', test_results, epoch)
        # noun_cm_figure = plots.plot_confusion_matrix(noun_cm, list(test_data_loader.dataset.class_labels.values()))
        # verb_cm_figure = plots.plot_confusion_matrix(verb_cm, list(test_data_loader.dataset.class_labels.values()))
        # noun_cm_img = plots.figure_to_img(noun_cm_figure)
        # verb_cm_img = plots.figure_to_img(verb_cm_figure)

        try:
            # writer.add_image('Noun_Confusion_Matrix', torch.from_numpy(noun_cm_img).permute(2, 0, 1), epoch)
            # writer.add_image('Verb_Confusion_Matrix', torch.from_numpy(verb_cm_img).permute(2, 0, 1), epoch)
            save_results(path_for_results(export_path), results, noun_cm, verb_cm, epoch)
        except Exception:
            traceback.print_exc()
            warnings.warn('Could not add confusion matrix image to tensorboard logs!')

    else:
        cm = confusion_matrix(np.array(gt_labels), np.array(predicted_all))
        results = pd.DataFrame(results)
        save_epoch_info(settings, test_results, epoch, 'test')

        write_tensorboard_accuracy_loss_logs(writer, 'test', test_results, epoch)
        cm_figure = plots.plot_confusion_matrix(cm, list(test_data_loader.dataset.class_labels.values()))
        cm_img = plots.figure_to_img(cm_figure)

        try:
            writer.add_image('Confusion_Matrix', torch.from_numpy(cm_img).permute(2, 0, 1), epoch)
            save_results(path_for_results(export_path), results, cm, epoch)
        except Exception:
            traceback.print_exc()
            warnings.warn('Could not add confusion matrix image to tensorboard logs!')

    print('-' * 80)


def classify_action_with_tsn_bni(input, settings, num_crops=None, n_samples=None, average_scores=True):
    """
    Test routine for tsn bni, for a single action

    :param input: Pytorch variable containing image values
    :param settings: dictionary settings
    :param num_crops: number of crops (should be either 1 or 10) for testing
    :param n_samples: number of samples to evaluate the model on
    :param average_scores: if True, will average the scores obtained from each sample to get a 1 x m array of scores,
     where m is the number of classes. If False with return a n_samples x m array of scores
    :return: the classification scores for the action
    """
    if input.shape[0] != 1:
        # in this case we feed one action per batch since we might be taking loads of crops in
        raise Exception('Please set batch size equal to 1!')

    mode = settings['mode'].lower()
    model = settings['model'].module
    of_stack_size = settings['of_stack_size']
    n_classes = settings['n_classes']

    if n_samples is None:
        n_samples = settings['testing']['n_samples']

    if num_crops is None:
        num_crops = settings['testing']['n_crops']

    if mode == 'rgb':
        length = 3
    elif mode == 'flow':
        length = of_stack_size * 2
    else:
        raise ValueError("Unknown modality " + mode)

    if num_crops == 10:
        input_var = input.view(-1, length, input.size(2), input.size(3))
    elif num_crops == 1:
        input_var = input
    else:
        raise Exception('Cannot deal with num_crops={}'.format(num_crops))

    action_output = model(input_var, do_consensus=False, override_reshape=False).data
    action_output = torch.mean(action_output.view((num_crops, n_samples, n_classes)), 0).view((n_samples, n_classes))

    if average_scores:
        # averaging the scores
        action_output = torch.mean(action_output, 0, keepdim=True).cuda()

    return action_output


def run(settings):
    """
    Runs the whole shebang

    :param settings: settings dictionary
    :return: nothing
    """
    untrimmed_data_loader = settings['untrimmed_data_loader']  # this is used to extract softmax scores
    test_data_loader = settings['test_data_loader']  # this is used for testing
    model = settings['model']
    start_epoch = settings['start_epoch']
    n_epochs = settings['n_epochs']
    baseline = settings['baseline']
    export_path = settings['export_path']
    writer = settings['log_writer']
    initial_plateaus = settings['initial_plateaus']
    save_scores = settings['save_softmax_scores'] if 'save_softmax_scores' in settings else True
    optimizer = settings['optimizer']

    untrimmed_dataset = untrimmed_data_loader.dataset
    train_df = untrimmed_dataset.data_frame

    if 'train_dict' in settings and settings['train_dict']:
        train_dict = settings['train_dict']
    else:
        train_dict = {p.video_id: {} for p in train_df.itertuples()}

    plateaus_per_video = {}

    for video, td in train_dict.items():
        if 'plateaus' not in td:
            td['plateaus'] = initial_plateaus[video]

        plateaus_per_video[video] = td['plateaus']

    epoch_info = {'test_df': test_data_loader.dataset.data_frame}

    optimizer_step_frequency = 1 if 'accumulate_for_batch_size' not in settings['training'] else \
        settings['training']['accumulate_for_batch_size'] / settings['training']['batch_size']

    if optimizer_step_frequency < 1 or (optimizer_step_frequency % 2 != 0 and optimizer_step_frequency != 1):
        raise Exception('Please adjust parameter accumulate_for_batch_size and/or batch size in order to '
                        'have optimizer step frequency multiple of 2')

    if optimizer_step_frequency > 1 and 'accumulate_for_batch_size' in settings['training']:
        print('-> Accumulating gradient for batch size {} with gpu mini-batch of size {}'.format(
            settings['training']['accumulate_for_batch_size'], settings['training']['batch_size']))

    settings['optimizer_step_frequency'] = optimizer_step_frequency

    if settings['scores_only']:
        print('# GETTING SCORES ONLY')
        (softmax_scores, logit_scores, _) = get_softmax_scores_for_untrimmed_videos(untrimmed_data_loader, model,
                                                                                    settings['mode'],
                                                                                    settings['of_stack_size'],
                                                                                    return_logits=True)
        save_video_scores(path_for_train_dicts(export_path), softmax_scores, logit_scores, start_epoch)
        writer.close()
        return

    if settings['test_only']:
        print('# TESTING ONLY')
        test(start_epoch, settings)
        writer.close()
        return

    for iteration, epoch in enumerate(range(start_epoch, n_epochs)):
        print('\n')
        print('=' * 80)
        print('-> Running epoch {}/{}'.format(epoch+1, n_epochs))
        print('=' * 80)
        print('\n')

        epoch_info['epoch'] = epoch

        for video, videoData in train_dict.items():
            if 'update_info' in videoData:
                del train_dict[video]['update_info']  # deleting previous update info (if any) to avoid duplicate data

        torch.cuda.empty_cache()

        do_update = is_baseline_with_update(baseline) and should_update(iteration, epoch, settings)

        if do_update:
            (softmax_scores, logit_scores, _) = get_softmax_scores_for_untrimmed_videos(untrimmed_data_loader, model, settings['mode'], settings['of_stack_size'])
            if save_scores:
                save_video_scores(path_for_train_dicts(export_path), softmax_scores, logit_scores, epoch)

            update_plateaus = True
        else:
            softmax_scores = {}
            update_plateaus = False

        torch.cuda.empty_cache()

        if is_gt_baseline(baseline):
            train_with_gt(settings, epoch)
        elif is_ts_baseline(baseline):
            train_with_ts(settings, train_dict, softmax_scores, plateaus_per_video, epoch, update_plateaus)
        else:
            raise Exception('Unrecognised baseline: {}'.format(baseline))

        if epoch % settings['export_frequency'] == 0 or epoch == n_epochs-1 or do_update:
            print('--> Saving checkpoint')

            torch.save({
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(path_for_models(export_path), 'model_{}.pt'.format(epoch)))

        if epoch % settings['testing']['frequency'] == 0 or epoch == n_epochs-1 or do_update:
            torch.cuda.empty_cache()
            test(epoch, settings)

    writer.close()


def should_update(iteration, epoch, settings):
    """
    Tells whether it is time to update the plateaus or not

    :param iteration: iteration number
    :param epoch: epoch number
    :param settings: settings dictionary
    :return: True if it is time for an update, and False otherwise
    """
    no_update = False if 'no_update' not in settings else settings['update']['no_update']

    if no_update:
        return False

    return epoch == settings['update']['start_epoch'] or \
           (epoch > settings['update']['start_epoch'] and epoch % settings['update']['frequency'] == 0) or \
           (settings['update_first_iteration'] and iteration == 0)


def update_has_started(epoch, settings):
    """
    Tells whether update has started or not

    :param epoch: epoch number
    :param settings: settings dictionary
    :return: True if the update has started, False otherwise
    """
    return is_baseline_with_update(settings['baseline']) and epoch >= settings['update']['start_epoch']


def save_epoch_info(settings, results, epoch, train_or_test):
    """
    Writes/updates a csv containing information for each epoch

    :param settings: dictionary settings
    :param loss: StatMeter loss
    :param top1_accuracy: StatMeter top1
    :param top5_accuracy: StatMeter top5
    :param epoch: epoch number
    :param train_or_test: either `train` or `test`
    :return: nothing
    """
    df_path = os.path.join(settings['export_path'], '{}_info.csv'.format(train_or_test))
    if settings['num_tasks'] == 2:
        epoch_info = {
            'epoch': epoch,
            'action_loss': results['action_loss'].total,
            'noun_loss': results['noun_loss'].total,
            'verb_loss': results['verb_loss'].total,
            'action_top1_accuracy': results['action_top1'].total,
            'action_top5_accuracy': results['action_top5'].total,
            'noun_top1_accuracy': results['noun_top1'].total,
            'noun_top5_accuracy': results['noun_top5'].total,
            'verb_top1_accuracy': results['verb_top1'].total,
            'verb_top5_accuracy': results['verb_top5'].total,
        }
    else:
        epoch_info = {
            'epoch': epoch,
            'loss': results['losses'].total,
            'top1_accuracy': results['top1'].total,
            'top5_accuracy': results['top5'].total
        }

    epoch_df = pd.DataFrame([epoch_info])

    if os.path.exists(df_path):
        if settings['num_tasks'] == 2:
            df = pd.read_csv(df_path, usecols=['epoch', 'action_loss', 'noun_loss', 'verb_loss', 'action_top1_accuracy', 'action_top5_accuracy', 'noun_top1_accuracy', 'noun_top5_accuracy', 'verb_top1_accuracy', 'verb_top5_accuracy'])
        else:
            df = pd.read_csv(df_path, usecols=['epoch', 'loss', 'top1_accuracy', 'top5_accuracy'])
        df = df.append(epoch_df, ignore_index=True)
    else:
        df = epoch_df

    df.to_csv(df_path, index=False)


def write_tensorboard_accuracy_loss_logs(writer, phase, results, epoch):
    """
    Writes tensorboard accuracy and loss logs

    :param writer: tensorboard writer
    :param phase: either `train` or `test`
    :param loss: StatMeter loss
    :param top1: StatMeter top1
    :param top5: StatMeter top5
    :param epoch: epoch number
    :return: nothing
    """
    if settings['num_tasks'] == 2:
        writer.add_scalar('{}/action_loss'.format(phase), results['action_loss'].total, epoch)
        writer.add_scalar('{}/noun_loss'.format(phase), results['noun_loss'].total, epoch)
        writer.add_scalar('{}/verb_loss'.format(phase), results['verb_loss'].total, epoch)
        writer.add_scalar('{}/Action_Acc@1'.format(phase), results['action_top1'].total, epoch)
        writer.add_scalar('{}/Action_Acc@5'.format(phase), results['action_top5'].total, epoch)
        writer.add_scalar('{}/Noun_Acc@1'.format(phase), results['noun_top1'].total, epoch)
        writer.add_scalar('{}/Noun_Acc@5'.format(phase), results['noun_top5'].total, epoch)
        writer.add_scalar('{}/Verb_Acc@1'.format(phase), results['verb_top1'].total, epoch)
        writer.add_scalar('{}/Verb_Acc@5'.format(phase), results['verb_top5'].total, epoch)
    else:
        writer.add_scalar('{}/loss'.format(phase), results['losses'].total, epoch)
        writer.add_scalar('{}/Acc@1'.format(phase), results['top1'].total, epoch)
        writer.add_scalar('{}/Acc@5'.format(phase), results['top5'].total, epoch)
        writer.add_scalar('{}/Verb_Acc@5'.format(phase), results['learning_rate'], epoch)
    if phase == 'train':
        writer.add_scalar('learning_rate', results['learning_rate'], epoch)


def save_results(results_export_path, results, noun_confusion_matrix, verb_confusion_matrix, epoch):
    """
    Saves a csv containing results information, as well as the confusion matrices as text files

    :param results_export_path: where to save the csv
    :param results: the results pandas dataframe
    :param confusion_matrix: the confusion matrix
    :param epoch: epoch number
    :return: nothing
    """
    cm_folder = os.path.join(results_export_path, 'confusion_matrices')

    if not os.path.exists(cm_folder):
        os.makedirs(cm_folder)

    csv_path = os.path.join(results_export_path, 'epoch_{}.csv'.format(epoch))
    results.to_csv(csv_path, index=False)
    cm_path = os.path.join(cm_folder, 'epoch_{}.csv'.format(epoch))
    np.savetxt(cm_path, noun_confusion_matrix, delimiter=',')
    np.savetxt(cm_path, verb_confusion_matrix, delimiter=',')


def save_training_info(train_dict_export_path, train_dict, epoch, used_for_training=None):
    """
    Save training information to disk

    :param train_dict_export_path: where to save the stuff
    :param train_dict: the training dictionary
    :param epoch: epoch number
    :param used_for_training: list of plateaus' ids used for training
    :return: nothing
    """
    summary = []

    with tqdm.tqdm(desc='-> Saving training dictionaries...', total=len(train_dict), file=sys.stdout) as bar:
        for video, video_data in train_dict.items():
            video_export_path = os.path.join(train_dict_export_path, 'epoch_{}'.format(epoch), video)

            if not os.path.exists(video_export_path):
                os.makedirs(video_export_path)

            plateaus = video_data['plateaus']
            pf.PlateauFunction.write_plateaus_to_file(plateaus,
                                                      os.path.join(video_export_path, video + '_plateaus.csv'))
            np.savetxt(os.path.join(video_export_path, video + '_points.csv'), video_data['sampled_points'],
                       delimiter=',', fmt='%d')

            if 'update_info' in video_data:
                update.write_update_info_to_file(video_data['update_info'],
                                                             os.path.join(video_export_path, video + '_update_info'))
                with_update = True
            else:
                with_update = False

            for g in plateaus:
                g_dict = pf.PlateauFunction.to_dict(g, add_history=False)
                _, gt_start, gt_end = single_timestamps.get_video_name_and_bounds_from_id(g.id)
                g_dict['gt_start'] = gt_start
                g_dict['gt_end'] = gt_end
                g_dict['updated'] = False
                g_dict['update_proposal'] = None
                g_dict['used_for_training'] = True if used_for_training is None else g.id in used_for_training

                if with_update:
                    if 'chosen_proposal' in video_data['update_info'][g.id]:
                        q = video_data['update_info'][g.id]['chosen_proposal']
                        g_dict['update_proposal'] = pf.PlateauFunction.params_to_string(q, add_confidence=True)

                    if 'updated' in video_data['update_info'][g.id]:
                        g_dict['updated'] = video_data['update_info'][g.id]['updated']

                summary.append(g_dict)

            bar.update()

    summary_path = os.path.join(train_dict_export_path, 'epoch_{}'.format(epoch), 'summary.csv')
    pd.DataFrame(summary).to_csv(summary_path, header=True, index=False, columns=[
        'video', 'index', 'id', 'label', 'c', 'w', 's', 'N', 'gt_start', 'gt_end',
        'used_for_training', 'updated', 'update_proposal'])


def save_video_scores(train_dict_export_path, softmax_scores, logit_scores, epoch, save_logits=False):
    """
    Save softmax and optionally logit scores to disk

    :param train_dict_export_path: where to save the stuff
    :param softmax_scores: dictionary whose keys are video ids and values are softmax scores
    :param logit_scores: dictionary whose keys are video ids and values are logit scores
    :param epoch: epoch number
    :param save_logits: if True will save logit scores
    :return: nothing
    """
    with tqdm.tqdm(desc='-> Saving softmax scores (saving logits:{})...'.format(save_logits),
                   total=len(softmax_scores), file=sys.stdout) as bar:
        for video, softScores in softmax_scores.items():
            video_export_path = os.path.join(train_dict_export_path, 'epoch_{}'.format(epoch), video)

            if not os.path.exists(video_export_path):
                os.makedirs(video_export_path)

            np.savetxt(os.path.join(video_export_path, video + '_softmax_scores.csv'), softScores, delimiter=',')

            if save_logits and video in logit_scores:
                np.savetxt(os.path.join(video_export_path, video + '_logit_scores.csv'), logit_scores[video],
                           delimiter=',')

            bar.update()


def find_best_model(export_path, train_or_test='train', value='loss', metric='min'):
    """
    Finds the model with min/max loss/accuracy

    :param export_path: where you exported everything
    :param train_or_test: either `train` or `test`, to specify the phase you are looking the best model for
    :param value: either `loss` or `top1_accuracy`
    :param metric: either `min` or `max`
    :return: (best_model_path, best_epoch)
    """
    info_df_path = os.path.join(export_path, '{}_info.csv'.format(train_or_test))
    models_path = os.path.join(export_path, 'models')
    exported_models_epochs = [os.path.basename(p) for p in glob.glob(os.path.join(models_path, 'model_*.pt'))]
    exported_models_epochs = sorted([int(p.replace('.pt', '').replace('model_', '')) for p in exported_models_epochs])

    df = pd.read_csv(info_df_path)
    df = df[df['epoch'].isin(exported_models_epochs)]

    if metric == 'min':
        best_epoch = df.loc[df[value].idxmin(), 'epoch']
        print('{} {} {}: {}'.format(metric, train_or_test, value, df[value].min()))
    elif metric == 'max':
        best_epoch = df.loc[df[value].idxmax(), 'epoch']
        print('{} {} {}: {}'.format(metric, train_or_test, value, df[value].max()))
    else:
        raise Exception('Metrics must be either min or max, found {}'.format(metric))

    best_model_path = os.path.join(export_path, 'models', 'model_{}.pt'.format(best_epoch))

    return best_model_path, best_epoch


def find_checkpoint(export_path, epoch=0):
    """
    Finds a checkpoint in a given folder

    :param export_path: where you saved the models (.pth files)
    :param epoch: specifies the epoch from which you want to load the model. If 0 will try to load the last model
    :return: (model_path, epoch), where epoch is the loaded model's epoch
    """
    if epoch == 0:
        models_list = glob.glob(os.path.join(export_path, '*.pt'))

        if len(models_list) == 0:
            return None, -1

        # the lambda thing is to do perform natural sorting
        model_path = sorted(models_list, key=lambda x: int(os.path.basename(x).replace('.pt', '').split('_')[1]))[-1]
        epoch = int(os.path.basename(model_path).split('_')[1].split('.pt')[0])
    else:
        model_path = os.path.join(export_path, 'model_{}.pt'.format(epoch))

    return model_path, epoch


def load_training_dicts(export_path, epoch):
    """
    Loads training dictionaries from disk

    :param export_path: where you saved the stuff
    :param epoch: epoch number
    :return: the training dictionary
    """
    epoch_path = os.path.join(export_path, 'epoch_{}'.format(epoch), '*')
    video_folders = [p for p in glob.glob(epoch_path) if os.path.isdir(p)]
    train_dict = {}

    if video_folders:
        print('-> Loading training dictionaries from {}'.format(epoch_path))

    for vf in video_folders:
        video = os.path.basename(vf)
        plateaus = pf.PlateauFunction.load_plateaus_from_file(os.path.join(vf, video + '_plateaus.csv'),
                                                              generate_xy=True)
        train_dict[video] = {'plateaus': plateaus}

    return train_dict


def get_training_sampling_strategy(settings):
    """
    :param settings: settings dictionary
    :return: `random_tsn` if running gt baseline, `samples_from_plateaus` otherwise
    """
    if is_baseline_with_random_sampling(settings['baseline']):
        return 'random_tsn'
    else:
        return 'samples_from_plateaus'


def path_for_models(export_path):
    return os.path.join(export_path, 'models')


def path_for_train_dicts(export_path):
    return os.path.join(export_path, 'training_dicts')


def path_for_logs(export_base_folder, run_id):
    return os.path.join(export_base_folder, 'logs', run_id)


def path_for_results(export_path):
    return os.path.join(export_path, 'results')


def is_baseline_with_random_sampling(baseline):
    return baseline == 'gt'


def is_baseline_with_update(baseline):
    return is_ts_baseline(baseline)


def is_gt_baseline(baseline):
    return baseline == 'gt'


def is_ts_baseline(baseline):
    return baseline == 'ts'


def setup(settings, tag=None):
    """
    Setups the whole shebang

    :param settings: settings dictionary
    :param tag: optional tag to be appended to the export path
    :return: nothing
    """
    print('=' * 80)
    print('-> Setting up...')

    if tag is not None and tag:
        tag = '_{}'.format(tag)
    else:
        tag = ''

    conf_name = os.path.basename(settings['settings_path']).split('.yaml')[0]

    os.environ['TORCH_MODEL_ZOO'] = './models'  # where to download pretrained models
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['gpus']

    run_id = os.path.join(settings['dataset_name'], conf_name + tag, settings['model_name'])
    export_path = os.path.join(settings['export_base_folder'], run_id)
    settings['export_path'] = export_path

    models_export_path = path_for_models(export_path)
    dicts_export_path = path_for_train_dicts(export_path)
    logs_export_path = path_for_logs(settings['export_base_folder'], run_id)
    results_export_path = path_for_results(export_path)

    if not os.path.exists(models_export_path):
        os.makedirs(models_export_path)

    if not os.path.exists(dicts_export_path):
        os.makedirs(dicts_export_path)

    if not os.path.exists(logs_export_path):
        os.makedirs(logs_export_path)

    if not os.path.exists(results_export_path):
        os.makedirs(results_export_path)

    yaml_output_file = os.path.join(export_path, os.path.basename(settings['settings_path']))

    with open(yaml_output_file, 'w') as yaml_file:
        yaml.dump(settings, yaml_file, default_flow_style=False)

    train_sampling_stragegy = get_training_sampling_strategy(settings)

    print('-> Training Sampling strategy: {}'.format(train_sampling_stragegy))

    if 'checkpoint_path' in settings:
        checkpoint_path = settings['checkpoint_path']
        checkpoint_epoch = int(os.path.basename(checkpoint_path).split('_')[1].split('.pt')[0])

        try:
            settings['train_dict'] = load_training_dicts(os.path.join(export_path, 'train_dicts'), checkpoint_epoch)
        except Exception:
            warnings.warn('Could not load training dict for epoch {} from {} ', checkpoint_epoch,
                          os.path.join(export_path, 'train_dicts'))
    else:
        if settings['load_checkpoint_epoch'] >= 0:
            (checkpoint_path, checkpoint_epoch) = find_checkpoint(os.path.join(export_path, 'models'),
                                                                  epoch=settings['load_checkpoint_epoch'])

            if checkpoint_path is not None:
                settings['train_dict'] = load_training_dicts(os.path.join(export_path, 'train_dicts'), checkpoint_epoch)
        else:
            checkpoint_path = None
            checkpoint_epoch = -1

    (train_df, test_df, class_labels, class_map, video_lengths) = load_annotations(settings)
    if isinstance(settings['num_classes'], dict):
        n_classes = sum([num for _, num in settings['num_classes'].items()])
    else:
        n_classes = max(len(class_labels), settings['num_classes'])

    if checkpoint_path is not None:
        print('-> Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = None

    if settings['model_name'] == 'tsn_bni':
        kinetics_init = settings['kinetics_init']
        model = setup_tsn_bn_inception(n_classes, settings['training']['n_samples'],
                                       mode=settings['mode'], stack_size=settings['of_stack_size'],
                                       checkpoint_state=checkpoint, load_kinetics_weights=kinetics_init)
        settings['n_frames_untrimmed_sampling'] = 1  # when mode is flow, this has to be one, otherwise can be any n > 0
    else:
        raise Exception('Unrecognised model: {}'.format(settings['model_name']))

    settings['model'] = model

    (untrimmed_data_loader, train_data_loader, test_data_loader, class_labels, initial_plateaus) = load_data(
        train_df,
        test_df,
        class_labels,
        class_map,
        video_lengths,
        train_sampling_stragegy,
        settings)

    # define loss function (criterion) and optimiser
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['training']['learning_rate'])
    if settings['training']['learning_rate_decay']:
        scheduler = LinearRampUpExponentialDecay(optimizer, 5, 70, 0.95, last_epoch=-1, verbose=False)
        settings['lr_scheduler'] = scheduler

    if checkpoint_path is not None and 'optimizer' in checkpoint:
        print('-> Loading optimizer state')
        optimizer.load_state_dict(checkpoint['optimizer'])

    settings['train_data_loader'] = train_data_loader
    settings['test_data_loader'] = test_data_loader
    settings['untrimmed_data_loader'] = untrimmed_data_loader
    settings['criterion'] = criterion
    settings['optimizer'] = optimizer
    settings['start_epoch'] = checkpoint_epoch + 1
    settings['log_writer'] = SummaryWriter(log_dir=logs_export_path)
    settings['initial_plateaus'] = initial_plateaus
    settings['n_classes'] = n_classes

    print('=' * 80)


if __name__ == '__main__':
    import yaml
    import matplotlib
    import argparse

    # to avoid crashes if there is no X server running. Note: call this before importing the pyplot module
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_path', nargs=1, type=str, help='path to yaml configuration file')
    parser.add_argument('--tag', nargs='?', type=str, default='', help='tag to be appended to the export folder')
    parser.add_argument('--test_only', action='store_true', help='only test the model')
    parser.add_argument('--epoch', nargs='?', type=int, default=-1, help='load checkpoint saved at given epoch '
                                                                         'from the export folder ')
    parser.add_argument('--checkpoint_path', nargs='?', type=str, default=None, help='load checkpoint from the given'
                                                                                     ' .pt/pth file')
    parser.add_argument('--override', nargs='?', type=str, default='{}', help='override any parameter in the yaml file')
    parser.add_argument('--start_from_best_in', nargs='?', type=str, default=None,
                        help='start from the model with lowest training loss from the given export folder. If passing '
                             'argument --test_only will start from the model with highest top-1 accuracy in testing')
    parser.add_argument('--get_scores_only', action='store_true',
                        help='get only the softmax and logit scores for the untrimmed training videos. The scores will '
                             'be saved to the export folder')

    args = parser.parse_args()
    settings_path = args.settings_path[0]

    with open(settings_path) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    settings['settings_path'] = settings_path
    settings['test_only'] = args.test_only
    settings['scores_only'] = args.get_scores_only
    settings['update_first_iteration'] = False

    tag = args.tag

    override = eval(args.override)
    assert isinstance(override, dict), 'Override parameter must be a dict-like string'

    for key, value in override.items():
        if isinstance(value, dict):
            for subKey, subValue in value.items():
                assert key in settings, 'Key {} not found in the configuration file!'.format(key)
                assert subKey in settings[key], 'Key {}/{} not found in the configuration file!'.format(key, subKey)
                settings[key][subKey] = subValue
        else:
            assert key in settings, 'Key {} not found in the configuration file!'.format(key)
            settings[key] = value

    assert settings['num_tasks'] == len(settings['num_classes'])

    if args.start_from_best_in is not None:
        if settings['test_only']:
            best_set = 'test'
            value = 'top1_accuracy'
            metric = 'max'
        else:
            best_set = 'train'
            value = 'loss'
            metric = 'min'

        (modelPath, epoch) = find_best_model(os.path.join(args.start_from_best_in, settings['model_name']),
                                             train_or_test=best_set, value=value, metric=metric)
        settings['checkpoint_path'] = modelPath
        settings['n_epochs'] += epoch  # will run for n_epochs anyway regardless of the start epoch
        settings['update_first_iteration'] = True
        settings['update']['start_epoch'] = epoch - 1
        settings['update']['no_update'] = False

        init_tag = args.start_from_best_in.split(os.sep)[-1]
        tag = '_'.join([tag, 'bestFrom_{}'.format(init_tag)])
    else:
        if args.epoch > 0:
            settings['load_checkpoint_epoch'] = args.epoch

        if args.checkpoint_path is not None:
            settings['checkpoint_path'] = args.checkpoint_path

    setup(settings, tag=tag)
    run(settings)
