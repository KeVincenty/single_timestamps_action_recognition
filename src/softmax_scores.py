import torch
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
from typing import Dict

def split_task_outputs(output: torch.Tensor, tasks: Dict) -> Dict[str, torch.Tensor]:
    offset = 0
    outputs = dict()
    for task, n_units in tasks.items():
        outputs[task] = output[..., offset : offset + n_units]
        offset += n_units
    return outputs

def get_softmax_scores_for_untrimmed_videos(data_loader, model, mode, stack_size, return_logits=False, tasks={'noun':300, 'verb':97}):
    """
    Gets the softmax scores for each frame in each untrimmed video

    :param data_loader: Pytorch data loader
    :param model: Pytorch model
    :param mode: either `rgb` or `flow`
    :param stack_size: optical flow stack size
    :param return_logits: if True will also return logit scores
    :return: (video_softmax, video_logits, video_n_frames). These are all dictionary whose keys are video ids and values
     are either softmax/logit scores or number of frames in the video
    """
    video_softmax = {}
    video_logits = {}
    video_n_frames = {}
    model.train(False)
    stride = data_loader.dataset.softmax_stride

    n_classes = data_loader.dataset.n_classes
    data_length = len(data_loader)

    message = '-> Extracting sofmax scores (mode: {})'.format(mode)

    if mode == 'flow':
        message += ' (stride={})'.format(stride)

    bar = tqdm(total=data_length, desc=message, file=sys.stdout)

    video_lengths = data_loader.dataset.video_lengths

    if model.training:
        model.eval()
        was_training = True
        print('-> Model in training mode: switching to test mode now')
    else:
        was_training = False
    
    with torch.no_grad():

        for video in data_loader.dataset.data_frame.video_id.unique():
            n_frames = video_lengths[video]
            video_softmax[video] = np.zeros((n_classes, n_frames), dtype=np.float32)

            if return_logits:
                video_logits[video] = np.zeros((n_classes, n_frames), dtype=np.float32)

        for _, data in enumerate(data_loader):
            inputs, _, video_info = data
            video_ids = video_info['video_id']
            segments_indexes = video_info['indexes']

            inputs = inputs.cuda().float()
            logits = model(inputs, do_consensus=False, override_reshape=False)
            logits_values = logits.data.cpu().clone().numpy()
            logits_v = split_task_outputs(logits, tasks)

            # getting softmax values
            task_softmax_values = []
            for k,v in logits_v.items():
                task_softmax_value = F.softmax(v, dim=1)
                task_softmax_values.append(task_softmax_value)
            softmax_values = torch.cat(task_softmax_values, dim=-1).data.cpu().clone().numpy()

            softmax_values = softmax_values.transpose(1, 0)
            logits_values = logits_values.transpose(1, 0)

            # stack the scores
            for seg_i, indexes in enumerate(segments_indexes):
                video_id = video_ids[seg_i]

                if mode == 'rgb':
                    video_softmax[video_id][:, indexes] = softmax_values[:, seg_i, np.newaxis]

                    if return_logits:
                        video_logits[video_id][:, indexes] = logits_values[:, seg_i, np.newaxis]
                elif mode == 'flow':
                    s = indexes[0] + np.floor((stack_size - stride) / 2)
                    e = s + stride
                    scores_frames_idx = np.arange(s, e, dtype=np.int)
                    n_scores = len(scores_frames_idx)

                    if n_scores > 1:
                        video_softmax[video_id][:, scores_frames_idx] = np.repeat(softmax_values[:, seg_i, np.newaxis], n_scores,axis=1)

                        if return_logits:
                            video_logits[video_id][:, scores_frames_idx] = np.repeat(logits_values[:, seg_i, np.newaxis], n_scores,axis=1)
                    else:
                        video_softmax[video_id][:, scores_frames_idx] = softmax_values[:, seg_i, np.newaxis]

                        if return_logits:
                            video_logits[video_id][:, scores_frames_idx] = logits_values[:, seg_i, np.newaxis]
                else:
                    raise Exception('Unrecognised mode: {}'.format(mode))

            bar.update()

    bar.close()

    if was_training:
        print('-> Model was in training mode: switching back to train mode now')
        model.train()

    return video_softmax, video_logits, video_n_frames
