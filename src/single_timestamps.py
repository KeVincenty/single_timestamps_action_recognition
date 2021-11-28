from __future__ import division

import numpy as np
import plateau_function as pf
import fit as fit
import update
import sys
from tqdm import tqdm


def initialise_plateaus_for_dataset(df, settings, video_lengths):
    """
    Initialises a plateau for each action in each video

    :param df: pandas training dataframe
    :param settings: settings dictionary
    :param video_lengths: dictionary containing video lengths
    :return: initial_plateaus, a dictionary whose keys are video ids and values are list of plateau objects
    """
    print('-> Initialising plateaus')

    videos = df.video_id.unique()
    initial_plateaus = {}

    for video in videos:
        n_frames = video_lengths[video]

        initial_plateaus[video] = initialise_plateaus_for_video(
            df,
            video,
            n_frames,
            settings['plateaus_initial_parameters']['w'],
            settings['plateaus_initial_parameters']['s'],
            class0_indexed=True,
        )

    return initial_plateaus


def initialise_plateaus_for_video(df, video, n_frames, init_w, init_s, class0_indexed=False):
    """
    Initialise a plateaus for each action in a single video

    :param df: pandas training dataframe
    :param video: video id
    :param n_frames: number of frames in the video
    :param init_w: initial w for the plateaus
    :param init_s: initial s for the plateaus
    :param class0_indexed: if False will subtract 1 to class labels
    :return: plateaus, a list of plateau function objects
    """
    class_shift = 0 if class0_indexed else 1
    points_info = {}
    video_rows = df.loc[df.video_id == video]

    for _, row in video_rows.iterrows():  # we need to use iterrow to get the dictionary
        point = row['point']

        if point in points_info:
            raise Exception('Point {} is duplicated in video: {}. Fix table'.format(point, video))

        points_info[point] = {
                                'point': point,
                                'label': {'noun_class':int(eval(row['class_index'])[0])- class_shift, 'verb_class':int(eval(row['class_index'])[-1])- class_shift},
                                'id': generate_plateau_id(**row)
                            }

    points = sorted([c['point'] for c in points_info.values()])
    plateaus = pf.PlateauFunction.initialise_plateaus(points, n_frames, init_w, init_s)
                
    for i, g in enumerate(plateaus):
        point = g.c
        label = points_info[point]['label']
        g_id = points_info[point]['id']
        g.set_label(label)
        g.set_id(g_id)
        g.set_index(i) # this is the order of the action in the video
        g.set_video(video)

    # sanity check
    is_sorted = pf.PlateauFunction.check_order_is_preserved(plateaus)[0]

    if not is_sorted:
        raise Exception('Unsorted plateaus for video: {}'.format(video))
        
    return plateaus


def sample_points_from_plateaus(all_plateaus, mode, stack_size=10, n_samples=1):
    """
    Samples points from each plateau in each video

    :param all_plateaus: dictionary containing all plateaus, keys are plateaus's ids, values are the plateau objects
    :param mode: either `flow` or `rgb`
    :param stack_size: optical flow stack size
    :param n_samples: number of samples you want to draw from each plateau
    :return: sampled_points, dictionary whose keys are video ids and whose values are dictionary containing the sampled
     points as values as the plateaus ids as keys
    """
    sampled_points = {}
    h_stack_c = np.ceil(stack_size / 2)

    for g_id, g in all_plateaus.items():
        if mode == 'flow':
            x_range = np.arange(h_stack_c+1, g.n - h_stack_c, dtype=np.int32)
        else:
            x_range = None  # will take the whole x later for sampling

        if g.video not in sampled_points:
            sampled_points[g.video] = {}

        sampled_points[g.video][g_id] = g.sample_points(n_samples, x_range=x_range)

    return sampled_points


def update_plateaus_in_dataset(plateaus_per_video, softmax_scores, training_dict, settings):
    """
    Update plateaus in dataset

    :param plateaus_per_video: dictionary whose keys are video ids and values are list of plateau objects
    :param softmax_scores: dictionary whose keys are video ids and values are softmax scores
    :param training_dict: training dictionary
    :param settings: settings dictionary
    :return: number of plateaus that have been updated
    """
    update_proposals = {}
    update_info = {}
    n_videos = len(training_dict)

    assert settings['update']['z'] > 0 and settings['update']['z'] <= 1, 'update z must be between 0 (excluded) and 1!'

    bar = tqdm(total=n_videos, desc='-> Fitting softmax scores scores and generating update proposals...',
               file=sys.stdout)

    for video, scores in softmax_scores.items():
        (proposals, windows) = fit.fit_proposals(
            scores,
            settings['fit']['min_tau'],
            settings['fit']['max_tau'],
            settings['fit']['min_cc_length'],
            tau_step=settings['fit']['tau_step'])

        update_proposals[video] = proposals
        bar.update()

    bar.close()

    print('-> Updating plateaus...')

    total_updated = update.update_plateaus(plateaus_per_video, softmax_scores, update_proposals,
                                           settings['update'], update_info)

    for video, info in update_info.items():
        training_dict[video]['update_info'] = info

    return total_updated


def add_sampled_points_to_train_info(train_dict, sampled_points, n_point_sets, plateaus):
    for g_id, g in plateaus.items():
        video_id = g.video

        if 'sampled_points' not in train_dict[video_id]:
            plateaus = train_dict[video_id]['plateaus']
            points_matrix = np.empty([len(plateaus), n_point_sets], dtype=int)
            train_dict[video_id]['sampled_points'] = points_matrix

        train_dict[video_id]['sampled_points'][g.index] = sampled_points[video_id][g_id]


def generate_plateau_id(**kwargs):
    return '{}_{}_{}'.format(kwargs['video_id'], kwargs['start_frame'], kwargs['stop_frame'])


def get_video_name_and_bounds_from_id(g_id, are_bounds0_indexed=False, as_dict=False):
    splits = g_id.split('_')
    shift = 0 if are_bounds0_indexed else -1

    video = '_'.join(splits[:-2])
    start = int(splits[-2]) + shift
    end = int(splits[-1]) + shift

    if as_dict:
        return {'video': video, 'start_frame': start, 'stop_frame': end}
    else:
        return video, start, end


def get_all_plateaus_in_dataset(plateaus_per_video):
    """
    :param plateaus_per_video: dictionary whose keys are video ids and values are list of plateau objects
    :return: a dictionary whose keys are plateau ids and values are corresponding plateau objects
    """
    return {g.id: g for g in [p for plateaus in plateaus_per_video.values() for p in plateaus]}
