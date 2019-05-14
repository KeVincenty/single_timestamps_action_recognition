from __future__ import division

import glob
import itertools
import os
import pickle
import matplotlib.backends.backend_agg as plt_backend_agg
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import six
from natsort import natsorted

import dataset_utils
import single_timestamps
from plateau_function import PlateauFunction

if os.environ.get('DISPLAY') is None:
    plt.switch_backend('agg')

mark_dict = {
    'o': 'circle',
    'v': 'triangle_down',
    '^': 'triangle_up',
    '<': 'triangle_left',
    '>': 'triangle_right',
    '1': 'tri_down',
    '2': 'tri_up',
    '3': 'tri_left',
    '4': 'tri_right',
    '8': 'octagon',
    's': 'square',
    'p': 'pentagon',
    '*': 'star',
    'h': 'hexagon1',
    'H': 'hexagon2',
    '+': 'plus',
    'D': 'diamond',
    'd': 'thin_diamond',
    '|': 'vline',
    '_': 'hline'
}


def plot_plateau_evolution(plateau, colour=None, history=None, linewidth=1):
    """
    Plots the evolution of a single plateau

    :param plateau: the plateau function object
    :param colour: colour to be used
    :param history: optionally specify what epochs you want to use for the plot, as a list
    :param linewidth: line width
    :return:
    """
    if history is None:
        history = range(len(plateau.history))

    n_history = len(history)
    alpha_rescale = 0.5

    for i, epoch in enumerate(history):
        if i == 0:
            linestyle = '--'
            plt.plot(plateau.get_y(epoch=epoch), color=colour, linestyle=linestyle, linewidth=linewidth)
        elif i == n_history-1:
            linestyle = '-'
            plt.plot(plateau.get_y(epoch=epoch), color=colour, linestyle=linestyle, alpha=1, linewidth=2 * linewidth)
        else:
            linestyle = '-'
            plt.plot(plateau.get_y(epoch=epoch), color=colour, linestyle=linestyle,
                     alpha=(i / (n_history - 1)) * alpha_rescale, linewidth=linewidth)


def plot_plateaus_in_video(plateaus_in_video, colour_map=None, history=None, plot_evolution=False, plot_points=True,
                           linewidth=1, interactive=False, plot_gt=True, subplots=None, version=None):
    """
    Plots all plateaus in a video

    :param plateaus_in_video: list of plateau function objects
    :param colour_map: dictionary mapping classes to colours
    :param history: optionally specify what epochs you want to use for the plot, as a list
    :param plot_evolution: if True plots evolution of each plateau
    :param plot_points: if True plots a dot at each plateau's centre
    :param linewidth: line width
    :param interactive: if True will plot each plateau separately and pause, waiting for a key to be pressed to continue
    :param plot_gt: if True will plot the corresponding ground truth bounds
    :param subplots: if True will create one subplot per plateau
    :param version: optionally specify what version (epoch) of the plateau you want to plot
    :return: plot handles
    """
    handles = []

    for i, g in enumerate(plateaus_in_video):
        if subplots is not None:
            plt.axes(subplots[i])

        if version is not None:
            g = g.get_version(version, generate_xy=False, return_new_g=True)

        h = plot_single_plateau(g, history, colour_map, linewidth, plot_evolution, plot_points, plot_gt)
        handles.append(h)

        if interactive or subplots is not None:
            if plot_evolution and g.history is not None and g.history:
                cs = [version['c'] for version in g.history] + [g.c]
                ws = [version['w'] for version in g.history] + [g.w]
                left_lim = min(cs)
                right_lim = max(cs)
                window = max(ws)
            else:
                left_lim = g.c
                right_lim = g.c
                window = g.w

            expanded_window = window * 3
            plt.xlim(left_lim - expanded_window, right_lim + expanded_window)

        if interactive:
            plt.show()

            if six.moves.input('Press any key to move to the next plateau. Press q to quit ').lower() == 'q':
                break

            plt.clf()

        if subplots is not None:
            plt.axis('off')

    return handles


def plot_single_plateau(plateau, history, colour_map, linewidth, plot_evolution, plot_points, plot_gt):
    """
    Plots a single plateau

    :param plateau: the plateau object
    :param history: optionally specify what epochs you want to use for the plot, as a list
    :param colour_map: dictionary mapping classes to colours
    :param linewidth: line width
    :param plot_evolution: if True plots evolution of each plateau
    :param plot_points: if True plots a dot at each plateau's centre
    :param plot_gt: if True will plot the corresponding ground truth bounds
    :return: plot handle
    """
    colour = None if colour_map is None else colour_map[plateau.label]
    h = None

    if plot_evolution:
        plot_plateau_evolution(plateau, colour=colour, history=history, linewidth=linewidth)
    else:
        linestyle = '-'
        h = plt.plot(plateau.get_y(), color=colour, linestyle=linestyle, linewidth=linewidth)

    if plot_points:
        plt.scatter(plateau.c, 1, color=colour, linewidth=linewidth)
        plt.text(plateau.c, 1.005, 'c={:0.2f}'.format(plateau.c), color=colour)

    if plot_gt:
        (_, gtStart, gtEnd) = single_timestamps.get_video_name_and_bounds_from_id(plateau.id)
        plt.plot([gtStart, gtEnd], [0.5, 0.5], color=colour, linewidth=linewidth*2, linestyle='-')

    return h


def create_colour_map(classes):
    """
    Maps each class to a different colour

    :param classes: list of integer class indices
    :return: a dictionary whose keys are class indices and values are colours
    """
    color = iter(plt.cm.tab20(np.linspace(0, 1, len(classes))))
    colour_map = {}

    for class_index in classes:
        c = next(color)
        colour_map[class_index] = c

    return colour_map


def make_class_legend(class_map, colour_map, n_columns=3, filter_classes=None, linewidth=1, split_text=True):
    """
    Creates and displays class legend

    :param class_map: dictionary whose keys are integer (class indices) and values are strings (description)
    :param colour_map: dictionary mapping classes to colours
    :param n_columns: number of columns for legend
    :param filter_classes: optionally specify a list of class indices you only want to plot
    :param linewidth: line width
    :param split_text: if True replaces `_` and `-` with spaces
    :return: label handles
    """
    label_handles = []
    plotted_classes = []

    for class_index, class_string in class_map.items():
        if class_index in plotted_classes or (filter_classes is not None and class_index not in filter_classes):
            continue

        plotted_classes.append(class_index)

        if split_text:
            class_string = class_string.replace('-', ' ').replace('_', ' ')

        gt_handle = mlines.Line2D([], [], color=colour_map[class_index], label=class_string, linewidth=linewidth)
        label_handles.append(gt_handle)

    plt.legend(handles=label_handles, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=n_columns, borderaxespad=0.)

    return label_handles


def plot_gt_from_df(df, colour_map, start_key='start_frame', end_key='stop_frame', is0_indexed=False, y=0.5,
                    linestyle='-'):
    """
    Plots labelled action bounds from a pandas dataframe. Optional parameters allow you to specify the dataframe column
    names

    :param df:
    :param colour_map:
    :param start_key:
    :param end_key:
    :param is0_indexed:
    :param y:
    :param linestyle:
    :return:
    """
    shift = 0 if is0_indexed else 1

    for row in df.itertuples():
        plt.plot([getattr(row, start_key) - shift, getattr(row, end_key) - shift], [y, y],
                 color=colour_map[getattr(row, 'class_index')], linestyle=linestyle)
        plt.scatter(getattr(row, start_key) - shift, y, marker='|', color=colour_map[getattr(row, 'class_index')])
        plt.scatter(getattr(row, end_key) - shift, y, marker='|', color=colour_map[getattr(row, 'class_index')])


def plot_update_proposals(plateaus, update_info, softmax_scores=None, colour_map=None, add_legend=True,
                          interactive=True, df=None, plot_all_proposals=True, plot_plateau=True):
    """
    Plots update proposals for all plateaus in a video

    :param plateaus: list of plateau objects
    :param update_info: update info pickle
    :param softmax_scores: softmax scores for the video
    :param colour_map: dictionary mapping classes to colours
    :param add_legend: if True adds a legend to the plot
    :param interactive: if True will plot each plateau separately and pause, waiting for a key to be pressed to continue
    :param df: pandas dataframe
    :param plot_all_proposals: if True will plot all proposals, otherwise will plot only the selected proposal
    :param plot_plateau: if True will plot the original plateau
    :return:
    """
    proposals_key = 'all_proposals'

    for g in plateaus:
        colour = None if colour_map is None else colour_map[g.label]

        if plot_plateau:
            plt.plot(g.get_y(epoch=0), linestyle='--', label='updating g', color=colour)  # initial plateau

        info = update_info[g.id]
        marker_cycle = itertools.cycle(mark_dict.keys())

        for prop in info[proposals_key]:
            label = 'candidate, conf={:0.3f}'.format(prop.confidence)
            chosen = False
            linestyle = ':'

            if info['updated'] and prop.has_same_parameters_as(info['chosen_proposal']):
                chosen = True

            if chosen:
                label += ' (chosen)'
                linestyle = '-'

            if plot_all_proposals or (not plot_all_proposals and chosen):
                plt.plot(prop.y, label=label, color=colour, linestyle=linestyle, marker=next(marker_cycle))

        if softmax_scores is not None:
            plt.plot(softmax_scores[g.label], color=colour, linestyle='-.', linewidth=0.5, label='softmax scores')

        if df is not None:
            plot_gt_from_df(df, colour_map)

        if add_legend:
            plt.legend()

        if interactive:
            six.moves.input('Press a key to continue')
            plt.clf()


def do_plots_for_video(df, video, plateaus_in_video, history=None, plot_evolution=True, plot_legend=True,
                       interactive=False):
    """
    Plots ground truth bounds and plateaus for all plateaus in a single video

    :param df: pandas training dataframe
    :param video: video id
    :param plateaus_in_video: list of plateau objects
    :param history: optionally specify what epochs you want to use for the plot, as a list
    :param plot_evolution: if True will plot the plateaus' evolution
    :param plot_legend: if True will plot the class legend
    :param interactive: if True will plot each plateau separately and pause, waiting for a key to be pressed to continue
    :return:
    """
    plt.clf()

    video_lines = dataset_utils.get_video_lines_from_df(df, video)
    classes = video_lines.class_index.unique()
    (class_labels, class_map) = dataset_utils.get_classes_from_df(video_lines)
    colour_map = create_colour_map(classes)

    plt.title(video)
    plot_gt_from_df(video_lines, colour_map)
    plot_plateaus_in_video(plateaus_in_video, colour_map=colour_map, history=history, plot_evolution=plot_evolution,
                           interactive=interactive)

    if plot_legend:
        make_class_legend(class_labels, colour_map)

    plt.show()

    return colour_map


def do_plots_for_epoch(train_dict_path, df, epoch=None, plot_evolution=True, videos=None, plot_legend=False,
                       interactive=False):
    """
    Plots everything (gt, plateaus and update proposals) for all plateaus from all videos, for a single epoch

    :param train_dict_path: path to training dictionaries
    :param df: training pandas dataframe
    :param epoch: epoch number
    :param plot_evolution: if True will plot the plateaus' evolution
    :param videos: optionally specify the video ids of the videos you want to plot, otherwise will plot all videos
    :param plot_legend: if True will add the class legend
    :param interactive: if True will plot each plateau separately and pause, waiting for a key to be pressed to continue
    :return: nothing
    """
    if epoch is None:
        epoch_folder = natsorted([p for p in glob.glob(os.path.join(train_dict_path, '*')) if os.path.isdir(p)])
        epoch_folder = epoch_folder[-1]
    else:
        epoch_folder = os.path.join(train_dict_path, 'epoch_{}'.format(epoch))

    print('Using epoch folder {}'.format(epoch_folder))

    if videos is None:
        video_folders = natsorted([p for p in glob.glob(os.path.join(epoch_folder, '*')) if os.path.isdir(p)])
    else:
        video_folders = natsorted(os.path.join(epoch_folder, video) for video in videos)

    for vf in video_folders:
        video = os.path.basename(vf)
        plateaus_in_video = PlateauFunction.load_plateaus_from_file(os.path.join(vf, '{}_plateaus.csv'.format(video)))
        colour_map = do_plots_for_video(df, video, plateaus_in_video, plot_evolution=plot_evolution,
                                        plot_legend=plot_legend, interactive=interactive)

        update_info_path = os.path.join(vf, '{}_update_info.pkl'.format(video))

        if os.path.exists(update_info_path):
            with open(update_info_path, 'rb') as f:
                update_info = pickle.load(f)
                plot_update_proposals(plateaus_in_video, update_info, softmax_scores=None, colour_map=colour_map,
                                      add_legend=False, interactive=False, df=None, plot_all_proposals=False,
                                      plot_plateau=False)

        if interactive and six.moves.input('Press any key to continue. Press q to quit').lower() == 'q':
            break


def plot_confusion_matrix(cm, classes, normalise=True, cmap=plt.cm.Blues, sort_by_n_instances=True):
    """
    Plots a confusion matrix

    :param cm: the confusion matrix, as NumPy 2D array
    :param classes: list of class strings
    :param normalise: if True will normalise the confusion matrix, i.e. numbers will be in percentages
    :param cmap: Matplotlib colour map
    :param sort_by_n_instances: if True sorts the rows according to the number of instances per class
    :return: Matplotlib figure
    """
    instances_per_class = np.sum(cm, axis=1)

    if sort_by_n_instances:
        sort_idx = np.flip(np.argsort(instances_per_class), axis=0)
        instances_per_class = instances_per_class[sort_idx]
        cm = cm[sort_idx,:]

        for r in range(cm.shape[0]):
            cm[r] = cm[r, sort_idx]

        classes = [classes[s_idx] for s_idx in sort_idx]

    classes = ['{}:{} ({})'.format(idx, s, instances_per_class[idx]) for idx, s in enumerate(classes)]

    fig_size = 10 * (len(classes) / 34)  # proportions based on beoid's classes
    fig = plt.figure(figsize=(fig_size, fig_size))

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks, classes)

    try:
        plt.tight_layout()
    except Exception:
        pass

    return fig


def figure_to_img(figure, close_figure=True):
    """
    Converse a matplotlib figure to an image

    :param figure: Matplotlib image
    :param close_figure: if True closes the figure
    :return: the image as NumPy array
    """
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image = data.reshape([h, w, 4])[:, :, 0:3]

    if close_figure:
        plt.close(figure)

    return image


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../annotations/beoid/train_df.csv')
    train_dicts_path = '/home/viki/Davide/temp/beoid/ts/tsn_bni/training_dicts'
    interactive = False

    if interactive:
        plt.ion()

    do_plots_for_epoch(train_dicts_path, df, plot_evolution=False, epoch=10, videos=None, interactive=interactive,
                       plot_legend=False)

