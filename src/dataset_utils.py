from __future__ import division

import collections
import pandas as pd


def parse_annotations(train_df_path, test_df_path):
    """
    Loads csv annotations as pandas data frames. The training csv should contain the following columns: class,
    class_index, point, start_frame, stop_frame, video_id. Testing csv do not have to contain the `point` column, which
    would be ignored anyway. Point ant class_indices should be 0-indexed

    :param train_df_path: train to train csv
    :param test_df_path: train to test csv
    :return: (train_df, test_df, class_labels, class_map). Train_df and test_df are pandas data frames, whereas
     class_labels is an ordered dictionary whose keys are integer (class_index column in the csv) and values are the
     corresponding class descriptive string (class column in the csv). class_map is the opposite of class_labels, i.e.
     keys are strings and values are integers
    """
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    (class_labels, class_map) = get_classes_from_df(train_df)
    class_labels = collections.OrderedDict(sorted(class_labels.items()))

    return train_df, test_df, class_labels, class_map


def get_classes_from_df(df):
    class_labels = {}
    class_map = {}

    for i, row in df.iterrows():
        class_labels[row['class_index']] = row['class']
        class_map[row['class']] = row['class_index']

    return class_labels, class_map


def get_video_lines_from_df(df, video, key='video_id'):
    return df.loc[df[key] == video]
