from __future__ import division

import numpy as np
import csv

import fit
import copy
from collections import OrderedDict

np.seterr(over='ignore')  # ignoring overflow error that occur during exp


class PlateauFunction:
    """
    The plateau function class. Contains functions to initialise plateaus, as well as functions for sampling
    and updating.

    The plateau function is defined as

    .. math::
        1 / {(e^{s(x-c-w)} + 1) (e^{s(-x+c-w)} + 1)}
    """

    @staticmethod
    def function(x, c, w, s):
        return 1 / ((np.exp(s * ((x - c) - w)) + 1) * (np.exp(s * (-(x - c) - w)) + 1))

    def __init__(self, c, w, s, n, generate_xy=True):
        """
        Creates a plateau function object.

        :param c: centre of the plateau
        :param w: width of the plateau (the actual width will be 2w)
        :param s: steepness of the side slopes
        :param n: number of frames of the untrimmed video
        """

        assert n > 1, 'n must be positive non-zero'
        assert w > 0, 'w must be positive non-zero'
        assert c >= 0 and c < n, 'c must be between 0 and n-1! c was %d, n was %d' % (c, n)
        assert s > 0 and s <= 1, 's must be between 0 and 1!'

        if generate_xy:
            x = np.arange(0, n)
        else:
            x = None
            self.y = None
            self.yn = None

        self.x = x
        self.c = c
        self.w = w
        self.s = s
        self.n = n               
        self.label = None
        self.id = None
        self.index = None  # this is the 0-index order of the g in its corresponding video
        self.confidence = None
        self.video = None
        self.connected_component = None  # used only by proposals
        self.tau = None  # the threshold used to obtain the connected_component
        self.is_proposal = False

        if generate_xy:
            self._generate_y()
        
        self.history = [{'c': c, 'w': w, 's': s}]  # this array contains the history of the settings

    def set_video(self, video):
        self.video = video

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_index(self, index):
        self.index = index

    def set_label(self, label):
        self.label = label

    def set_id(self, id):
        self.id = id

    def sample_points(self, n_samples=1, x_range=None):
        """
        Samples points from a single plateau function. These should be used as frame indexes.

        :param n_samples: number of samples to draw
        :param x_range: the x range where to sample. If None will use the entire video length
        :return: the sampled points, as NumPy array of integers
        """
        assert not self.is_proposal, 'why are you sampling points from a proposal?!'
        y = self.yn
        points = np.random.choice(self.x, size=n_samples, replace=False, p=y)

        if x_range is not None:
            min_range = x_range[0]
            max_range = x_range[-1] - 1

            for i in range(len(points)):
                points[i] = max(min_range, min(points[i], max_range))

        points = points.astype(np.int32)

        return points

    def match_to_proposals(self, q_proposals):
        """
        Matches a plateau function to an update proposal

        :param q_proposals: list of proposals
        :return: list of matched proposals
        """
        matching_proposals = []

        for q in q_proposals:
            proposal = q

            if q.index is not None:
                # in this case we are matching the same q to multiples plateaus sharing a same label
                # we need thus to duplicate the object
                proposal = copy.deepcopy(q)

            proposal.set_label(self.label)
            proposal.set_index(self.index)
            matching_proposals.append(proposal)

        return matching_proposals

    def update_parameters(self, q, lc, lw, ls):
        """
        Updates the parameters of the plateau function, given an update proposal and the velocity parameters as follows:

        .. math::
            c = g_c - lc * (g_c - q_c)

            w = g_w - lw * (g_w - q_w)

            s = g_s - ls * (g_s - q_s)
        """
        
        self.c = self.c - lc * (self.c - q.c)
        self.w = self.w - lw * (self.w - q.w)
        self.s = self.s - ls * (self.s - q.s)

        self._generate_y()  # regenerate y after we have updated the settings
        self.history.append({'c': self.c, 'w': self.w, 's': self.s})  # this array contains the history of the settings

    def no_update(self):
        self.history.append({'c': self.c, 'w': self.w, 's': self.s})  # updating history in any case

    def get_y(self, epoch=None):
        if epoch is None:            
            c = self.c
            w = self.w
            s = self.s
        else:
            c = self.history[epoch]['c']
            w = self.history[epoch]['w']
            s = self.history[epoch]['s']

        if self.is_proposal or self.x is None:
            x = np.arange(0, self.n)
        else:
            x = self.x
        
        return PlateauFunction.function(x, c, w, s)

    def get_version(self, epoch, return_new_g=False, generate_xy=False):
        if return_new_g:
            params = self.history[epoch]
            new_g = PlateauFunction(params['c'], params['w'], params['s'], self.n, generate_xy=generate_xy)
            new_g.video = self.video
            new_g.label = self.label
            new_g.id = self.id
            new_g.index = self.index

            return new_g
        else:
            return self.history[epoch]

    def _change_parameters(self, new_c, new_w, new_s):
        """
        this must be used only to change the parameters after we have matched the g to the fitted q
        this is intended to be used only when updating the distribution
        """
        self.c = new_c
        self.w = new_w
        self.s = new_s
        
        self._generate_y()  # regenerate y after we have updated the settings
        
        self.history[-1] = {'c': self.c, 'w': self.w, 's': self.s}  # changing the last history update

    def _generate_y(self):
        y = PlateauFunction.function(self.x, self.c, self.w, self.s)
        self.y = y
        y_for_sampling = np.array(y)

        self.yn = y_for_sampling / np.sum(y_for_sampling)

    def __str__(self):
        return 'c: {}\nw: {}\ns: {}\nlabel: {}\nid: {}\nindex: {}'.format(self.c, self.w, self.s, self.label, self.id,
                                                                          self.index)

    def has_same_parameters_as(self, g, eps_c=0, eps_w=0, eps_s=0):
        return abs(self.c - g.c) <= eps_c and abs(self.w - g.w) <= eps_w and abs(self.s - g.s) <= eps_s

    def has_same_parameters_as_in_list(self, plateaus, eps_c=0, eps_w=0, eps_s=0):
        same_parameters = False

        for g in plateaus:
            same_parameters = self.has_same_parameters_as(g, eps_c=eps_c, eps_w=eps_w, eps_s=eps_s)

            if same_parameters:
                break

        return same_parameters

    def was_updated(self):
        previous_version = self.history[-min(2, len(self.history))]
        # yeah this is ugly but is the right way to get the right number since history values are approximated like this
        c = float('{:.3f}'.format(self.c))
        w = float('{:.3f}'.format(self.w))
        s = float('{:.3f}'.format(self.s))

        return c != previous_version['c'] or w != previous_version['w'] or s != previous_version['s']

    def to_connected_component(self, tau=0.5, cheap=False):
        """
        Returns the connected component of indices where the plateau function is above a threshold tau

        :param tau: the threshold for the connected component
        :param cheap: if True, will approximate the connected component as [g.c - g.w, g.c + g.w] to avoid potentially
         expensive access of NumPy arrays
        :return: connected_component, a dictionary containing the connected component's start, end, length and indices
        """
        if not cheap:
            (cc_start, cc_end, cc_length, components) = fit.find_connected_components_in_indices(
                np.where(self.get_y() >= tau)[0])

            if len(components) != 1:
                raise Exception('Trying to calculate confidence of a suspicious g:'.format(self))

            cc = components[0]
            cc_start = cc_start[0]
            cc_end = cc_end[0]
            cc_length = cc_length[0]
        else:
            cheap_start = int(np.round(self.c - self.w))
            cheap_end = int(np.round(self.c + self.w))
            cc_start = max(0, cheap_start)
            cc_end = min(self.n-2, cheap_end)

            assert cc_end > cc_start, 'Check this g: {}'.format(self)

            cc_length = cc_end - cc_start + 1
            cc = np.arange(cc_start, cc_end+1).astype(np.int)

        connected_component = {
            'start_frame': cc_start,
            'stop_frame': cc_end,
            'length': cc_length,
            'cc': cc
        }

        return connected_component

    @staticmethod
    def to_dict(g, add_history=True):
        d = {
            'c': g.c,
            'w': g.w,
            's': g.s,
            'N': g.n,
            'label': g.label,
            'id': g.id,
            'index': g.index,
            'confidence': g.confidence,
            'video': g.video
        }

        if add_history:
            d['history'] = ';'.join(['c:{:.3f}|w:{:.3f}|s:{:.3f}'.format(h['c'], h['w'], h['s']) for h in g.history])

        return d

    @staticmethod
    def check_order_is_preserved(plateaus):
        """
        Checks the initial order of the actions in a video is preserved

        :param plateaus: list of plateaus
        :return: (is_order_preserved, good_g, bad_g). is_order_preserved is True if the order is preserved, good_g is a
         list containing the plateaus respecting the original order, while bad_g will contain those that do not respect
         it
        """
        points_g_map = {}

        for g in plateaus:
            if g.index is None:
                raise Exception('No index set for this g {}'.format(g))

            points_g_map[g.c] = g

        point_list = list(points_g_map.keys())
        new_order = np.argsort(point_list)
        good_g = []
        bad_g = []

        for i, index in enumerate(new_order):
            if index == points_g_map[point_list[i]].index:
                good_g.append(points_g_map[point_list[i]])
            else:
                bad_g.append(points_g_map[point_list[i]])

        is_order_preserved = len(bad_g) == 0

        return is_order_preserved, good_g, bad_g

    @staticmethod
    def from_df(df, c_key='g_c', w_key='g_w', s_key='g_s', n_key='g_N', id_key='id',
                sort_key='start_frame', label_key='class_index', video_id_key='video_id',
                confidence_key='confidence', index_key='index', return_seg_bounds=False, as_dict=False):
        """
        Creates a list of plateau function from a pandas data frame. Optional parameters can be used to specify the
        dataframe's column names to be used

        :param df:
        :param c_key:
        :param w_key:
        :param s_key:
        :param n_key:
        :param id_key:
        :param sort_key:
        :param label_key:
        :param video_id_key:
        :param confidence_key:
        :param index_key:
        :param return_seg_bounds:
        :param as_dict:
        :return: (plateaus, action_bounds)
        """
        plateaus = {} if as_dict else []
        action_bounds = []
        df = df.sort_values(sort_key, inplace=False)

        for tup in df.itertuples():
            c = getattr(tup, c_key)
            w = getattr(tup, w_key)
            s = getattr(tup, s_key)
            n = getattr(tup, n_key)
            label = getattr(tup, label_key)
            video_id = getattr(tup, video_id_key)
            confidence = getattr(tup, confidence_key)
            g_id = getattr(tup, id_key)
            index = getattr(tup, index_key)

            g = PlateauFunction(c, w, s, n, generate_xy=False)
            g.set_label(label)
            g.set_video(video_id)
            g.set_confidence(confidence)
            g.set_id(g_id)
            g.set_index(index)

            if as_dict:
                if video_id not in plateaus:
                    plateaus[video_id] = []

                plateaus[video_id].append(g)
            else:
                plateaus.append(g)

            if return_seg_bounds:
                bounds = [tup.start_frame, tup.stop_frame]
                action_bounds.append(bounds)

        return plateaus, action_bounds

    @staticmethod
    def from_dict(dictionary, c_key='c', w_key='w', s_key='s', n_key='N', has_label=True, generate_xy=False):
        """
        Creates a single plateau function from a dictionary

        :param dictionary:
        :param c_key:
        :param w_key:
        :param s_key:
        :param n_key:
        :param has_label:
        :param generate_xy:
        :return: the plateau function
        """
        c = dictionary[c_key]
        w = dictionary[w_key]
        s = dictionary[s_key]
        n = dictionary[n_key]
        video = dictionary['video']
        index = dictionary['index']
        id = dictionary['id']
        label = dictionary['label'] if has_label else None

        g = PlateauFunction(c, w, s, n, generate_xy=generate_xy)
        g.set_index(index)
        g.set_video(video)
        g.set_id(id)
        g.set_label(label)

        return g

    @staticmethod
    def initialise_plateaus(points, n_frames, init_w, init_s):
        """
        Initialise n plateau functions given n points (or single timestamps) for a single video.
        Each plateau will be centred on each point and will have fixed parameters w and s.

        :param points: list of points, referring to single timestamps (frame indices) in the video
        :param n_frames: number of frames in the video
        :param init_w: initial w
        :param init_s: initial s
        :return: list of plateau function objects
        """
        plateaus = []

        for point in points:
            g = PlateauFunction(point, init_w, init_s, n_frames)
            plateaus.append(g)

        return plateaus

    @staticmethod
    def sample_from_plateaus(plateaus, n_points, mode, max_attempts_before_hack=5, stack_size=10):
        """
        Samples points from all the plateaus in a video. Each point will correspond to a frame index which will be used
        for training. Each frame index will correspond to a class label. The function ensures that the order of the
        actions in the video is preserved

        :param plateaus: list of plateaus in video
        :param n_points: how many points to sample from each plateau
        :param mode: either rgb or flow
        :param max_attempts_before_hack: how many attempts before doing the hack to ensure actions' order is respected
        :param stack_size: the optical flow stack size
        :return: (sampled_points_dict, sampled_points_matrix). Sampled points as dictionary and NumPy matrix
        """
        n_plateaus = len(plateaus)
        sampled_points_matrix = np.empty([n_plateaus, n_points], dtype=int)
        generate_numbers = True
        n_attempts = 0

        while generate_numbers:
            n_attempts += 1
            
            for i, g in enumerate(plateaus):
                if mode == 'flow':
                    x_range = np.arange(stack_size / 2, g.n - stack_size / 2, dtype=np.int32)
                else:
                    x_range = None  # will take the whole x later for sampling

                # this generates n_set samples with no repetitions for one g only
                sampled_points_matrix[i] = g.sample_points(n_points, x_range=x_range)
                            
            # check sampled frames respect order of g
            # check if each column is sorted with diff. Contains true if the column is sorted
            columns_check = [np.all(np.diff(sampled_points_matrix[:, c]) > 0) for c in
                             np.arange(sampled_points_matrix.shape[-1])]
            
            generate_numbers = not np.all(columns_check)
            
            if n_attempts > max_attempts_before_hack and generate_numbers:
                # this happens when two gs are almost identical and it is not possible
                # to obtain sorted points from such gs. 
                # In this case we sort the troublesome columns
                
                for c in np.arange(sampled_points_matrix.shape[-1]):
                    if not columns_check[c]:  # if the column is not sorted
                        sampled_points_matrix[:, c] = np.sort(sampled_points_matrix[:, c])
                        # it may happen now that for a single g we have repeated points,
                        # this is not a big problem though
                
                generate_numbers = False

        sampled_points_dict = OrderedDict()

        for i, g in enumerate(plateaus):
            assert g.id is not None, 'No id set for this g: {} '.format(g)
            sampled_points_dict[g.id] = sampled_points_matrix[i]

        return sampled_points_dict, sampled_points_matrix

    @staticmethod
    def params_to_string(g, add_confidence=False):
        if add_confidence:
            return 'c:{g.c:.3f}|w:{g.w:.3f}|s:{g.s:.3f}|conf:{g.confidence:.3f}'.format(g=g)
        else:
            return 'c:{g.c:.3f}|w:{g.w:.3f}|s:{g.s:.3f}'.format(g=g)

    @staticmethod
    def write_plateaus_to_file(plateaus, path):
        with open(path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['c', 'w', 's', 'N', 'label', 'id', 'index', 'confidence', 'video',
                                                   'history'])
            writer.writeheader()
            
            for g in plateaus:
                writer.writerow(PlateauFunction.to_dict(g))

    @staticmethod
    def params_from_string(string, with_conf=False):
        params = string.split('|')

        d = {
              'c': float(params[0].split('c:')[1]),
              'w': float(params[1].split('w:')[1]),
              's': float(params[2].split('s:')[1])
        }

        if with_conf:
            d['confidence'] = float(params[3].split('conf:')[1])

        return d

    @staticmethod
    def load_plateaus_from_file(path, generate_xy=False):
        plateaus = []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                g = PlateauFunction(float(row['c']), float(row['w']), float(row['s']), float(row['N']),
                                    generate_xy=generate_xy)

                if 'label' in row and row['label']:
                    g.set_label(int(row['label']))

                if 'id' in row and row['id']:
                    g.set_id(row['id'])

                if 'index' in row and row['index']:
                    g.set_index(int(row['index']))

                if 'confidence' in row and row['confidence']:
                    g.set_confidence(float(row['confidence']))

                if 'video' in row and row['video']:
                    g.set_video(row['video'])

                history = []

                if 'history' in row and row['history']:
                    for h in row['history'].split(';'):
                        history.append(PlateauFunction.params_from_string(h))

                    g.history = history

                plateaus.append(g)
        
        return plateaus
