import os

import numpy as np
import torch.utils.data as data_utl
from PIL import Image

from single_timestamps import generate_plateau_id


class Dataset(data_utl.Dataset):
    """
    Class extending PyTorch's dataset class.

    This class handles both the training and test sets, as well the untrimmed training set used for getting the
    softmax scores for the update.
    """
    def __init__(self, name, data_frame, classes, frames_path, n_samples, mode, sampling_strategy, model_name,
                 video_lengths, softmax_stride=1, transforms=None, untrimmed=False, rgb_img_size=226,
                 flow_img_size=224, stack_size=5, frames_are1_indexed=True):
        """
        :param name: name of the dataset. At the moment must be either `beoid` or `thumos_14`
        :param data_frame: pandas data frame containing the training or testing annotations
        :param classes: list of classes, where each class be an integer
        :param frames_path: path to rgb or flow frames
        :param n_samples: number of frames/stacks to be sampled for each action instance
        :param mode: must be either 'rgb' or 'flow'
        :param sampling_strategy: tells how to sample frames from either gt segments or plateaus.
         Must be either 'random_tsn', 'uniform', 'samples_from_plateaus'
        :param model_name: name of pytorch model. Currently not used, can be useful to load the data in a specific way
         for a different model
        :param video_lengths: dictionary containing number of frames (values) for each untrimmed video (keys)
        :param softmax_stride: stride for when extracting softmax scores from the untrimmed videos
        :param transforms: pytorch transforms to be applied to the data
        :param untrimmed: if True will load the untrimmed videos
        :param rgb_img_size: long side of the rgb frames, in pixels
        :param flow_img_size: long side of the flow frames, in pixels
        :param stack_size: stack size for optical flow mode
        :param frames_are1_indexed: if True it assumes frames' file names are 1-indexed (i.e. first frame would be
         something like `frame_000001.jpg`
        """
        assert mode is not None and mode in ['rgb', 'flow'], 'You must specify a valid mode! (rgb or flow)'

        self.name = name
        self.data_frame = data_frame
        self.transforms = transforms
        self.mode = mode
        self.frames_path = frames_path
        self.n_samples = n_samples
        self.untrimmed = untrimmed
        self.classes = classes
        self.n_classes = len(classes)
        self.sampling_strategy = sampling_strategy
        self.rgb_img_size = rgb_img_size
        self.flow_img_size = flow_img_size
        self.model_name = model_name
        self.stack_size = stack_size
        self.video_lengths = video_lengths
        self.samples_from_plateaus = {}
        self.class_labels = {}  # keys are class indexes (int), values are class names (str)
        self.class_map = {}  # the inverse of class_labels
        self.frames_are1_indexed = frames_are1_indexed
        self.data = []

        if untrimmed:
            if mode == 'flow':
                assert softmax_stride > 0 and softmax_stride <= self.stack_size

            self.softmax_stride = softmax_stride
            self.n_frames_per_segment = n_samples if (self.mode == 'rgb' or self.mode == 'idt_fv') else stack_size

            self.make_dataset_untrimmed()
        else:
            self.n_frames_per_segment = None
            self.make_dataset()

    def __getitem__(self, index):
        """
        Generates frame samples and loads the corresponding images

        :param index: the index of the data sample, passed by the data loader
        :return: (imgs, label, video_info), where imgs is a list of PIL images, label is an integer
        """
        video_info, label, n_frames = self.data[index]

        (frame_samples, centre_points) = self.generate_frame_samples(video_info)
        video_info['frame_samples'] = frame_samples
        video_info['centre_points'] = centre_points

        if self.mode == 'rgb':
            imgs = self.load_rgb_frames(video_info['video_id'], frame_samples)
        elif self.mode == 'flow':
            imgs = self.load_flow_frames(video_info['video_id'], frame_samples)
        else:
            raise Exception('Cannot deal with mode {}'.format(self.mode))

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return imgs, label, video_info

    def __len__(self):
        return len(self.data)

    def make_dataset_untrimmed(self):
        """
        split the untrimmed videos into segments of equal length which will then be passed through the network
        """
        self.data = []
        untrimmed_videos = sorted(set(self.data_frame['video_id']))
        shift = 1 if self.frames_are1_indexed else 0

        for video in untrimmed_videos:
            if video not in self.video_lengths:
                raise Exception('You did not provide the number of frames for video {}. '
                                'Check the video_lengths.csv file'.format(video))
            else:
                n_frames = self.video_lengths[video]

            start_points_step = self.softmax_stride if self.mode == 'flow' else self.n_frames_per_segment
            start_points = np.arange(0, n_frames, start_points_step, dtype=np.int32)
            n_segments = len(start_points)

            for i, sp in enumerate(start_points):
                start_frame = sp + shift
                end_frame = min(sp + self.n_frames_per_segment, n_frames)
                segment_length = end_frame - start_frame + 1
                indexes = np.arange(sp, end_frame)

                if segment_length < self.n_frames_per_segment:
                    continue  # discarding the last segment if doesn't contain enough frames

                video_info = {
                    'start_frame': start_frame,
                    'stop_frame': end_frame,
                    'index': i,
                    'video_id': video,
                    'n_segments': n_segments,
                    'n_frames': n_frames,
                    'indexes': indexes
                }

                label = np.zeros((1, 1), np.float32)  # we don't need a real label
                self.data.append((video_info, label, segment_length))

    def make_label(self, video, label_format='single_int'):
        if label_format == 'single_int':
            class_str = video['class']
            class_index = video['class_index']
            label = int(class_index)
        else:
            raise Exception('Unrecognised label label_format: '.format(label_format))

        return class_str, class_index, label

    def make_dataset(self):
        self.data = []

        for _, row in self.data_frame.iterrows():
            (class_str, class_index, label) = self.make_label(row)
            video_dict = row.to_dict()
            video_dict['g_id'] = generate_plateau_id(**video_dict)
            self.data.append((video_dict, label, self.n_samples))

        return

    def load_rgb_frames(self, video, frame_samples):
        frames = list()

        for i in frame_samples:
            frame_path = self.build_rgb_path(video, i)

            if self.model_name == 'tsn_bni':
                try:
                    img = Image.open(frame_path).convert('RGB')
                    frames.append(img)
                except Exception:
                    raise Exception('Could not read file {}'.format(frame_path))

        return frames

    def load_flow_frames(self, video, frame_samples):
        frames = list()

        for si, samples in enumerate(frame_samples):
            for ji, j in enumerate(samples):
                (x_path, y_path) = self.build_flow_paths(video, j)

                if self.model_name == 'tsn_bni':
                    try:
                        img_x = Image.open(x_path).convert('L')
                    except Exception:
                        raise Exception('Could not read file {}'.format(x_path))
                    try:
                        img_y = Image.open(y_path).convert('L')
                    except Exception:
                        raise Exception('Could not read file {}'.format(y_path))

                    frames.extend([img_x, img_y])
                else:
                    raise Exception('Cannot deal with model {} at the moment'.format(self.model_name))

        return frames

    def build_rgb_path(self, video, frame_index):
        if self.name in ['beoid', 'thumos_14']:
            path = os.path.join(self.frames_path, 'jpegs', video, 'frame' + str(frame_index).zfill(6) + '.jpg')
        else:
            raise Exception('Unrecognised dataset: {}'.format(self.name))

        return path

    def build_flow_paths(self, video, frame_index):
        if self.name in ['beoid', 'thumos_14']:
            x_path = os.path.join(self.frames_path, 'u', video, 'frame' + str(frame_index).zfill(6) + '.jpg')
            y_path = os.path.join(self.frames_path, 'v', video, 'frame' + str(frame_index).zfill(6) + '.jpg')
        else:
            raise Exception('Unrecognised dataset: {}'.format(self.name))

        return x_path, y_path

    def generate_frame_samples(self, video, samples_are0_indexed=True):
        if self.untrimmed:
            start_f = video['start_frame']
            end_f = video['stop_frame']
            t = 1

            frame_samples = np.arange(start_f, end_f + t, dtype=np.int32)

            if self.mode == 'flow':
                frame_samples = frame_samples[np.newaxis, :]

            centre_points = frame_samples
        else:
            if self.sampling_strategy == 'uniform':
                (frame_samples, centre_points) = self._sample_uniformly(video)
            elif self.sampling_strategy == 'random':
                (frame_samples, centre_points) = self._sample_randomly(video)
            elif self.sampling_strategy == 'samples_from_plateaus':
                (frame_samples, centre_points) = self._sample_from_plateaus(video)
            elif self.sampling_strategy == 'random_tsn':
                (frame_samples, centre_points) = self._sample_randomly_with_tsn(video)
            else:
                raise Exception('Unrecognised sampling strategy: {}'.format(self.sampling_strategy))

            # frame samples should always be 0-indexed, so we need to add one if frames are 1-indexed
            if self.frames_are1_indexed and samples_are0_indexed:
                frame_samples += 1

        return frame_samples, centre_points

    def _sample_from_plateaus(self, video):
        start_f = video['start_frame']
        end_f = video['stop_frame']
        video_id = video['video_id']
        segment_id = generate_plateau_id(**video)
        untrimmed_length = self.video_lengths[video_id]

        if self.mode == 'rgb':
            frame_samples = self.samples_from_plateaus[video_id][segment_id]
            frame_samples = np.array([min(untrimmed_length-2, f) for f in frame_samples], dtype=np.int32)
            centre_points = frame_samples
        elif self.mode == 'flow':
            end_f -= 1

            if end_f - start_f + 1 <= self.stack_size:
                frame_samples = np.linspace(start_f, end_f, self.stack_size, dtype=np.int32)
                frame_samples = np.repeat(frame_samples[np.newaxis, :], self.n_samples, axis=0)
                centre_points = frame_samples
            else:
                centre_points = self.samples_from_plateaus[video_id][segment_id]

                if type(centre_points) is not np.ndarray:
                    centre_points = [centre_points]  # this should be a single number, so we wrap it in a list

                frame_samples = self._generate_stacks_from_centre_points(centre_points)
        else:
            raise Exception('Cannot deal with mode {}'.format(self.mode))

        return frame_samples, centre_points

    def _sample_randomly(self, video):
        start_f = video['start_frame']
        end_f = video['stop_frame']
        h_stack_f = np.floor(self.stack_size / 2)
        h_stack_c = np.ceil(self.stack_size / 2)

        if self.mode == 'flow':
            end_f -= 1

        if self.mode == 'rgb':
            frame_samples = np.random.random_integers(start_f, high=end_f, size=(self.n_samples, 1))
            centre_points = frame_samples
        elif self.mode == 'flow':
            if end_f - start_f + 1 <= self.stack_size:
                frame_samples = np.linspace(start_f, end_f, self.stack_size, dtype=np.int32)
                frame_samples = np.repeat(frame_samples[np.newaxis, :], self.n_samples, axis=0)
                centre_points = frame_samples
            else:
                centre_points = np.random.random_integers(start_f + h_stack_c, high=end_f - h_stack_f,
                                                          size=(self.n_samples, 1))
                frame_samples = self._generate_stacks_from_centre_points(centre_points)
        else:
            raise Exception('Cannot deal with mode {}'.format(self.mode))

        return frame_samples, centre_points

    def _sample_randomly_with_tsn(self, video):
        start_f = video['start_frame']
        end_f = video['stop_frame']
        h_stack_f = np.floor(self.stack_size / 2)

        if self.mode == 'flow':
            end_f -= 1

        # TSN SAMPLING CODE STARTS HERE
        L = self.stack_size if self.mode == 'flow' else 1
        action_frames = end_f - start_f + 1

        average_duration = (action_frames - L + 1) // self.n_samples

        if average_duration > 0:
            offsets = np.multiply(list(range(self.n_samples)), average_duration) + \
                      np.random.randint(average_duration, size=self.n_samples, dtype=np.int)
        elif action_frames > self.n_samples:
            offsets = np.sort(np.random.randint(action_frames - L + 1, size=self.n_samples, dtype=np.int))
        else:
            offsets = np.zeros((self.n_samples,), dtype=np.int)
        # TSN SAMPLING CODE ENDS HERE

        if self.mode == 'flow':
            centre_points = offsets + start_f + h_stack_f
            frame_samples = self._generate_stacks_from_centre_points(centre_points)
        elif self.mode == 'rgb' or self.mode == 'idt_fv':
            frame_samples = offsets + start_f
            centre_points = frame_samples
        else:
            raise Exception('Cannot deal with mode {}'.format(self.mode))

        return frame_samples, centre_points

    def _sample_uniformly(self, video):
        start_f = video['start_frame']
        end_f = video['stop_frame']
        h_stack_f = np.floor(self.stack_size / 2)
        h_stack_c = np.ceil(self.stack_size / 2)

        # evaluate all frames if n_samples is non positive
        n_samples = self.n_samples if self.n_samples > 0 else end_f - start_f + 1

        if self.mode == 'rgb':
            frame_samples = np.linspace(start_f, end_f, n_samples, dtype=np.int32)
            centre_points = frame_samples
        elif self.mode == 'flow':
            end_f -= 1
            centre_points = np.linspace(start_f + h_stack_c, end_f - h_stack_f, n_samples, dtype=np.int32)
            frame_samples = self._generate_stacks_from_centre_points(centre_points)
        else:
            raise Exception('Cannot deal with mode {}'.format(self.mode))

        return frame_samples, centre_points

    def _generate_stacks_from_centre_points(self, centre_points, override_n_samples=None):
        h_stack_f = np.floor(self.stack_size / 2)
        h_stack_c = np.ceil(self.stack_size / 2)

        n_samples = self.n_samples if override_n_samples is None else override_n_samples

        frame_samples = np.zeros((n_samples, self.stack_size), dtype=np.int32)

        for ip, cp in enumerate(centre_points):
            frame_samples[ip, :] = np.arange(cp - h_stack_f, cp + h_stack_c, dtype=np.int32)

        return frame_samples

    def set_class_labels(self, class_labels):
        self.class_labels = class_labels

    def set_class_map(self, class_map):
        self.class_map = class_map

    def set_plateaus_samples(self, samples_from_plateaus):
        self.samples_from_plateaus = samples_from_plateaus
