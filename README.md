# Action Recognition from Single Timestamp Supervision in Untrimmed Videos

This repository contains code and annotations for the CVPR 19 paper:

```
Action Recognition from Single Timestamp Supervision in Untrimmed Videos
```

When using this code, please reference the paper:

```
@InProceedings{moltisanti19action, 
author = "Moltisanti, Davide and Fidler, Sanja and Damen, Dima",
title = "{A}ction {R}ecognition from {S}ingle {T}imestamp {S}upervision in {U}ntrimmed {V}ideos",
booktitle = "Computer Vision and Pattern Recognition (CVPR)",
year = "2019"
}
```
[Project webpage](http://dimadamen.github.io/single_timestamps/)

## Code Contribution

Code by [Davide Moltisanti](http://www.davidemoltisanti.com/research/)

## Copyright

Authors and Department of Computer Science. University of Bristol.

Code is published under the Creative Commons Attribution-NonCommercial 4.0 International License. This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes.

## Disclaimer

We make no claims about the stability or usability of the code provided in this repository.

We provide no warranty of any kind, and accept no liability for damages of any kind that result from the use of this code. 

## Dependencies

I am providing a Python 3 Conda environment to install all the needed libraries. 

Note that the code is based on **PyTorch 0.3.1**. 
Due to Variables been removed in version 0.4, you will need to make some changes to the code
in order to port it to PyTorch 0.4 and newer versions.

## Preparing the data

Clone the whole repository and keep the folders structure. 
This will download both annotations for [*BEOID*](http://people.cs.bris.ac.uk/~damen/BEOID/index.htm) and [*THUMOS 14*](https://www.crcv.ucf.edu/THUMOS14/download.html) and the necessary pre-trained models to run the code. 

You will then need to extract RGB and optical flow frames yourself, and specify the path where the frames are stored 
in the settings file ([see below](#YAML-settings)).

I used [this tool](https://github.com/uob-vil/gpu_flow) to extract RGB and optical flow frames. 
The tool will organise the frames as follows, creating folders `jpeg`, `u` and `v` at the specified output. 
This is the format my code expects to load the data:

- `jpegs/${video_name}/frame%06d.jpg`
- `u/${video_name}/frame%06d.jpg`
- `v/${video_name}/frame%06d.jpg`
 
Once you've extracted the frames, change the settings file to set the path where you extracted the frames.


## How to run the code

To run the code you need to call `python run.py` from the `src/` folder. This script takes one mandatory positional argument
and several optional arguments. 

The positional argument must be the path to a `YAML` configuration file, which contains all the 
necessary configuration to run the code. 

Run the code with argument `-h` to print the usage of the script and check out the optional arguments:

```bash
usage: run.py [-h] [--tag [TAG]] [--test_only] [--epoch [EPOCH]]
              [--checkpoint_path [CHECKPOINT_PATH]] [--override [OVERRIDE]]
              [--start_from_best_in [START_FROM_BEST_IN]] [--get_scores_only]
              settings_path

positional arguments:
  settings_path         path to yaml configuration file

optional arguments:
  -h, --help            show this help message and exit
  --tag [TAG]           tag to be appended to the export folder
  --test_only           only test the model
  --epoch [EPOCH]       load checkpoint saved at given epoch from the export
                        folder
  --checkpoint_path [CHECKPOINT_PATH]
                        load checkpoint from the given .pt/pth file
  --override [OVERRIDE]
                        override any parameter in the yaml file
  --start_from_best_in [START_FROM_BEST_IN]
                        start from the model with lowest training loss from
                        the given export folder. If passing argument
                        --test_only will start from the model with highest
                        top-1 accuracy in testing
  --get_scores_only     get only the softmax and logit scores for the
                        untrimmed training videos. The scores will be saved to
                        the export folder

```

## YAML settings

I am providing the setting files to run the code on the *BEOID* and *THUMOS 14* datasets.

Each entry of the YAML files is commented, so you should be able to understand how to change the 
settings according to your needs.

#### Overriding settings

The argument `override` allows you to change any property defined in the YAML files without 
editing the file. 
The argument must be passed as a string representing a valid dictionary.

For example, if you want to change the number of epochs you want to run the model for, which is defined 
in the YAML file as follows:

```yaml
n_epochs: 1000
```

you can do:

```bash
python run.py ../settings/beoid/ts.yaml --override "{'n_epochs': 500}"
```

If you want to change nested properties, e.g. the training batch size, which is defined in the YAML file as follows:

```yaml
training:
    batch_size: 64
```

you can do:

```bash
python run.py ../settings/beoid/ts.yaml --override "{'training': {'batch_size': 128}}"
```

Or, if you dare doing both:

```bash
python run.py ../settings/beoid/ts.yaml --override "{'n_epochs': 500, 'training': {'batch_size': 128}}"
```

## Exported files

The program will save a bunch of stuff as it runs. The base folder where everything will be contained is defined 
in the settings file with the property `export_base_folder`. The base folder is organised as follows:

- Main export folder: `${dataset_name}/${baseline}/tsn_bni/`
- TensorBoard logs folder: `logs/${dataset_name}/${baseline}/tsn_bni/`

Where:

- `${dataset_name}` is the dataset's name, either *beoid* or *thumos_14* in the provided settings
- `${baseline}` is the baseline, either *ts*, *ts_in_gt* or *gt*
- `tsn_bni` stands for the model name, in this case TSN with based on Batch Normalisation Inception. 

### Content of main export folder

Here you will find the following files and folders:

- `models/`: folder containing PyTorch checkpoints. Each checkpoint stores both the model and the
optimiser's state. These are used for automatically resuming your run if you want to stop the program
- `results/`: folder containing CSV files, one per tested epoch, as well as confusion matrices 
- `training_dicts/`: folder containing training information, exported only when running 
*ts* and *ts_in_gt* baselines
- `test_info.csv`: CSV file containing loss, top-1 and top-5 accuracy per tested epoch
- `train_info.csv`: CSV file containing loss, top-1 and top-5 accuracy per training epoch
- `ts.yaml`: the YAML filed used to run the program. Overridden settings will be updated accordingly in the file

#### Results CSV

The CSV files contain useful information gathered during testing, namely the following columns:

- `class`: string action label
- `class_index`: integer action label	
- `correct`: 1 if the action segment was correctly classified, 0 otherwise	
- `frame_samples`: list of frames sampled for testing	
- `predicted`: list of top-5 predicted classes 
- `start_frame`: action's start frame in the untrimmed video 
- `stop_frame`: action's end frame in the untrimmed video	
- `video_id`: untrimmed video identifier

#### `training_dicts/` content

This folder is organised in sub-folders, one per epoch, as follows:

`training_dicts/epoch_${epoch_number}/${video_id}/`

Each `training_dicts/${epoch_number}/` folder contains one folder per untrimmed training video, as well a CSV file named `summary.csv`. 
This file contains one row per plateau, with the following columns:

- `video`: untrimmed video identifier
- `index`: the order of the action in the  untrimmed video
- `id`: unique identifier of the plateau 
- `label`: integer action label
- `c`: the *c* parameter of the plateau function at the given epoch
- `w`: the *w* parameter of the plateau function at the given epoch
- `s`: the *s* parameter of the plateau function at the given epoch
- `N`: number of frames in the untrimmed video
- `gt_start`: ground truth start of the action (not used for training)
- `gt_end`: ground truth end of the action (not used for training)
- `used_for_training`: True if the plateau was used for training (i.e. it was selected with the Curriculum Learning ranking)
- `updated`: True if the plateau was updated during the given epoch
- `update_proposal`: the selected update proposal for the plateau. This is a string displaying the proposal's parameters and its confidence score

Each `training_dicts/epoch_${epoch_number}/${video_id}/` folder contains the following files:

- `${video_id}_plateaus.csv`
- `${video_id}_points.csv`
- `${video_id}_update_info.pkl` (if plateaus were updated during the given epoch)
- `{video_id}_softmax_scores.csv` (optional)

##### `${video_id}_plateaus.csv` 
CSV containing information about the video's plateaus, with columns:

- `c`: the *c* parameter of the plateau function at the given epoch
- `w`: the *w* parameter of the plateau function at the given epoch
- `s`: the *s* parameter of the plateau function at the given epoch
- `N`: number of frames in the untrimmed video
- `label`: integer action label
- `id`: unique identifier of the plateau 
- `index`: the order of the action in the  untrimmed video
- `video`: untrimmed video identifier
- `history`: the parameters' history of the plateau

##### `${video_id}_points.csv`

CSV file where row *i* contains the sampled points/frames, one per column, 
of the *i*-th plateau in the video


##### `${video_id}_update_info.pkl`

Standard Python pickle file containing a dictionary whose keys are plateau ids. Each entry contains a dictionary 
with the following keys:

- `all_proposals`: list of plateau objects representing all the potential update proposals, i.e. all the plateaus fitted to the softmax scores for the 
class corresponding to the plateau's class
- `left_contraint`: left constraint for the proposals of the updating plateau
- `right_contraint`: right constraint for the proposals of the updating plateau
- `valid_proposals`: sublist of `all_proposal` containing plateaus that respect the left/right constraints and have confidence > 0 
- `chosen_proposal`: the proposal in `valid_proposals` with highest confidence. This is the proposal that will be used to update the plateau, depending on the 
Curriculum Learning ranking. This corresponds to Equation 7 in the paper
- `updated`: True if the plateau was updated

##### `{video_id}_softmax_scores.csv`

CSV file containing softmax scores, where rows correspond to classes and columns correspond to frames. 
You can save the softmax scores setting `save_softmax_scores` to `True` in the YAML setting file.

## Baselines

The code implements two baselines:

- `ts`: trains the model using single timestamp supervision
- `gt`: trains the model using ground truth start/end times

Use the `baseline` parameter in the YAML setting to select which baseline you want to run. Testing is identical for all baselines.

When using the `ts` baseline the code uses the `TS` point set by default (see paper). If you want to 
use the `TS in GT` point set, add the following to the YAML setting file

```yaml
points_set: ts_in_gt 
```

or use the provided setting files `ts_in_gt.yaml` 

## Annotations files

The code expects the following CSV files to run:

- `test_df.csv`
- `train_df.csv`
- `train_df_ts_in_gt.csv`
- `video_lengths.csv`

##### `test_df.csv`

This should contain one row per action segment in the test set, with the following columns:

- `class`: string action label
- `class_index`: integer action label	
- `start_frame`: ground truth start of the action
- `stop_frame`: ground truth end of the action
- `video_id`: untrimmed video id

##### `train_df.csv`

This should contain one row per action segment/plateau in the train set, with the following columns:

- `class`: string action label
- `class_index`: integer action label
- `point`: single timestamp annotation for the action, in frames. This should be the annotation for the `TS` point set
- `start_frame`: ground truth start of the action, used only with `gt` baseline
- `stop_frame`: ground truth end of the action, used only with `gt` baseline
- `video_id`: untrimmed video id

##### `train_df_ts_in_gt.csv`

Same as `train_df.csv`, but with `point` being the annotation for the `TS in GT` point set. 

##### `video_lengths.csv`

This should contain one row per untrimmed training video, with 2 columns:

- `video`: training video id
- `frames`: number of RGB frames in the video

I am providing the annotations for *BEOID* and *THUMOS 14*. I cannot release the annotations for *EPIC Kitchens* since its 
test labels are still private (note that I'm one of the authors of EPIC Kitchens).

## Pre-trained models

I am providing three pre-trained models for TSN with batch-normalised Inception. 
These are stored in `src/models/` and are:

- `bn_inception.pth`: this works for both RGB and optical flow. Downloaded from [TensorFlow Model Zoo](https://github.com/yjxiong/tensorflow-model-zoo.torch/tree/9788c674f3058741a4c4128142c21877052863ee), specifically from [this link](https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth)
- `kinetics_tsn_flow.pth`: Kinetics pre-train weights for optical flow, downloaded from [here](http://yjxiong.me/others/kinetics_action/)
- `kinetics_tsn_rgb.pth`: Kinetics pre-train weights for RGB, downloaded from [here](http://yjxiong.me/others/kinetics_action/)

**Note**: I am providing these models solely for ensuring that the code runs smoothly should the above download links break in the future. 
I did not produce these models myself and I do not claim any authorship nor contribution on them. 

## Adding your dataset

Running the code on another dataset should be relatively easy. 

Once you have extracted RGB and flow frames, you need to follow the steps below. 
Let's suppose your new dataset is called `my_new_dataset`:

- create a folder named `my_new_dataset` under `annotations/`
- put the necessary annotation CSV files inside `annotations/my_new_dataset/` (see [above](#annotations-files) for format) 
- change the following properties in the YAML setting file:

```yaml
dataset_name: my_new_dataset
annotation_path: '../annotations/my_new_dataset'
frames_path: '/media/DATA/my_new_dataset/frames'
```

You then need to edit the following functions in `src/dataset.py`

- `build_rgb_path(self, video, frame_index)`
- `build_flow_paths(self, video, frame_index)`

In order to return the expected frame path for a given frame index.  
If you use the same tool I used to extract video frames ([see above](#preparing-the-data)) then you simply need to add 
`my_new_dataset` in the list of accepted datasets inside the two functions:

```python
def build_rgb_path(self, video, frame_index):
    if self.name in ['beoid', 'thumos14', 'my_new_dataset']:
        path = os.path.join(self.frames_path, 'jpegs', video, 'frame' + str(frame_index).zfill(6) + '.jpg')
    else:
        raise Exception('Unrecognised dataset: {}'.format(self.name))

    return path
    
```

You can use the `dataset`'s class fields `rgb_img_size` and `flow_img_size` to set the images' size for the network.

## Adding another model

It is also possible to use another model different from TSN with BN Inception, provided that the new model can 
produce frame-level classification scores. 
Our single timestamp supervision approach in fact does not make any specific 
assumption on the model, and only needs to receive classification scores for each frame in the untrimmed training videos.

Here are some directions for adding a new model:

- Look for the following in `src/run.py`

  ```python
  raise Exception('Unrecognised model: {}'.format(settings['model_name']))
  ```
  `model_name` is a property coming from the YAML files. This should give you a rough idea of where
  you need to implement your changes to add a new model
- Edit the `src/softmax_scores.py` file to extract softmax scores with your new model
- If necessary, change the way data is loaded in `src/dataset.py`. Functions to have a look at are:

  - `load_rgb_frames(self, video, frame_samples)`
  - `load_flow_frames(self, video, frame_samples)`
  - `generate_frame_samples(self, video, samples_are0_indexed=True)`
  
## Help

Most functions (at least, all the important ones) are documented in the code.
If you need help, please open an issue on this repository and I will try to help you as much as I can.

Any feedback is appreciated too!
