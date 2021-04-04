# Context R-CNN

An implementation of [Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection](https://arxiv.org/abs/1912.03538) on top of the [Detectron2](https://github.com/facebookresearch/detectron2) object detection library. 

This implementation takes advantage of many features offered out of the box with Detectron2: multiple backbone architecture options (e.g. C4, FPN), easy setup using COCO files, easy distributed training, Tensorboard logging, Pytorch native mixed precision training, etc.

## Environment Setup

Detectron2 0.3, Pytorch 1.6, and CUDA 10.1 were used during development. PRs welcome which handle new versions, especially of Detectron2. For an environment that definitely works -

```
conda create -n contextrcnn python=3.7
conda activate contextrcnn
pip install -r requirements.txt -f \ 
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html -f \
  https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

## Dataset Format and Setup

Data is expected in COCO Camera Traps format. Your json files should have the standard high-level COCO fields:
```
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation]
}
```

For each image in the "images" list, add datetime, seq_id, seq_num_frames, and location:
```
image
{
  # Required
  "id" : str,
  "file_name" : str,
  "width" : int,
  "height" : int,
  
  # Optional
  "location": str,
  "date_captured": str, # format: YYYY-mm-dd HH:MM:SS   
  "seq_id": str,
  "seq_num_frames": int,
  "frame_num": int,
  "rights_holder" : str, 
}
```

All other fields are the same as standard COCO. 

Then, add your dataset info to ```context_rcnn.data._DATASETS``` in the format provided. All paths are relative to your data_dir when running scripts (e.g. ```./data```):

```
dataset_name: {
        "imgs_loc": "path/to/image/directory/",
        "labels_loc": "path/to/labels/directory/",
        
        subset0_name: {
            "train_filename": str,
            "val_filename": str,
            "num_classes": int
        },
        
        subset1_name: {
            ...
        }
}
```

## Preprocessing (Generate memory banks)

To train a Context R-CNN, you must first generate memory banks of extracted features for your dataset using an existing Faster R-CNN model. This can be, for example, a COCO-pretrained model or a model you have previously trained.

To do this, add info about your model to ```scripts.preprocess._MODELS``` in the format provided. All paths are relative to your models_dir when running scripts.

```
model_name: {
        "weights": "path/to/model.pth",
        "config": "path/to/config.yaml"
    },
```

Then, use ```scripts/preprocess.py``` to generate memory banks and write them to disk. This can be run in a distributed fashion on multiple GPUs.

```
python scripts/preprocess.py --model model_name --dataset dataset_name --banks-dir path/to/output/ --data-dir ./data --models-dir ./models --num-gpus 1
```

This will write a 2 pickle files for each location in your dataset to ```banks-dir```, one containing features extracted from unmodified images and one containing features extracted from horizontally-flipped images.

## Training

An example training script is included in scripts/train.py. The default options:
- use a batch size of 2 ims/gpu, or 4 ims/gpu if mixed precision training is enabled with ```--amp```
- use a default LR of .005 and batch size of 16 (2 ims/gpu * 8 gpus), with automatic linear learning rate scaling according to the batch size
- train for 18 epochs, reducing learning rate by a factor of 10 after epochs 12 and 16
- evaluate once per epoch (and use a custom visualizer which shows top attention weights at each epoch of training)
- can send Tensorboard output (including visualizer) to Weights & Biases if wandb is installed and ```--wandb your_project_name``` is used

To train with default configuration options:

```
python scripts/train.py --model context-rcnn-fpn --data-dir ./data --banks path/to/banks_dir --dataset dataset_name --num-gpus 8
```

Run ```python scripts/train.py -h``` for all training options, and see configs directory for other configuration options included in Detectron2.

## Model Zoo

Coming soon.

## TODO

- [ ] Short-term network
- [ ] Extract multiple regions per image for memory banks

# References
[Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection](https://arxiv.org/abs/1912.03538) Sara Beery, Guanhang Wu, Vivek Rathod, Ronny Votel, Jonathan Huang

[Official Context R-CNN Implementation (Tensorflow)](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Detectron2 (Wu et al 2019)](https://github.com/facebookresearch/detectron2) Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick
