# Context R-CNN

An implementation of [Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection](https://arxiv.org/abs/1912.03538) on top of the [Detectron2](https://github.com/facebookresearch/detectron2) object detection library. 

This is in progress, and currently supports the long-term network only, and one contextual proposal per image. PRs are welcome.

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
  "rights_holder" : str,    
  "seq_id": str,
  "seq_num_frames": int,
  "frame_num": int
}
```

All other fields are the same as standard COCO. 

Then, add your dataset info to context_rcnn.data.\_DATASETS in the format provided. All paths are relative to your data_dir when running scripts (e.g. context_rcnn/data):

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

Add your model for feature extraction to scripts.preprocess.\_MODELS in the format provided. All paths are relative to your models_dir when running scripts.

```
model_name: {
        "weights": "path/to/model.pth",
        "config": "path/to/config.yaml"
    },
```

Then, use scripts/preprocess.py to generate memory banks, e.g.:

```
python scripts/preprocess.py --model model_name --dataset dataset_name --banks-dir path/to/output/ --data-dir ./data --models-dir ./models --num-gpus 1
```

This will write a 2 pickle files for each location in your dataset to banks-dir, one containing features extracted from unmodified images and one containing features extracted from horizontally-flipped images.

## Training

To train with default configuration options:

```
python scripts/train.py --model context-rcnn-fpn --data-dir ./data --banks path/to/banks_dir --dataset dataset_name --num-gpus 8
```

See configs directory for other training configuration options.

## Model Zoo

Coming soon.

# References
[Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection (Beery et al 2020)](https://arxiv.org/abs/1912.03538)

[Official Tensorflow implementation](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Detectron2 (Wu et al 2019)](https://github.com/facebookresearch/detectron2)
