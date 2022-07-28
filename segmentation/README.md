# Semantic Segmentation Fine-tuning
Code for training semantic segmentation models on the Pannuke and Lizard datasets.

### Requirements
- Python 3.8+
- `pip install -r requirements.txt`

### Data Preparation
#### _Pannuke_
1. Download all three folds of the Pannuke dataset from [here](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke).
2. Extract all archives into one directory (e.g. `pannuke/`).
3. To convert the `.npy` files to images, run `python scripts/process_pannuke -i path/to/pannuke`.

- __Note__: The original `masks.npy` and `types.npy` files for each fold are used for evaluation.
    - i.e. `--test_target_path fold1/masks/fold1/masks.npy --test_types_path fold1/images/types.npy`

#### _Lizard_
1. Download the patch-level Lizard dataset from [here](https://conic-challenge.grand-challenge.org/Data/) (registration required) and label files from [here](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_labels.zip).
2. Extract all archives into one directory (e.g. `lizard/`).
3. To convert the `.npy` files to images run `python scripts/process_lizard.py -i path/to/lizard`
4. To generate target files run `python scripts/create_targets_lizard.py -i path/to/processed/lizard`
    - These `.npy` files are used for evaluation (i.e. `--test_target_path targets-fold1.npy`).

### Usage
#### _Training_
- To train from scratch on Pannuke run: 
```
python train.py  --gpus 1 --precision 16  --name pannuke --model.n_classes 6 --max_epoch 100 
--data.train_fold "['data/pannuke/fold1/', 'data/pannuke/fold2']" 
--data.test_fold data/pannuke/fold3  --test pannuke 
--test_targets_path eval/targets/pannuke-f3.py --test_types_path eval/types/pannuke-f3.npy 
```
- [`configs/`](configs/) contains example configuration files and can be run with `python train.py --config path/to/config/file`.
- Run `python train.py --help` for information on all options.
- This codebase uses the [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) library and therefore `--model.arch` can be any of architectures from [here](https://github.com/qubvel/segmentation_models.pytorch#models) and `--model.encoder` can be any of the encoders from [here](https://github.com/qubvel/segmentation_models.pytorch#encoders).


#### _Mask Generation:_
- To generate pseudo-label masks for a dataset using a trained segmentation model run:
```
python scripts/generate_masks.py -w path/to/segmentation/checkpoint -i path/to/image/directory
-o path/to/output/masks/directory --n_classes <number-of-classes>
```


### Acknowledgements

Evaluation code is taken from the official [Pannuke evaluation repo](https://github.com/TissueImageAnalytics/PanNuke-metrics).
