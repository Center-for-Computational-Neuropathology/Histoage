# HistoAge

This repository contains the code associated with the published academic research article titled **"Histopathologic brain age estimation via multiple instance learning"**.

[Link to article](https://pubmed.ncbi.nlm.nih.gov/37815677/)

## Abstract

Understanding age acceleration, the discordance between biological and chronological age, in the brain can reveal mechanistic insights into normal physiology as well as elucidate pathological determinants of age-related functional decline and identify early disease changes in the context of Alzheimer’s and other disorders. Histopathological whole slide images provide a wealth of pathologic data on the cellular level that can be leveraged to build deep learning models to assess age acceleration. Here, we used a collection of digitized human post-mortem hippocampal sections to develop a histological brain age estimation model. Our model predicted brain age within a mean absolute error of 5.45 ± 0.22 years, with attention weights corresponding to neuroanatomical regions vulnerable to age-related changes. We found that histopathologic brain age acceleration had significant associations with clinical and pathologic outcomes that were not found with epigenetic-based measures. Our results indicate that histopathologic brain age is a powerful, independent metric for understanding factors that contribute to brain aging.

## Repository Overview

This repository allows users to:

- **Train your own model**: Utilize the provided scripts to train a histopathologic brain age estimation model.
- **Evaluate the model**: Assess the performance of the trained model using the evaluation scripts.
- **Generate attention maps**: Create attention maps to interpret the model's focus during age prediction.

## Preprocessing and Data Format

Before using this repository, each digitized whole slide image must be segmented into tiles, with the top-left coordinate position of each tile saved, and each tile must be extracted into feature vectors. For each slide, these data should be saved in an `.h5` file containing two objects:

- `coords`: A `k x 2` numpy array where each row is the `(x, y)` slide-level coordinate of the top-left pixel of each tile.
- `features`: A `k x feature_dimension` torch tensor where each row is the feature vector for each tile.

We recommend using the scripts provided in the [CLAM repository](https://github.com/mahmoodlab/CLAM/):

- Use `create_patches_fp.py` to segment the whole slide images into tiles.
- Use `extract_features_fp.py` to extract the feature vectors for each tile.

### Directory Structure

Create a directory for each dataset. Each dataset directory should contain two subdirectories: `raw` and `processed`. The `raw` subdirectory should include a CSV file named `h5files_age.csv`, which has two columns: `h5file` and `AGE`. Each row corresponds to a case in the dataset, with `h5file` being the path to the `.h5` file containing the slide's data and `AGE` being the associated age (an integer or float).

## Installation

To install the required dependencies, please run:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, use the `train_histoage.py` script. This script provides several options to customize the training process, including specifying the data path, model architecture, and training parameters.

### Basic Usage

The script can be run with default parameters as shown in the example below:

```bash
python train_histoage.py --data /path/to/data --dump_path . --network trans --hdim 256 --nheads 2 --learning 1e-4 --reg 1e-5 --dropout 0.25 --optim adamw --batch 8 --split 0.90 --n_splits 50 --n_epochs 100
```

### Key Arguments

* **Data and Output Paths**:
  * `--data` (`-d`): Path to the directory containing your dataset, formatted in the torch geometric format.
  * `--dump_path`: Directory where experiment results and logs will be saved. By default, this is the current directory (`.`).

* **Model Configuration**:
  * `--network` (`-net`): Choose the model architecture. Options are `trans` (transformer-based) or `set_mean`. The default is `trans`.
  * `--hdim` (`-hd`): Specify the hidden layer dimension size. Default is `256`.
  * `--nheads` (`-nh`): Number of attention heads for multiheaded attention. Default is `2`.
  * Other options such as `--mab_fc` and `--layernorm` can be toggled to apply additional layers and normalization techniques.

* **Regularization and Optimization**:
  * `--learning` (`-lr`): Learning rate for the optimizer. Default is `1e-4`.
  * `--reg` (`-r`): Regularization strength. Default is `1e-5`.
  * `--dropout` (`-do`): Dropout rate to avoid overfitting. Default is `0.25`.
  * `--optim` (`-opt`): Choose the optimizer. Options are `adam`, `adamw`, and `sgd`. Default is `adamw`.

* **Training Parameters**:
  * `--split`: Fraction of the dataset to be used for training. The rest will be used for validation/testing. Default is `0.90`.
  * `--n_splits`: Number of data splits to run. Default is `50`.
  * `--n_epochs`: Number of training epochs per split. Default is `100`.
  * Additional parameters such as `--stratify` can be used to ensure that training and test sets have similar distributions of ages.

### Example
If you wanted to train the model using a specific dataset and save the results to a custom directory, you might run:

```bash
python train_histoage.py --data /my/data/path --dump_path /my/results/path --network trans --hdim 512 --nheads 4 --learning 5e-5 --dropout 0.3 --optim adam --batch 16 --split 0.85 --n_splits 100 --n_epochs 150
```
This command will train a transformer-based model with a hidden dimension of 512, using 4 attention heads, a learning rate of 5e-5, and a dropout rate of 0.3. The results will be saved in `/my/results/path.`

For a full list of arguments and options, you can view the script's help documentation:
```bash python train_histoage.py --help ```

## Evaluating the Model

To evaluate the model, use the `eval_histoage.py` script. This script requires specifying the data path and the model checkpoint to evaluate. You can also customize options for the output directory, model training arguments, and the device to run the evaluation on.

### Basic Usage

The script can be run with default parameters by specifying the required data and checkpoint paths. Here is an example of running the evaluation script with all default values:

```bash
python eval_histoage.py --data /path/to/data --checkpoint /path/to/checkpoint --outputdir /path/to/output --argpath /path/to/args.json --device cpu 
```

### Key Arguments

* **Necessary Arguments**:
  * `--data` (`-d`): Path to the directory containing your dataset, formatted in the torch geometric format. This argument is required.
  * `--checkpoint` (`-ckp`): Path to the model checkpoint you wish to evaluate. This argument is required.

* **Optional Arguments**:
  * `--outputdir` (`-o`): Path to the output directory where evaluation results will be stored. If not provided, the script will create an output directory in the same parent directory as the checkpoint file.
  * `--argpath` (`-a`): Path to a JSON file containing the model's training options. If not specified, the script will assume the correct `args.json` file is located in the parent directory of the checkpoint file.
  * `--device`: Device on which to run the evaluation. This can be either `"cuda"` (to use a GPU) or `"cpu"` (to run locally). The default is `"cpu"`.

### Example

If you wanted to evaluate the model with a specific dataset and checkpoint, while saving the results to a custom directory and using a GPU for evaluation, you might run the following:

```bash
python eval_histoage.py --data /my/data/path --checkpoint /my/checkpoint/path --outputdir /my/output/path --argpath /my/args/path/args.json --device cuda
```

This command will evaluate the model using the specified dataset, checkpoint, and arguments, and it will save the results to `/my/output/path` while running on a GPU (cuda).

For a full list of arguments and their descriptions, you can view the script's help documentation:
```bash
python eval_histoage.py --help
```

## Creating Attention Heatmaps

To generate attention heatmaps for the model's predictions, use the `make_attn_map.py` script. This script requires a model checkpoint, an H5 file with coordinates and feature vectors, and the whole slide image (WSI) file. Additionally, several options allow you to customize the output and visualization parameters.

### Basic Usage

The script can be run with default parameters by specifying the necessary checkpoint, H5 file, and WSI file paths. Here is an example of running the attention map script with default values:

```bash
python make_attn_map.py --checkpoint /path/to/checkpoint --h5path /path/to/h5file --wsipath /path/to/slide --datapath /path/to/datafile --outputdir /path/to/output --outputbasename output --argpath /path/to/args.json --downsample 50 --tile_size 256 --sigma 1.25 --alpha 0.5
```

### Key Arguments

* **Necessary Arguments**:
  * `--checkpoint` (`-ckp`): Path to the model checkpoint to extract attention from. This argument is required.
  * `--h5path` (`-h5`): Path to the `.h5` file that contains the coordinates array and feature vectors. This argument is required.
  * `--wsipath` (`-s`): Path to the whole slide image (WSI) file, which must be compatible with OpenSlide. This argument is required.

* **Optional Arguments**:
  * `--datapath` (`-d`): Path to the torch geometric data file. If not provided, the script will create one.
  * `--outputdir` (`-o`): Path to the directory where the output will be saved. If not provided, an output directory will be created in the same parent directory as the checkpoint.
  * `--outputbasename` (`-b`): Basename to use for the output filenames. If not provided, the script will use the basename of the H5 file.
  * `--argpath` (`-a`): Path to a JSON file with the model's training options. If not provided, the script assumes `args.json` is located in the parent directory of the checkpoint.

* **Optional Parameters**:
  * `--downsample` (`-ds`): Downsampling factor for the WSI resolution. Default is `50`.
  * `--tile_size` (`-ts`): Size of each tile in pixels (height/width). Default is `256`.
  * `--sigma` (`-sig`): Standard deviation of the Gaussian kernel used to smooth the attention map. This can make the map visually more appealing. Default is `1.25`.
  * `--alpha` (`-alph`): Alpha value for overlaying the attention map onto the base slide image, in the range of `(0, 1)`. Default is `0.5`.


For a full list of arguments and their descriptions, you can view the script's help documentation:
```bash
python make_attn_map.py --help
```


## Citation

If you use this code or our findings in your research, please cite our [article](https://pubmed.ncbi.nlm.nih.gov/37815677/):

Marx GA, Kauffman J, McKenzie AT, Koenigsberg DG, McMillan CT, Morgello S, Karlovich E, Insausti R, Richardson TE, Walker JM, White CL 3rd, Babrowicz BM, Shen L, McKee AC, Stein TD; PART Working Group; Farrell K, Crary JF. Histopathologic brain age estimation via multiple instance learning. *Acta Neuropathol*. 2023 Dec;146(6):785-802. doi: [10.1007/s00401-023-02636-3](https://doi.org/10.1007/s00401-023-02636-3). Epub 2023 Oct 10. PMID: 37815677; PMCID: PMC10627911.


```bibtex
@article{marx2023histopathologic,
  title={Histopathologic brain age estimation via multiple instance learning},
  author={Marx, Gregory A and Kauffman, Jill and McKenzie, Alexander T and Koenigsberg, Drew G and McMillan, Corey T and Morgello, Susan and Karlovich, Emily and Insausti, Ricardo and Richardson, Todd E and Walker, James M and White, CL and Babrowicz, Brian M and Shen, Li and McKee, Ann C and Stein, Thor D and PART Working Group and Farrell, Katherine and Crary, John F},
  journal={Acta Neuropathologica},
  volume={146},
  number={6},
  pages={785--802},
  year={2023},
  publisher={Springer},
  doi={10.1007/s00401-023-02636-3},
  pmid={37815677},
  pmcid={PMC10627911}
}
```