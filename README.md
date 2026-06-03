# MV_AGE: Multi-view Active Learning for Graph Embedding

Runnable utilities for a multi-view extension of Active Learning for Graph Embedding (AGE; Cai, Zheng, and Chang, 2017) using Multi-view Graph Convolutional Networks with Attention Mechanism (MAGCN; Yao, Liang, Liang, Li, and Cao, 2022) and Multiplex PageRank (Halu, Mondragon, Panzarasa, and Bianconi, 2013). See [Core References](#core-references).

Use this method when you want strong food-item classification performance on large datasets while keeping human labeling cost low. MV_AGE prioritizes the next most useful unlabeled item by combining model uncertainty, MAGCN embedding-space density, and Multiplex PageRank centrality across ingredient and nutrient views, so each label can improve the classifier efficiently.

**MV_AGE uses the AGE active-learning scoring method, adapted for multi-view data**. The centrality term comes from Multiplex PageRank over the ingredient and nutrient graph views, and the density term comes from the network embeddings after MAGCN fuses those views.

Download or clone this repository before using MV_AGE. Run the commands below from the repository root so Python can find the local `mv_age/` source folder.

The prepared project is meant to be ready for food-item labeling. During labeling, each terminal round shows the next food item name and its ingredient list, then prompts for the correct class label. With the standard `food_names.csv` file, the `food_name` values are shown alongside the ingredient text so the item is easier to identify.


The workflow is a simple two-step process:

1. put input CSV files in the repository's `input_data/` folder
2. prepare a project, then start labeling



The labeling loop queries one item at a time, which keeps the exact single-sample active-learning behavior described in the AGE paper.

<img src="https://ar5iv.org/html/1705.05085/assets/x1.png" alt="AGE Figure 1: Framework of AGE" width="400">

*Figure: AGE active-labeling framework from Cai, Zheng, and Chang (2017), [Active Learning for Graph Embedding](https://arxiv.org/abs/1705.05085).*

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0004370222000480-gr001_lrg.jpg" alt="MAGCN Figure 1: Overall structure of MAGCN" width="500">

*Figure: MAGCN multi-view attention architecture from Yao, Liang, Liang, Li, and Cao (2022), [Multi-view graph convolutional networks with attention mechanism](https://doi.org/10.1016/j.artint.2022.103708).*

## Requirements

- Python 3.10 or 3.11
- MATLAB available from the command line
- the Python dependencies from `requirements.txt`

Python 3.12 is not currently advertised because the TensorFlow 2.15 runtime used by the exact MAGCN backend does not provide compatible wheels across the supported target platforms. On Windows, the runtime depends on `tensorflow-intel`; on Linux and macOS it depends on `tensorflow`.

The default MATLAB command is `matlab`.
If needed, set a different command with:

```bash
set MATLAB_CMD=matlab
```

or pass `--matlab-cmd` to the `prepare` command.

`--matlab-cmd` names the MATLAB executable or cluster wrapper command. Use it when the default `matlab` command is not the right command for your environment.

## What Users Provide

Users provide four CSV files:

1. `ingredients.csv`
   Required columns:
   - `item_id`
   - `ingredient_list`

2. `nutrients.csv`
   Required columns:
   - `item_id`
   - one or more numeric nutrient columns

3. `initial_labels.csv`
   Required columns:
   - `item_id`
   - `label`

4. `food_names.csv`
   Required columns:
   - `item_id`
   - `food_name`

Notes:

- `initial_labels.csv` may contain only the initially labeled subset, or all rows with blanks for unlabeled items.
- zero-label starts are not supported; the exact AGE/MAGCN workflow needs starting labels.
- the starting labels should include at least one example for every class MV_AGE should use.
- `food_names.csv` is required for project preparation and is used for display during labeling. The model still encodes the ingredient text.
- optionally, provide a `full_labels.csv` with `item_id` and `true_label` to report prediction accuracy during evaluation or demo runs
- the easiest workflow is to put those files in the included `input_data/` folder using the standard filenames above, then run `prepare --input-dir input_data`
- the default sentence encoder is `all-MiniLM-L6-v2`
- the default sentence-encoder device is `auto`; it uses `cuda` when PyTorch can see a GPU, otherwise `cpu`
- the code creates the ingredient embeddings and graph files during `prepare`
- later labels must come from the same class set that appears in the starting labels

## Quick Start

### 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### GPU dependencies for sentence embeddings

GPU setup is for the sentence-embedding step that encodes ingredient text. On Linux clusters, PyTorch and TensorFlow may need extra environment-specific setup. To verify both frameworks after install:

```bash
python - <<'PY'
import torch, tensorflow as tf
print("torch cuda available:", torch.cuda.is_available())
print("tf gpus:", tf.config.list_physical_devices("GPU"))
PY
```

To require GPU for sentence embeddings, add `--device cuda` to the `prepare` command. That fails fast with a clear error if PyTorch cannot access CUDA. To force CPU, pass `--device cpu`.

`--device` controls only the sentence-transformers encoder used to embed ingredient text. The default is `auto`, which uses `cuda` when PyTorch can see a GPU and otherwise uses `cpu`.

If `torch cuda available` is `False` or `import torch` fails with a CUDA or NCCL error, reinstall a matching PyTorch build for the machine using the official commands from:
https://docs.pytorch.org/get-started/previous-versions/

`tensorflow 2.15.x` also requires `numpy < 2.0`, and the requirements files pin that automatically. If the environment already has `numpy 2.x`, reinstall the dependencies so pip can bring `numpy` back into a TensorFlow-compatible range.

### 2. Prepare input CSV files

Add your CSV files to the included `input_data/` folder:

```bash
input_data/
  ingredients.csv
  nutrients.csv
  initial_labels.csv
  food_names.csv
  full_labels.csv  # optional
```

Those files must follow the required columns listed above.

### 3. Prepare the project

```bash
python -m mv_age prepare --input-dir input_data --project my_project
```

Arguments:

- `--input-dir input_data` tells MV_AGE to read the standard CSV filenames from the `input_data/` folder.
- `--project my_project` tells MV_AGE where to save the prepared project files.

This step:

- encodes ingredient lists with `all-MiniLM-L6-v2`
- prepares the nutrient feature matrix
- builds the two exact kNN graph views
- runs MATLAB Multiplex PageRank
- saves the project folder

### 4. Start labeling

```bash
python -m mv_age label --project my_project --rounds 3
```

Arguments:

- `--project my_project` tells MV_AGE which prepared project folder to open.
- `--rounds 3` runs three labeling rounds in this terminal session. Each round asks for one label.

Each labeling round follows the multi-view AGE workflow:

- one training epoch
- one queried item
- user enters one label
- repeat

Each round prints the allowed label options before prompting for input.

## Bundled Demo Data

For a compact 48-row demo dataset to test the CLI before using project data, use the same `prepare` command shape:

```bash
python -m mv_age prepare --input-dir demo --project example_project
python -m mv_age label --project example_project --rounds 3
```

Arguments:

- `--input-dir demo` uses the bundled demo CSV files when the `demo` folder is missing or empty.
- `--project example_project` saves the demo project in `example_project`, then opens that same project for labeling.
- `--rounds 3` runs three demo labeling rounds.

If `demo` is missing or empty, `prepare` falls back to the bundled demo csv files. If the folder contains `ingredients.csv`, `nutrients.csv`, `initial_labels.csv`, and `food_names.csv`, those are used instead. Partial folders fail with an error so the code does not silently mix user data with demo data. The same missing-folder demo fallback also works with `demo_inputs`.

The bundled demo keeps 16 starting labels, four per class, so the workflow has all classes available at initialization.

## Hide or Show Guesses

By default, the labeling screen shows:

- example predictions
- current best class guesses for each queried item
- when `full_labels.csv` was included during `prepare`, current prediction accuracy on the available truth labels

For a cleaner human-labeling screen without guesses, use:

```bash
python -m mv_age label --project my_project --rounds 3 --hide-guesses
```

Arguments:

- `--hide-guesses` hides example predictions and class-probability guesses during labeling. The food item name and ingredient list are still shown.
- `--project my_project` and `--rounds 3` have the same meanings as in the labeling command above.

## Example Category Groups

The bundled four-class sample example uses these branded food category groups:

- `cat1`
  `Pepperoni, Salami & Cold Cuts`
  `Sausages, Hotdogs & Brats`
- `cat2`
  `Canned Soup`
  `Other Soups`
- `cat3`
  `Cookies & Biscuits`
  `Crackers & Biscotti`
- `cat4`
  `Cakes, Cupcakes, Snack Cakes`
  `Croissants, Sweet Rolls, Muffins & Other Pastries`

## Multi-View AGE Configuration

`MV_AGE` scores unlabeled food items with:

- uncertainty from class-probability entropy
- density from the post-fusion MAGCN embedding space
- centrality from Multiplex PageRank over the ingredient and nutrient graph views

The package defaults are:

- Graph construction: `graph_k_view = 30`
- MAGCN hidden size: `hidden1 = 32`
- MAGCN learning rate: `learning_rate = 0.01`
- MAGCN weight decay: `weight_decay = 5e-4`
- MAGCN dropout: `dropout = 0.5`
- AGE time-sensitive schedule: `age_basef = 0.995`
- Multiplex PageRank alpha: `mpr_alpha = 0.85`
- Multiplex PageRank beta: `mpr_beta = 1.0`
- Multiplex PageRank gamma: `mpr_gamma = 1.0`

## Core References

- Cai, H., Zheng, V. W., and Chang, K. C.-C. (2017). *Active Learning for Graph Embedding*. arXiv preprint arXiv:1705.05085. https://arxiv.org/abs/1705.05085
- Yao, K., Liang, J., Liang, J., Li, M., and Cao, F. (2022). *Multi-view graph convolutional networks with attention mechanism*. Artificial Intelligence, 307, 103708. https://doi.org/10.1016/j.artint.2022.103708
- Halu, A., Mondragon, R. J., Panzarasa, P., and Bianconi, G. (2013). *Multiplex PageRank*. PLOS ONE, 8(10), e78293. https://doi.org/10.1371/journal.pone.0078293

## Development Checks

From a clean checkout, the lightweight unit suite can be run without MATLAB, TensorFlow, or sentence-transformer model downloads:

```bash
pip install numpy pandas scipy scikit-learn
python -m unittest discover -s tests -v
```

These checks are also captured in `.github/workflows/tests.yml` for the standalone repository.

## License and Vendored Code

This repository includes vendored research code for the exact backend. The repository license note uses `GPL-3.0-or-later` because the bundled Multiplex PageRank MATLAB implementation is GPL-3.0-or-later. See `THIRD_PARTY_NOTICES.md` before publishing or redistributing the code.

## What the Labeling Screen Shows

Each labeling round shows:

- current label counts
- labeled percentage
- label options that are allowed for this project
- current model accuracy, if the project was prepared with truth labels
- a few example predictions, unless `--hide-guesses` is used
- the next item to label
- the food item name
- the ingredient list text
- current class-probability guesses, unless `--hide-guesses` is used

## Saved Project Files

The project folder stores:

- `foods.csv` - merged working food table with current labels and optional truth labels
- `ingredient_embeddings.npy` - sentence-embedding matrix for the ingredient text
- `nutrient_features.npy` - numeric nutrient feature matrix
- `x_dense.npy` - combined ingredient and nutrient features used by MAGCN
- `ingredient_graph.npz` - sparse kNN graph built from ingredient embeddings
- `nutrient_graph.npz` - sparse kNN graph built from nutrient features
- `centrality_values.npy` - Multiplex PageRank centrality score for each item
- `metadata.json` - project configuration, column names, class names, and AGE settings
- `state.json` - labeling progress, checkpoint round, and random-number generator state
- `query_history.csv` - log of queried items, entered labels, and AGE scoring values
- `predictions.csv` - most recent prediction and query-score table
- `checkpoints/` - TensorFlow checkpoint files used to resume the model state
