# Fruit Classification — Small CNN & VGG16 (Keras + PyTorch)

This project builds and compares two image classifiers — a **small custom CNN** trained from scratch and **VGG16 transfer learning** — on a 24-class subset of the [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits). The same experiment is implemented twice, once in **TensorFlow/Keras** and once in **PyTorch**, so the frameworks can be compared directly.

## Dataset

A 24-class subset of **Fruits-360** (original-size version, 2021.09.12.0), included in this repository under [`data/Images/`](data/Images/):

- **12,455 images**, **24 classes** (apple varieties, pears, cucumbers, zucchini, carrot, cabbage, eggplant)
- Images at their **original filmed size** (sizes vary — all pipelines resize to 100×100)
- Each fruit was filmed rotating on a motor; file names encode the rotation axis and frame index (`r0_103.jpg`)

## The data-leakage problem (and how it is fixed)

The official Fruits-360 split interleaves **consecutive video frames of the same fruit** across Training/Validation/Test (frames `k`/`k+2` → Training, `k+1` → Validation, `k+3` → Test). Neighbouring frames are nearly identical, so models can score almost perfectly by memorising each fruit — the usual near-100% accuracies on this dataset measure memorisation, not generalisation.

This project replaces that split with a **contiguous-block split**: for each class and rotation axis, frames are sorted by index and cut into consecutive blocks (≈70% train / 15% val / 15% test, see [`data/splits.csv`](data/splits.csv)). Only the few frames at block boundaries remain temporal neighbours across sets. One limitation cannot be fixed: each class contains a single physical object, so the test set measures generalisation to unseen *views*, not unseen *fruits*. Details and visual proof in the [EDA notebook](notebooks/01_eda.ipynb).

## Notebooks

| Notebook | Purpose |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Class distribution, image sizes, example images, demonstration of the leakage in the official split |
| [02_preprocessing](notebooks/02_preprocessing.ipynb) | Builds the leakage-aware contiguous-block split → `data/splits.csv` |
| [03_modelling_keras](notebooks/03_modelling_keras.ipynb) | Small CNN + frozen-base VGG16 with TensorFlow/Keras |
| [04_modelling_pytorch](notebooks/04_modelling_pytorch.ipynb) | Custom CNN + frozen-base VGG16 with PyTorch |

Augmentation is applied to the training set only; validation/test images are just resized and normalised. Both modelling notebooks evaluate with training curves, test accuracy, a confusion matrix, a classification report and example predictions.

## Running the notebooks

**Google Colab (recommended):** open a notebook in Colab and run it — the first cell clones this repository automatically (no Kaggle download or Drive mount needed). For the modelling notebooks switch to a GPU runtime (*Runtime → Change runtime type → GPU*).

**Locally:** `pip install -r requirements.txt`, then run the notebooks from the `notebooks/` folder. `data/splits.csv` is committed, so the modelling notebooks work without re-running `02_preprocessing`.

## Tools and libraries

- **TensorFlow/Keras** and **PyTorch / torchvision** — model building and training
- **scikit-learn** — confusion matrix and classification report
- **pandas, matplotlib, Pillow** — data handling and visualisation

## References

- Fruits-360 dataset: [Kaggle](https://www.kaggle.com/datasets/moltean/fruits) — Mihai Oltean, *Fruits 360 dataset: new research directions*, 2021 (MIT License, see [`data/Images/readme.md`](data/Images/readme.md))
