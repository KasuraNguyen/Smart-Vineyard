[Uploading README.md…]()
# Smart-Vineyard

A data mining project for grapevine leaf disease classification and the analysis of disease distribution and environmental associations using the GVLiD (GrapeVine Leaf Identification of the Disease) dataset.

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology and Workflow](#methodology-and-workflow)
- [Task 1 -- Grapevine Leaf Disease Classification](#task-1----grapevine-leaf-disease-classification)
- [Task 2 -- Disease Distribution and Environmental Associations](#task-2----disease-distribution-and-environmental-associations)
- [Experimental Results](#experimental-results)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [External Image Prediction](#external-image-prediction)
- [Outputs](#outputs)
- [Limitations and Notes](#limitations-and-notes)
- [References](#references)

## Overview

Smart-Vineyard explores how data mining and machine learning can support grapevine health analysis. The project is organized around two complementary tasks:

1. **Image-based disease classification:** classify grapevine leaf images into four categories.
2. **Metadata analysis:** investigate disease distribution across vineyards and explore statistical associations between disease labels, environmental attributes, and geographic location.

The workflow covers exploratory data analysis (EDA), image-to-metadata matching, data quality checks, duplicate detection, preprocessing, feature extraction, model training, evaluation, and statistical analysis.

## Project Objectives

- Explore the structure, quality, and distribution of the GVLiD dataset.
- Match grapevine leaf images with their corresponding metadata records.
- Detect exact and near-duplicate images and address conflicting labels before model evaluation.
- Build and evaluate traditional machine learning and deep learning classifiers.
- Analyze whether disease patterns differ across vineyards or environmental conditions.
- Report findings with appropriate statistical tests and acknowledge uncertainty and dataset limitations.

## Dataset

**Dataset:** GVLiD -- GrapeVine Leaf Identification of the Disease  
**Source:** [Mendeley Data -- GVLiD](https://data.mendeley.com/datasets/wkymf8bhcg/3)

The dataset contains grapevine leaf images and metadata. The project works with four image classes:

| Label | Description |
|---|---|
| `Black Rot` | Black rot |
| `Esca` | Esca |
| `Healthy` | Healthy leaf |
| `Leaf Blight` | Leaf blight |

The metadata includes disease labels, vineyard identifiers, GPS coordinates, weather, lighting, time of capture, and device-related fields.

> **Dataset availability:** The original image dataset is not included in this repository archive. Download it from the source above and place it in the directory expected by the notebooks. The notebooks currently contain local Windows paths that must be updated for your machine.

Expected dataset location used in the notebooks:

```text
Smart-Vineyard/
└── Dataset/
    └── GVLiD GrapeVine Leaf identification of the Disease/
        └── DATASET/
            ├── metadata.json
            ├── Black rot/
            ├── Esca/
            ├── Healthy/
            └── Leaf blight/
```

Check the downloaded dataset's actual folder and class names before running the notebooks. If they differ, update the corresponding paths and folder-name handling in the notebook configuration cells.

## Project Structure

```text
Smart-Vineyard/
│
├── Dataset/
│   ├── DataSource.txt
│   └── GVLiD GrapeVine Leaf identification of the Disease/
│       └── DATASET/                         # Download separately
│
├── Notebook/
│   ├── 01_EDA.ipynb
│   ├── 02_DataMatching.ipynb
│   ├── 03_Image_Preprocessing.ipynb
│   ├── Task01_Grape_Leaf_Disease_Classification.ipynb
│   └── Task02_Analysis of Disease Distribution and Environmental Associations.ipynb
│
├── outputs/
│   ├── bai1/                                # Task 1 models, metrics, tables and figures
│   │   ├── ml_models/
│   │   ├── best_resnet18_gvlid.pt
│   │   └── result files (*.png, *.csv)
│   │
│   ├── bai2/                                # Task 2 analysis figures
│   │   └── result files (*.png)
│   │
│   └── processed/
│       ├── image_metadata.json
│       ├── metadata_without_image.csv
│       ├── skipped_files.csv
│       └── step03_clean/
│           ├── preprocessing_config.json
│           ├── removed_label_conflicts.csv
│           ├── task1_handcrafted_features.csv
│           ├── task1_images_split.csv
│           ├── task2_env_dedup.csv
│           └── task2_env_full.csv
│
├── test_images/                             # Images for inference/testing
├── predict_ml.py                            # Exported by Task 1 notebook (cell 8.1)
├── CommandToRunML.txt                       # Traditional ML command examples
├── .gitignore
└── README.md
```

Some generated files and folders may not exist until the relevant notebook cells have been executed. In particular, `predict_ml.py` is generated by the Task 1 notebook's model-export section; it may therefore be absent from a fresh checkout or an archive created before that cell was run.

## Methodology and Workflow

Run the notebooks in the following order:

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `01_EDA.ipynb` | Explore metadata, class distribution, data quality, image counts, GPS, and initial duplicate patterns. |
| 2 | `02_DataMatching.ipynb` | Match image files with metadata records and export `image_metadata.json`. |
| 3 | `03_Image_Preprocessing.ipynb` | Detect exact/near duplicates, handle conflicting labels, prepare clean datasets, split data, and extract handcrafted features. |
| 4 | `Task01_Grape_Leaf_Disease_Classification.ipynb` | Train and evaluate traditional ML models and ResNet18 for image classification. |
| 5 | `Task02_Analysis of Disease Distribution and Environmental Associations.ipynb` | Analyze disease distribution, vineyard groups, environmental associations, and metadata-based prediction. |

### Data preparation pipeline

1. **EDA:** inspect metadata fields, image counts, class distributions, missing values, invalid records, and initial data quality.
2. **Data matching:** pair image paths with metadata records and save the matched records as JSON.
3. **Exact duplicate detection:** use MD5 hashes to identify byte-identical images and investigate label conflicts.
4. **Near-duplicate detection:** use perceptual hashing (dHash), then verify candidate pairs with image-pixel comparison (MSE).
5. **Dataset construction:** create separate data versions for the image-classification task and metadata analysis.
6. **Train/validation/test split:** create stratified 70%/15%/15% splits after duplicate handling and check for duplicate-group leakage.
7. **Feature extraction:** calculate handcrafted color, texture, and shape features for traditional ML.
8. **Modeling and analysis:** run the two task-specific notebooks using the prepared outputs.

## Task 1 -- Grapevine Leaf Disease Classification

### Problem statement

Given a grapevine leaf image, predict one of four classes:

- `Black Rot`
- `Esca`
- `Healthy`
- `Leaf Blight`

### Traditional machine learning

The preprocessing notebook extracts **25 handcrafted image features**:

| Feature group | Method | Number of features |
|---|---|---:|
| Color | HSV channel mean, standard deviation, and skewness | 9 |
| Texture | GLCM properties: contrast, dissimilarity, homogeneity, energy, correlation, and ASM (mean and standard deviation) | 12 |
| Shape | First four Hu Moments calculated from an Otsu binary mask | 4 |
| **Total** | | **25** |

The classification notebook compares several traditional models, including:

- Dummy Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM, RBF kernel)
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoosting
- Multi-layer Perceptron (MLP)

The notebook uses stratified 5-fold cross-validation and a separate hold-out test set. Because the classes are imbalanced, Macro-F1, Balanced Accuracy, and per-class metrics are important alongside ordinary Accuracy.

### Deep learning

The deep learning branch uses **ResNet18 transfer learning** on the original leaf images.

The notebook includes:
- Image resizing to 224 × 224.
- ImageNet normalization.
- Training-time augmentation, including ColorJitter, flips, and rotations.
- Class-weighted CrossEntropyLoss using square-root-balanced weights.
- AdamW optimization and a cosine learning-rate schedule.
- Validation Macro-F1 for checkpoint selection.
- Final evaluation on the hold-out test set.

The saved checkpoint is:

```text
outputs/bai1/best_resnet18_gvlid.pt
```

### Task 1 reported results

The notebook reports the following hold-out test results for one recorded run:

| Model | Evaluation | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|
| SVM RBF (25 handcrafted features) | Hold-out test, 308 images | 0.9545 | 0.8591 | 0.8546 |
| ResNet18 + augmentation | Hold-out test, 308 images | 0.9773 | 0.9280 | 0.929 |

The notebook also records a previous ResNet18 run with a higher Macro-F1 (0.973). This variation is explicitly noted in the experiment discussion; the result should not be treated as a stable estimate from repeated-seed experiments.

The traditional ML notebook reports SVM 5-fold Macro-F1 of approximately **0.844 ± 0.029**. The model comparison and detailed per-class metrics are saved in the Task 1 output files.

## Task 2 -- Disease Distribution and Environmental Associations

### Problem statement

Analyze whether disease labels and disease prevalence are associated with vineyard identity, weather, lighting, time of capture, or geographic proximity.

The notebook uses several complementary analyses:

| Analysis objective | Methods |
|---|---|
| Compare disease prevalence across vineyards | Disease proportions, Wilson confidence intervals, binomial tests, chi-square tests, FDR correction |
| Check the stability of vineyard rankings | Spearman correlation and random split-half stability analysis |
| Explore groups of vineyards with similar disease profiles | K-Means, hierarchical clustering (Ward), silhouette analysis, permutation tests |
| Examine geographic similarity | GPS distance and Mantel test |
| Explore environmental/disease patterns | Association rules using support, confidence, and lift; Fisher's exact test and FDR correction |
| Test whether metadata predicts disease class | Cross-validation with baseline and machine learning models; permutation testing |

### Important interpretation

The Task 2 notebook distinguishes association from causation. Its statistical analyses do **not** establish that an environmental factor causes a disease.

The notebook's recorded findings include:
- No vineyard had a disease rate significantly different from the overall rate after FDR correction (0/28).
- No vineyard-by-disease cell remained significant after FDR correction (0/112).
- The observed vineyard clustering did not provide convincing evidence of real clusters under the notebook's permutation tests.
- The Mantel analyses did not find a significant positive relationship between geographic proximity and similarity of disease profiles.
- None of the examined environmental attributes showed a statistically significant association with disease class across the tested data versions.
- Metadata-based classifiers did not outperform the majority-class baseline in a meaningful way.

These findings should be interpreted in light of the dataset limitations described below, especially the duplicate-image and metadata-consistency issues.

## Experimental Results

### Dataset after preprocessing

The preprocessing notebook records the following final data sizes:

| Data version | Number of records/images | Intended use |
|---|---:|---|
| Task 1 cleaned image dataset | 2,047 unique images | Image classification |
| Task 2 Full | 2,904 consistent records, before deduplication | Metadata analysis / sensitivity comparison |
| Task 2 Dedup | 2,047 records, one per retained image | Main deduplicated metadata analysis |

Task 1 class distribution after preprocessing:

| Class | Images | Approx. proportion |
|---|---:|---:|
| Healthy | 1,092 | 53.4% |
| Esca | 707 | 34.5% |
| Leaf Blight | 152 | 7.4% |
| Black Rot | 96 | 4.7% |
| **Total** | **2,047** | **100%** |

The split recorded in the notebook is:

| Split | Images |
|---|---:|
| Train | 1,432 |
| Validation | 307 |
| Test | 308 |
| **Total** | **2,047** |

### Generated result files

Task 1 and Task 2 figures and tables are stored under `outputs/bai1/` and `outputs/bai2/`. The processed datasets and configuration files are stored under `outputs/processed/step03_clean/`.

## Requirements

The notebooks use Python and the following libraries (the exact installed versions are not pinned in the project files):

- Python
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- Seaborn
- Pillow
- OpenCV (`opencv-python`)
- SciPy
- scikit-learn
- scikit-image
- PyTorch
- torchvision
- IPython

Some notebooks may use additional utilities depending on the cells executed. Install the packages in a virtual environment and resolve any missing imports reported by the notebook.

Example installation:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn pillow opencv-python scipy scikit-learn scikit-image jupyter
```

Install PyTorch and torchvision using the command appropriate for your operating system and CUDA setup. Refer to the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Setup

1. Clone or download this repository.
2. Download the GVLiD dataset from [Mendeley Data](https://data.mendeley.com/datasets/wkymf8bhcg/3).
3. Extract the dataset into the expected `Dataset/` location.
4. Open the notebooks and update `PROJECT_DIR` and `DATASET_DIR` to match your local paths.
5. Install the required Python packages.
6. Start Jupyter from the project root:

```bash
jupyter notebook
```

> **Path configuration:** Several notebooks currently use a hard-coded Windows path such as `D:\DataMining26\Smart-Vineyard`. Update these path variables before running the notebooks on another computer. Run notebooks from the project root unless their configuration specifies otherwise.

## How to Run

Run the notebooks in the workflow order listed above.

### Step 1 -- EDA

Open:

```text
Notebook/01_EDA.ipynb
```

Run the notebook to inspect the source metadata and image dataset. The EDA notebook expects the downloaded GVLiD dataset to be available locally.

### Step 2 -- Data matching

Open:

```text
Notebook/02_DataMatching.ipynb
```

Run the notebook to match images and metadata. Its main output is:

```text
outputs/processed/image_metadata.json
```

### Step 3 -- Image preprocessing

Open:

```text
Notebook/03_Image_Preprocessing.ipynb
```

Run the notebook from top to bottom. It creates the cleaned datasets and configuration files in:

```text
outputs/processed/step03_clean/
```

### Step 4 -- Task 1

Open:

```text
Notebook/Task01_Grape_Leaf_Disease_Classification.ipynb
```

Run the notebook to train and evaluate traditional ML models and, when enabled in the notebook configuration, ResNet18. The notebook reads the processed files from Step 3 and saves results under `outputs/bai1/`.

### Step 5 -- Task 2

Open:

```text
Notebook/Task02_Analysis of Disease Distribution and Environmental Associations.ipynb
```

Run the notebook to perform the vineyard, environmental, clustering, association-rule, and metadata-prediction analyses. It reads the processed Task 2 files from Step 3 and saves figures under `outputs/bai2/`.

## External Image Prediction

### ResNet18 inference

The Task 1 notebook defines `predict_grape_leaf()` after loading the saved ResNet18 checkpoint. After running the relevant inference cells, use:

```python
result = predict_grape_leaf("test_images/leaf_external.jpg")
```

For multiple images:

```python
image_paths = [
    "test_images/leaf_01.jpg",
    "test_images/leaf_02.jpg",
    "test_images/leaf_03.jpg",
]

results = [
    predict_grape_leaf(path, show_chart=False)
    for path in image_paths
]
```

### Traditional ML inference

The Task 1 notebook's export section (cell 8.1) generates `predict_ml.py` and saves the selected traditional ML model as:

```text
outputs/bai1/ml_models/best_traditional_ml.joblib
```

After generating the script, run it from the project root:

```bash
python predict_ml.py "test_images/leaf_01.jpg"
```

To select a particular saved model, use the `--model` argument. Example:

```bash
python predict_ml.py "test_images/leaf_01.jpg" --model "outputs/bai1/ml_models/all_models/3_SVM_RBF_Kernel_C_3.joblib"
```

Additional model command examples are documented in `CommandToRunML.txt` when that file is present.

> **Inference caution:** softmax scores are relative model confidence scores, not guaranteed probabilities that a prediction is correct. The classifier is trained on four classes and will still return one of those classes for an unrelated image or a disease not represented in the training data. Predictions are not a substitute for expert plant-disease diagnosis.

## Outputs

| Directory | Contents |
|---|---|
| `outputs/processed/` | Matched image metadata and intermediate processing outputs |
| `outputs/processed/step03_clean/` | Cleaned Task 1 and Task 2 datasets, split information, features, and preprocessing configuration |
| `outputs/bai1/` | Traditional ML models, ResNet18 checkpoint, evaluation tables, and figures |
| `outputs/bai2/` | Task 2 analysis figures |

Large generated artifacts (for example, model checkpoints and serialized models) may not be included in every repository copy or ZIP archive.

## Limitations and Notes

1. **Class imbalance:** after duplicate handling, Black Rot and Leaf Blight are minority classes. Accuracy alone can hide poor minority-class performance; inspect Macro-F1, Balanced Accuracy, and per-class Recall/F1.
2. **Small minority-class test samples:** the hold-out test set contains only 15 Black Rot images and 23 Leaf Blight images. Per-class test metrics therefore have substantial sampling uncertainty.
3. **Duplicate and conflicting records:** the source data contains exact and near-duplicate images, including cases where visually identical or near-identical images have conflicting class labels. The preprocessing workflow removes ambiguous groups and checks for leakage across splits.
4. **Metadata reliability:** the preprocessing analysis found that many duplicate image records carried different metadata values. Consequently, Task 2 results should be interpreted cautiously; metadata may not reliably describe the capture conditions of each image.
5. **Association is not causation:** statistical relationships, where observed, do not establish causal effects of weather, lighting, time, or vineyard location on disease.
6. **Model run variability:** the notebook records variation between ResNet18 runs. Report the run configuration and avoid presenting a single run as a robust estimate without repeated-seed evaluation.
7. **Local paths:** update hard-coded paths before running on a different machine.
8. **External validation:** a small number of example predictions is illustrative only. Reliable claims about generalization require a separately sourced, sufficiently large, independently labeled external dataset.

## References

- GVLiD dataset: [GrapeVine Leaf Identification of the Disease -- Mendeley Data](https://data.mendeley.com/datasets/wkymf8bhcg/3)
- ResNet: He et al., *Deep Residual Learning for Image Recognition* (2015).
- scikit-learn documentation: https://scikit-learn.org/stable/
- PyTorch documentation: https://pytorch.org/docs/stable/
