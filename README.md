# Lung Cancer Detection with CNNs

This project is a PyTorch-based lung CT image classification system with a Streamlit dashboard for model comparison, prediction demos, and Grad-CAM explainability.

It classifies CT scan images into four classes:

- `adenocarcinoma`
- `large.cell.carcinoma`
- `normal`
- `squamous.cell.carcinoma`

This project is for research, education, and demonstration only. It is not a medical diagnosis tool.

## What This Project Does

The project covers the full deep learning workflow:

- Loads CT scan images from train, validation, and test folders.
- Trains multiple CNN architectures.
- Evaluates models using medical-relevant metrics.
- Saves model weights, metrics, plots, architectures, and Grad-CAM outputs.
- Compares all saved experiments in a Streamlit dashboard.
- Allows image upload and prediction through the dashboard.
- Generates Grad-CAM heatmaps to explain model predictions.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- start.ipynb
|-- dash/
|   `-- app.py
|-- data/
|   |-- train/
|   |-- valid/
|   `-- test/
|-- dashboard_gradcam_outputs/
|-- results/
`-- src/
    |-- deployment/
    |-- evaluation/
    |-- interpretability/
    |-- loss/
    |-- model/
    |-- plot/
    |-- Save/
    |-- training/
    |-- transforms/
    `-- utils/
```

## Dataset

The dataset is organized like a standard image classification dataset:

```text
data/
|-- train/
|   |-- adenocarcinoma/
|   |-- large.cell.carcinoma/
|   |-- normal/
|   `-- squamous.cell.carcinoma/
|-- valid/
|   |-- adenocarcinoma/
|   |-- large.cell.carcinoma/
|   |-- normal/
|   `-- squamous.cell.carcinoma/
`-- test/
    |-- adenocarcinoma/
    |-- large.cell.carcinoma/
    |-- normal/
    `-- squamous.cell.carcinoma/
```

Current dataset size:

| Split | Adenocarcinoma | Large Cell Carcinoma | Normal | Squamous Cell Carcinoma | Total |
|---|---:|---:|---:|---:|---:|
| Train | 195 | 115 | 148 | 155 | 613 |
| Valid | 23 | 21 | 13 | 15 | 72 |
| Test | 120 | 51 | 54 | 90 | 315 |

## Main Dependencies

The project dependencies are listed in `requirements.txt`. They include:

- PyTorch, Torchvision, and Torchaudio
- NumPy, Pandas, Scikit-learn, and SciPy
- Matplotlib and OpenCV
- Pillow for image loading
- Streamlit for the dashboard
- Altair for dashboard charts
- Grad-CAM and TTach for explainability
- Torchview and Graphviz for architecture visualization
- Jupyter and IPython for notebook experiments

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install all project packages:

```powershell
pip install -r requirements.txt
```

Graphviz architecture rendering may also require installing the Graphviz desktop/system package and making sure its executable is available on `PATH`.

## How to Run the Dashboard

Run this from the project root:

```powershell
streamlit run dash/app.py
```

The dashboard is defined in:

```text
dash/app.py
```

It automatically scans the `results/` folder for saved experiment JSON files and builds the model comparison views from those files.

## Dashboard Pages

The Streamlit dashboard contains four main pages:

- **Compare All Models**: compares every saved experiment using metrics, rankings, and parameter counts.
- **Best Model**: selects the best model using medical-priority ranking instead of accuracy alone.
- **Individual Model**: shows detailed information for one model, including metrics, confusion matrix, classification report, architecture, curves, and deployment readiness.
- **Prediction Demo**: lets the user upload a CT image, run prediction with a saved model, view class probabilities, and generate a Grad-CAM heatmap.

## Models

Model architectures are stored in:

```text
src/model/
```

Implemented models include:

- `CNNBaseline`
- `ResNet18Scratch`
- `ResNet18Transfer`
- `VGG16Scratch`
- `MobileNetV2Scratch`
- `EfficientNetB0Scratch`
- `EfficientNetB3`
- `InceptionV3Transfer`

The prediction dashboard loads models through the registry in:

```text
src/deployment/model_loader.py
```

Any new model that should be usable in the dashboard must be added to this registry.

## Training

Training helpers are in:

```text
src/training/training.py
```

Main functions:

- `train_one_epoch(...)`
- `validate_one_epoch(...)`

The training code expects dataloader batches with this format:

```python
{
    "image": image_tensor,
    "label": label_tensor
}
```

The project also supports models with auxiliary outputs, such as Inception-style architectures. Output handling is centralized in:

```text
src/utils/model_outputs.py
```

## Image Preprocessing

Transforms are stored in:

```text
src/transforms/ct_transforms.py
```

Training transforms use:

- resize to the configured image size
- light affine augmentation
- optional horizontal flip
- tensor conversion
- normalization

Validation, test, and prediction transforms use:

- resize
- tensor conversion
- normalization

Default normalization:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## Loss Function

The project includes Focal Loss:

```text
src/loss/FocalLoss.py
```

Focal Loss helps with class imbalance by giving more attention to harder examples.

Saved training configurations may also include:

- `WeightedRandomSampler`
- balanced training
- early stopping
- learning-rate scheduling
- weight decay

## Evaluation

Evaluation utilities are stored in:

```text
src/evaluation/evaluate.py
src/evaluation/metric.py
```

The evaluation pipeline computes:

- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- classification report
- per-class recall
- minimum per-class recall
- weakest class

For this project, recall is especially important because false negatives are serious in cancer-related classification tasks. The dashboard therefore ranks models using medical-priority metrics, not only accuracy.

## Grad-CAM Explainability

Grad-CAM code is stored in:

```text
src/interpretability/gradcam.py
```

The Grad-CAM workflow:

1. Finds the final convolutional layer.
2. Computes the activation heatmap for the target class.
3. Overlays the heatmap on the CT image.
4. Saves the result as a PNG image.

Experiment Grad-CAM files are saved under model result folders, such as:

```text
results/EfficientNet/gradcam/
```

Dashboard upload Grad-CAM outputs are saved in:

```text
dashboard_gradcam_outputs/
```

## Saving Experiment Results

Result saving is handled by:

```text
src/Save/save.py
```

The saved result package can include:

- `model.pth`
- metrics JSON
- training history JSON
- accuracy and loss curves
- architecture SVG
- parameter counts
- layer information
- deployment configuration
- class names
- normalization values
- Grad-CAM image paths
- medical threshold checks

Typical result folder:

```text
results/
`-- ModelName/
    |-- model.pth
    |-- test_metrics.json
    |-- training_history.json
    |-- accuracy_curve.png
    |-- loss_curve.png
    |-- architecture.svg
    `-- gradcam/
```

## Prediction Pipeline

Prediction code is stored in:

```text
src/deployment/predict.py
src/deployment/model_loader.py
```

Prediction steps:

1. Load the selected architecture from the model registry.
2. Load saved weights from `model.pth`.
3. Resize and normalize the uploaded CT image.
4. Run the model in evaluation mode.
5. Apply softmax to produce class probabilities.
6. Return predicted class, confidence, probabilities, and the image tensor used for Grad-CAM.

Default class order:

```python
[
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]
```

## Notebook Workflow

The main experimentation notebook is:

```text
start.ipynb
```

It is used for:

- loading datasets
- building dataloaders
- selecting model architectures
- training models
- validating models
- evaluating on the test set
- saving experiment artifacts
- generating plots
- generating Grad-CAM samples

Open it with:

```powershell
jupyter notebook start.ipynb
```

## Useful Commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```powershell
streamlit run dash/app.py
```

Open the notebook:

```powershell
jupyter notebook start.ipynb
```

## Adding a New Model

1. Add the model implementation in `src/model/`.
2. Make it inherit from `torch.nn.Module`.
3. Make sure its final output has shape `[batch_size, num_classes]`.
4. Add support in `src/utils/model_outputs.py` if it returns auxiliary outputs.
5. Register it in `src/deployment/model_loader.py`.
6. Train and evaluate it from `start.ipynb`.
7. Save results using `save_evaluation_results(...)`.
8. Confirm it appears in the Streamlit dashboard.

## Important Notes

- This is a learning and research project, not a clinical product.
- The dataset is limited in size.
- Some models may perform well overall but poorly on specific cancer classes.
- Macro recall and minimum per-class recall are important metrics here.
- Grad-CAM helps inspect model attention, but it does not prove medical correctness.
- Clinical use would require larger datasets, external validation, expert review, regulatory work, and robust safety testing.

---

# Authors

 - Louay Blel
 - Ahmed Bouzidi
 - Yessine Friaa
