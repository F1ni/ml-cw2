# Analysis of TypiClust on CIFAR-10: Active Learning in the Low-Budget Regime

## Outline
* **Self-Supervised Embeddings:** Implements SimCLR code provided by Van Gansbeke.
* **Active Learning Baselines:** Compares TypiClust against Random sampling strategies and against a fully supervised framework (ResNet from scratch).
* **Adaptive K-Means:** Used Adaptive K method to improve accuracy results of TypiClust.

## Prerequisites
1. Clone Repository
   ```bash
   git clone https://github.com/F1ni/ml-cw2.git
   cd ml-cw2

2. Install Dependencies
   ```bash
   pip install -r requirements.txt

## Usuage
### Training SimCLR model (SimCLR code)
```bash
cd UnsupervisedClassification
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
```
This is to train the SimCLR model over 500 epochs. This has already been trained and the model is stored in embeddings/model.pth.tar. 

For convenience, the extracted features and cleaned model weights are **already included in this repository**:
* `cifar10_simclr_embeddings.pt`: The normalised 512-dimensional feature arrays and labels used by TypiClust to calculate K-Means clusters.
* `cifar10_simclr_cleaned.pth`: The cleaned SimCLR encoder weights (with the projection head removed)

These features and weights were extracted using 'extract_features.ipynb'.

### main.ipynb
Main.ipynb is the code to reproduce the results from the original paper.

Run all cells to see the output results that are present in the report. Results may differ slightly due to randomness.

Results are shown in the reports.
