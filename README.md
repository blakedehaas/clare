# CLARE: Binned Classification Neural Network for Electron Temperature Prediction in the Plasmasphere

We present CLARE, the first machine learning model tailored to predict electron temperatures in Earth’s plasmasphere, covering altitudes from 1,000 to 8,000 km, by integrating geospatial and temporal solar indices.

CLARE is an 84-million-parameter neural network that uses a hybrid classification-regression architecture. This approach discretizes the continuous output space into bins to enhance prediction accuracy while naturally embedding uncertainty estimates.

**Key Features:**

*   Predicts electron temperature (Te) in the plasmasphere (1,000 - 8,000 km altitude).
*   Utilizes in-situ geospatial data from the Akebono satellite and solar activity indices (Kp, AL, SYM-H, F10.7) as inputs.
*   Employs a novel binned classification approach for improved accuracy and uncertainty quantification.
*   Achieves predictions within 10% absolute deviation for 69.82% of observations under typical solar conditions.
*   Demonstrates an accuracy of 21.39% on a  solar storm test set.

**Paper:** For a detailed description of the model architecture, dataset, and results, please refer to our paper:
[CLARE: Binned Classification Neural Network for
Electron Temperature Prediction (Google Docs Link)](https://docs.google.com/document/d/17t7eBduGdQoqOX6EXzHKLkKA3nxHLFlVrFWtiy-d-cA/edit?usp=sharing)

---

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [Installation](#installation)
*   [Data Preparation](#data-preparation)
*   [Training](#training)
*   [Evaluation](#evaluation)
*   [License](#license)

---

## Prerequisites

Before you begin, set up the following:

1.  **Python:** Required for using the repository, download version 3.9 or above.
    *   [Download Python](https://www.python.org/downloads/)
2.  **(Optional) Weights & Biases Account:** Optional for experiment tracking and model management.
    *   [Sign up at Weights & Biases](https://wandb.ai/site)

---

## Installation

Follow these steps to set up the project environment:

1.  **Clone the Repository:**
    Open your terminal or command prompt and navigate to the directory where you want to store the project. Clone the repository using *either* SSH or HTTPS:

    *   **Using SSH (Recommended):**
        ```bash
        git clone git@github.com:blakedehaas/clare.git
        ```
    *   **Using HTTPS:**
        ```bash
        git clone https://github.com/blakedehaas/clare.git
        ```
    Navigate into the cloned directory:
    ```bash
    cd clare
    ```

2.  **Install Dependencies:**
    Install all the required Python packages using pip and the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Set up Weights & Biases:**
    If using Weights & Biases, log in to your Weights & Biases account from the command line. You will be prompted to enter your API key (found on your W&B profile settings page):
    ```bash
    wandb login
    ```

---

## Data Preparation

Prepare the necessary datasets for training and evaluation:

1.  **Download Input Data:**
    Download the following data files [from the data repository](https://drive.google.com/drive/folders/1WqUIcDYlR20UxPlgKlU30UZ3rHW6OfIi?usp=sharing)
    *   `Akebono_combined.tsv` (this file is currently restricted, reach out to the paper authors for access)
    *   `omni_kp_index.lst`
    *   `omni_al_index_symh.zip`
    *   `omni_f107.zip`

2.  **Place Data Files:**
    Move the downloaded files into the `clare/dataset/input_dataset/` directory.

3.  **Unzip Archives:**
    Navigate to the `clare/dataset/input_dataset/` directory and unzip the `.zip` files:
    ```bash
    cd dataset/input_dataset/
    unzip omni_al_index_symh.zip
    unzip omni_f107.zip
    ```
    After successful extraction, delete the original `.zip` files:
    ```bash
    rm omni_al_index_symh.zip omni_f107.zip
    ```

4.  **Run Dataset Creation Script:**
    Navigate back to the `clare/dataset/` directory and run the script to process the raw data and create the final datasets:
    ```bash
    cd ..  # Move up from input_dataset to dataset
    python create_dataset.py
    ```
    This script will generate the necessary processed data files used for training and evaluation, saving to the `clare/dataset/output_dataset` directory.

---

## Training

Train the CLARE model using the prepared datasets:

1.  **Configure Experiment Name:**
    *   Open the `clare/train.py` file in your editor.
    *   Manually update the `model_name` variable (line 24) to a unique identifier for your training run (e.g., `model_name = "clare_experiment1"`). This name will be used for saving checkpoints and normalization statistics.

2.  **Run Training Script:**
    *   Navigate to the top-level project directory (`clare/`).
    *   Execute the training script:
        ```bash
        python train.py
        ```
    *   Training progress will be logged to your terminal and to Weights & Biases under the project `clare`.
    *   The trained model checkpoint (`.pth` file) will be saved to `clare/checkpoints/` using the specified `model_name`.
    *   Normalization statistics (e.g., mean, std) used during training will be saved to `clare/data/` using the specified `model_name`.

---

## Evaluation

Evaluate the performance of a trained model checkpoint:

1.  **Configure Model Name for Evaluation:**
    *   Open the `clare/evaluate.py` file in your editor.
    *   Manually update the `model_name` variable (line 18) to match the exact `model_name` of the trained checkpoint you want to evaluate (the one you set in `train.py`).

2.  **Select Test Dataset for Evaluation:**
    *   Manually update the `dataset` variable (line 19) and select between `test-normal` and `test-storm`. The `test-normal` dataset consists of 50,000 randomly selected points over the entire dataset and the `test-storm` dataset consists of a continuous known solar storm period from June 2nd through 8th, 1991.
3.  **Run Evaluation Script:**
    *   Ensure you are in the top-level project directory (`clare/`).
    *   Execute the evaluation script:
        ```bash
        python evaluate.py
        ```
    *   The script will load the specified checkpoint and normalization statistics, run predictions on the test set, and print evaluation metrics. Results may also be logged to Weights & Biases if configured within the script.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.