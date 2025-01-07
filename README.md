# Theory of Mind (ToM) for Large Language Model (LLM) Alignment

This implementation builds upon the [LatentQA](https://github.com/aypan17/latentqa/tree/main) repository, with modifications to enable reading additional datasets for training Theory of Mind (ToM) in LLMs.

## Dataset Preparation

This section details how each dataset is prepared for training:

### CaSiNo Dataset

The CaSiNo dataset is used directly without any changes from the original paper's repository: [link](https://github.com/kushalchawla/CaSiNo).

### CraigslistBargain Dataset

The CraigslistBargain dataset is retrieved from the webpage associated with the paper: [link](https://stanfordnlp.github.io/cocoa/).

### FanToM Dataset

The FanToM dataset is downloaded from the link provided in the paper's repository: [link](https://github.com/skywalker023/fantom/tree/main). This link points to a zip file hosted on Google Drive. After downloading, the dataset is divided into training, validation, and test sets using the `train_test_split` function from the `sklearn.model_selection` library. The random state is set to `42` for reproducibility. Here's the breakdown of the split:

- **Test Set**: 30% of the data is reserved for unseen testing.
- **Train and Validation Sets**: The remaining 70% of the data is further split into training and validation sets with an 80:20 ratio.

### Negotiation ToM Dataset

Similar to the FanToM dataset, the Negotiation ToM dataset is downloaded from the paper's repository: [link](https://github.com/HKUST-KnowComp/NegotiationToM) and processed as follows:

1. Download the dataset.
2. Split the data into training, validation, and test sets using the same procedure as the FanToM dataset.

## Training New Decoder Models

To train a new model, you need to specify the desired parameters in the `run_train.sh` and `train_config.py` files. All results are reproducible by setting the following in the configuration files:

- **LLM Name**: The name of the large language model you want to train (e.g., `mistralai/Ministral-8B-Instruct-2410`).
- **Dataset**: The dataset to use for training. Choose from `CaSiNo`, `CraigslistBargain`, `FanToM`, or `NegotiationToM`.
- **Output Location**: The location where the final decoder model will be stored (defined in the configuration file).

**Note:** Currently supported LLM models include `mistralai/Ministral-8B-Instruct-2410`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, and `meta-llama/Llama-3.2-1B-Instruct`.

