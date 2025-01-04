# Theory of Mind (ToM) for LLM Alignment  

This implementation is adapted from the [LatentQA](https://github.com/aypan17/latentqa/tree/main) repository, with minimal modifications to accommodate reading additional datasets.  

## Dataset Preparation  

### CaSiNo Dataset  
The CaSiNo dataset is directly copied without modifications from its original paper's [repository](https://github.com/kushalchawla/CaSiNo).  

### CraigslistBargain Dataset  
The CraigslistBargain dataset is sourced from the associated paper's [webpage](https://stanfordnlp.github.io/cocoa/).  

### FanToM Dataset  
The FanToM dataset is obtained from the link provided in its paper's [repository](https://github.com/skywalker023/fantom/tree/main), which points to a zip file hosted on Google Drive. After downloading, the dataset is split into `Train`, `Validation`, and `Test` sets using the `train_test_split()` function from the `sklearn.model_selection` library, with `random_state=42` to ensure reproducibility. The dataset is divided as follows:  
- **Test Set**: 30% of the data is reserved as unseen test data.  
- **Train and Validation Sets**: The remaining 70% of the data is split into training and validation sets in an 80:20 ratio.  

### Negotiation ToM Dataset  
The same procedure used for the FanToM dataset is applied here. The dataset is downloaded from the paper's [repository](https://github.com/HKUST-KnowComp/NegotiationToM) and processed accordingly.  
