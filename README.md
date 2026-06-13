# Network Intrusion Detection using Machine Learning

## Overview

This project implements a Machine Learning-based Network Intrusion Detection System (NIDS) to identify malicious network traffic and detect cyberattacks.

The system uses the CICIDS2017 dataset and applies Machine Learning models to classify network traffic into benign or malicious categories.

## Objectives

* Detect malicious network traffic
* Compare ML models for intrusion detection
* Improve cybersecurity monitoring using AI
* Evaluate model performance using metrics

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* VS Code

## Dataset

Dataset Used: CICIDS2017

File Used:
`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

The dataset contains network traffic information with attack and benign labels.

## Machine Learning Models Used

### 1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees for accurate classification.

### 2. XGBoost

XGBoost is an optimized gradient boosting algorithm that improves prediction performance.

## Workflow

1. Load dataset
2. Clean missing/infinite values
3. Encode labels
4. Split data into training and testing
5. Train ML models
6. Compare model accuracy
7. Generate classification report
8. Save trained model

## Results

The project successfully trained and evaluated:

* Random Forest
* XGBoost

The generated accuracy comparison graph shows model performance.

## Folder Structure

```text
network-intrusion-detection-ml/
│
├── data/
├── src/
├── outputs/
├── models/
├── screenshots/
├── README.md
├── requirements.txt
└── .gitignore
```

## Future Scope

* Real-time network packet monitoring
* Deep learning-based intrusion detection
* Explainable AI for attack analysis
* Web dashboard for monitoring

## Conclusion

This project demonstrates how Machine Learning can improve cybersecurity by identifying suspicious network behavior and helping detect cyberattacks efficiently.
