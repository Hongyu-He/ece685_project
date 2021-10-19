# ECE685 Project: Collaborative Learning of Medical Imaging Models

This is the repository for ECE685 project: Collaborative Learning of Medical Imaging Models.

- [ECE685 Project: Collaborative Learning of Medical Imaging Models](#ece685-project-collaborative-learning-of-medical-imaging-models)
  - [Week Process](#week-process)
  - [Introduction](#introduction)
  - [Project Goal](#project-goal)
  - [Dataset](#dataset)
    - [How to download this dataset](#how-to-download-this-dataset)
  - [References](#references)

## Week Process

- [ ] Download datasets
- [ ] Configure your experiments environments
- [ ] Get familiar with the dataset, draft the dataset part of the report
- [ ] Pickup a CNN model, implement it to perform binary classification on the dataset

## Introduction

One of the main challenge in training medical image models is the lack of large, well labelled datasets, since it requires radiologists to perform strenuous labelling work and to prepare the dataset for training.

Federated Learning enables collaborative training of a global model by aggregating training results from multiple sites without directly sharing datasets, hence also ensures patient privacy.

## Project Goal

- **Overall goal:** Explore ways to improve federated learning model accuracy.
- Literature review;
- Find a medical dataset to work with;
- Figure out how to prepare data for non-iid and iid scenarios for each client;
- Propose how to improve model accuracy for non-iid and iid scenarios;
- Discuss security of federated learning model and how to prevent leakage of patient information or data;

## Dataset

Dataset used in this project: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

### How to download this dataset

For downloading the dataset to your local machine, just visit the link above and click download.

For downloading the dataset to your remote server (or download with command line), you can use [kaggle-api](https://github.com/Kaggle/kaggle-api). 

After downloading the kaggle-api, use `kaggle datasets download --unzip paultimothymooney/chest-xray-pneumonia` to download the dataset to your desired place.

## References

- [A pytorch implementation of federated learning](https://github.com/AshwinRJ/Federated-Learning-PyTorch)
- [Data Poisoning Attacks Against Federated Learning Systems](https://github.com/git-disl/DataPoisoning_FL)
- [Pytorch Image Classification](https://github.com/bentrevett/pytorch-image-classification)