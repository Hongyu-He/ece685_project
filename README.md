# ECE685 Project: Collaborative Learning of Medical Imaging Models

This is the repository for ECE685 project: Collaborative Learning of Medical Imaging Models.

[Overleaf project](https://www.overleaf.com/4377316726bvykzpfydnwj)

## Authors

Yuxuan Chen, Hongyu He, Wei Wu

## Contents

- [ECE685 Project: Collaborative Learning of Medical Imaging Models](#ece685-project-collaborative-learning-of-medical-imaging-models)
  - [Authors](#authors)
  - [Contents](#contents)
  - [Questions](#questions)
  - [TO-DO](#to-do)
  - [Introduction](#introduction)
  - [Project Goal](#project-goal)
  - [Dataset](#dataset)
    - [imagenette](#imagenette)
    - [Other options](#other-options)
  - [References](#references)

## Questions

No questions for now.

## TO-DO

- [ ] Download datasets: ImageNet
- [ ] Configure your experiments environments
- [ ] Implement dataset: 
  - [ ] sample data from dataset and plot
  - [ ] implement non-iid sampling
- [ ] Summarize dataset: 
  - [ ] total numbers
  - [ ] number of positive samples
  - [ ] number of negative samples
  - [ ] train : test : val
- [x] implement MLP model
- [ ] Implement more complexed models (CNN with 2 conv layers, VGG, ResNet etc.) Try some more sota image classification models.
- [ ] compare running time of FL & non-FL
- [ ] implement baseline_main.py
- [ ] implement federated_main.py
- [ ] Research sota federated learning algorithms to improve performance.

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

### imagenette

Github repository of [imagenette](https://github.com/fastai/imagenette).

Imagenette is a subset of 10 easily classified classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

### Other options

- [Collection of textures in colorectal cancer histology](https://zenodo.org/record/53169#.YXWTFy-B1hE): Details about this datasets [Multi-class texture analysis in colorectal cancer histology](https://www.nature.com/articles/srep27988)

## References

- Codes reference:
  - [A pytorch implementation of federated learning](https://github.com/AshwinRJ/Federated-Learning-PyTorch)
  - [Data Poisoning Attacks Against Federated Learning Systems](https://github.com/git-disl/DataPoisoning_FL)
  - [Pytorch Image Classification](https://github.com/bentrevett/pytorch-image-classification)
- Privacy issues of federated learning:
  - [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning](https://arxiv.org/abs/1702.07464)