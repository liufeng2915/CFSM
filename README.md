# **Controllable and Guided Face Synthesis for Unconstrained Face Recognition**

European Conference on Computer Vision (ECCV 2022). [[Arxiv](https://arxiv.org/abs/2207.10180), [PDF](http://cvlab.cse.msu.edu/pdfs/Liu_Kim_Jain_Liu_ECCV2022.pdf), [Supp](http://cvlab.cse.msu.edu/pdfs/Liu_Kim_Jain_Liu_ECCV2022_supp.pdf), [Project](http://cvlab.cse.msu.edu/project-cfsm.html)]

**[Feng Liu](http://cvlab.cse.msu.edu/pages/people.html), [Minchul Kim](https://scholar.google.com/citations?user=8tOJ80IAAAAJ&hl=en), [Anil Jain](https://www.cse.msu.edu/~jain/),  [Xiaoming Liu](http://cvlab.cse.msu.edu/pages/people.html)**

We propose a controllable face synthesis model (CFSM) that can mimic the distribution of target datasets in a style latent space. CFSM learns a linear subspace with orthogonal bases in the style latent space with precise control over the diversity and degree of synthesis. Furthermore, the pre-trained synthesis model can be guided by the face recognition (FR) model, making the resulting images more beneficial for FR model training. Besides, target dataset distributions are characterized by the learned orthogonal bases, which can be utilized to measure the distributional similarity among face datasets. Our approach yields significant performance gains on unconstrained benchmarks, such as IJB-B, IJB-C, TinyFace and IJB-S (+5.76% Rank1).

## Introduction 

This paper aims to answer the following three questions:

* Can we learn a face synthesis model that can discover the styles in the target unconstrained data, which enables us to precisely control and increase the diversity of the labeled training samples?
* Can we incorporate the feedback provided by the FR model in generating synthetic training data, towards facilitating FR model training?
*  Additionally, as a by-product of our proposed style based synthesis, can we model the distribution of a target dataset, so that it allows us to quantify the distributional similarity among face datasets?

## Prerequisites

This code is developed with

* Python 3.7
* Pytorch 1.8
* Cuda 11.1 

## Stage1: Controllable Face Synthesis Model (CFSM)

Please refer to [CFSM/README](CFSM/README.md) for the details.

## Stage2: Guided Face Synthesis for Face Recognition 

Please refer to [GuidedFaceRecognition/README](GuidedFaceRecognition/README.md) for the details.

## Citation

```bash
@inproceedings{liu2022cfsm,
title={Controllable and Guided Face Synthesis for Unconstrained Face Recognition},
author={Liu, Feng and Kim, Minchul and Jain, Anil and Liu, Xiaoming},
booktitle={ECCV},
year={2022}}
```

## Acknowledgments

Here are some great resources we benefit from:

* [MUNIT](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/munit) for the multimodal face image translation network.
* [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) and [AdaFace](https://github.com/mk-minchul/AdaFace) for the face recognition module. 
* [advertorch](https://github.com/BorealisAI/advertorch) and [RobustAdversarialNetwork](https://github.com/DengpanFu/RobustAdversarialNetwork) for the adversarial regularization.

## License

[MIT License](LICENSE)

## Contact

For questions feel free to post here or drop an email to - liufeng6@msu.edu
