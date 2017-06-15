# HSN for Remote sensing

**This repository contains project codes of following paper: "Hourglass-Shape Network Based Semantic Segmentation for High Resolution Aerial Imagery"**

**As the project is still in progress, the code for preprocessing, WBP post processing, and evaluation is currently under code review, and will be published progressively in the near future**

## Abstract
A new convolution neural network (CNN) architecture for semantic segmentation of high resolution aerial imagery is proposed in this paper. The proposedarchitecture follows an hourglass-shaped network (HSN) design being structuredinto encoding and decoding stages. By taking advantage of recent advances in CNN designs, we use the composed inception module to replace common convolutional layers, providing the network with multi-scale receptive areas withrich context. Additionally, in order to reduce spatial ambiguities in the upsampling stage, skip connections with residual units are also employed to feed forward encoding-stage information directly to the decoder. Moreover, overlap inference is employed to alleviate boundary effects occurring when high resolution images are inferred from small-sized patches. Finally, we also propose a post-processing method based on weighted belief propagation to visually enhance the classification results. Extensive experiments based on the Vaihingen and Potsdam datasets demonstrate that the proposed architectures outperform three reference state-of-the-art network designs both numerically and visually.

## HSN net Specification

## HSN example

## Dataset

## License and Citation

Please cite the following paper if you find the project helpful to your research.

	@article{liu2017hourglass,
	title={Hourglass-ShapeNetwork Based Semantic Segmentation for High Resolution Aerial Imagery},
	author={Liu, Yu and Minh Nguyen, Duc and Deligiannis, Nikos and Ding, Wenrui and Munteanu, Adrian},
	journal={Remote Sensing},
	volume={9},
	number={6},
	pages={522},
	year={2017},
	publisher={Multidisciplinary Digital Publishing Institute}
	}

This code is shared under a Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) Creative Commons licensing. 

You are free to:
Share — copy and redistribute the material in any medium or format
Adapt — remix, transform, and build upon the material
The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial — You may not use the material for commercial purposes.
No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Acknowlegement
The research has been supported by Fonds Wetenschappelijk Onderzoek (project no.
G084117), Brussels Institute for Research and Innovation (project 3DLicornea) and Vrije Universiteit Brussel
(PhD bursary Duc Minh Nguyen).
