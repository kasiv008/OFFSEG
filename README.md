# OFFSEG for Culvert Degradation
Official implementation of "OFFSEG: A Semantic segmentation framework for Offroad-Driving"\
For paper click [here](https://arxiv.org/abs/2103.12417). For sample video on RELLIS-3d dataset click [here](https://drive.google.com/drive/folders/1r7wsQMBsgJOwPNnP0I8DHEjxDEocsMGj?usp=sharing)

## OFFSEG test on NYCC Inspection Report



<img width="1626" height="1176" alt="Screenshot from 2025-07-23 10-37-43" src="https://github.com/user-attachments/assets/872f014b-1460-464f-9c40-e1e5123ff134" />

Segmantation folder includes both BiSeNetV2 and HRNETV2+OCR used for training. The instructions for re-training is same as that of the respective main repositories.
The pipline is configured for RUGD dataset in the repository. 

Download pre-trained weights [here](https://drive.google.com/drive/folders/1v9xzKUjP-9ydOSIMFAOy4fAUMRcpo1r-?usp=sharing).

The pretrained weights of RUGD dataset gave robust results from tests on custom dataset.\

  1. TODO: Integrate the instance segmentations with LLM to generate reports.

##### I you find our work useful for your research, please do cite us:
```latex
@INPROCEEDINGS{9551643,
  author={Viswanath, Kasi and Singh, Kartikeya and Jiang, Peng and Sujit, P.B. and Saripalli, Srikanth},
  booktitle={2021 IEEE 17th International Conference on Automation Science and Engineering (CASE)}, 
  title={OFFSEG: A Semantic Segmentation Framework For Off-Road Driving}, 
  year={2021},
  volume={},
  number={},
  pages={354-359},
  doi={10.1109/CASE49439.2021.9551643}}
```
The repository is stll under updation. Stay tuned.


