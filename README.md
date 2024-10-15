# Lens

> The official implementation for our paper *Lens: Rethinking Multilingual Enhancement for Large Language Models*.

<!-- <img src="https://img.shields.io/badge/Venue-ACL--24-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red"> -->

## Requirements
* Python 3.9.0
* PyTorch 2.4.1
* Transformers 4.42.3
* CUDA 12.1

## Training

For Language Subspace Probing, run `scripts/run_probing.sh` to obtain the language-agnostic and -specific subspaces for the backbone model.

For Language Subspace Probing, run `scripts/run_manipulation.sh` to start the multilingual enhancement training.  

## Evaluation

Codes for multilingual evaluation are placed under `NLU_eval` folder.


## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{zhao2024lens,
  title={Lens: Rethinking Multilingual Enhancement for Large Language Models},
  author={Zhao, Weixiang and Hu, Yulin and Guo, Jiahe and Sui, Xingyu and Wu, Tongtong and Deng, Yang and Zhao, Yanyan and Qin, Bing and Che, Wanxiang and Liu, Ting},
  journal={arXiv preprint arXiv:2410.04407},
  year={2024}
}
```
