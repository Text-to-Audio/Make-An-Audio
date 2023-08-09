# Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models

#### Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, Zhou Zhao

PyTorch Implementation of [Make-An-Audio (ICML'23)](https://arxiv.org/abs/2301.12661): a conditional diffusion probabilistic model capable of generating high fidelity audio efficiently from X modality.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2301.12661)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio)
[![GitHub Stars](https://img.shields.io/github/stars/Text-to-Audio/Make-An-Audio?style=social)](https://github.com/Text-to-Audio/Make-An-Audio)

We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://text-to-audio.github.io/) for audio samples.

[Text-to-Audio HuggingFace Space](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio) | [Audio Inpainting HuggingFace Space](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio_inpaint)

## News
- Jan, 2023: **[Make-An-Audio](https://arxiv.org/abs/2207.06389)** submitted to arxiv.
- August, 2023: **[Make-An-Audio](https://arxiv.org/abs/2301.12661) (ICML 2022)** released in Github. 

## Quick Started
We provide an example of how you can generate high-fidelity samples using Make-An-Audio.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.


### Support Datasets and Pretrained Models

Simply run following command to download the weights from [Google drive](https://drive.google.com/drive/folders/1zZTI3-nHrUIywKFqwxlFO6PjB66JA8jI?usp=drive_link).
```
Download:
    maa1_full.ckpt and put it into ./useful_ckpts  
    BigVGAN vocoder and put it into ./useful_ckpts  
```
The directory structure should be:
```
useful_ckpts/
├── bigvgan
│   ├── args.yml
│   └── best_netG.pt
└── maa1_full.ckpt
```


### Dependencies
See requirements in `requirement.txt`:

## Inference with pretrained model
```bash
python gen_wav.py --prompt "a bird chirps" --ddim_steps 100 --duration 10 --scale 3 --n_samples 1 --save_name "results"
```


# TODO
- [ ] Release Training code
- [ ] Release Evaluation code
- [ ] Make Make-An-Audio available on Diffuser

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[CLAP](https://github.com/LAION-AI/CLAP),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion),
as described in our code.

## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
@article{huang2023make,
  title={Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models},
  author={Huang, Rongjie and Huang, Jiawei and Yang, Dongchao and Ren, Yi and Liu, Luping and Li, Mingze and Ye, Zhenhui and Liu, Jinglin and Yin, Xiang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.12661},
  year={2023}
}
```

# Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
