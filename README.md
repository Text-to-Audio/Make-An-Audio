# Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2301.12661)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio)

## [Paper](https://arxiv.org/abs/2301.12661) | [Hugging Face Space](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio) | [Project Page](https://text-to-audio.github.io/)

We provide our implementation and pretrained models as open source in this repository.
This repo currently support Text-to-Audio Generation.

You can use the huggingface space [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio) to quickly experience Make-An-Audio's text-to-audio generation capabilities.

And audio inpainting are provided here[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio_inpaint). 

## Pretrained checkpoints 

We provide model checkpoint trained on Audiocaps Dataset and model checkpoint trained on Full dataset in [Google drive](https://drive.google.com/drive/folders/1zZTI3-nHrUIywKFqwxlFO6PjB66JA8jI?usp=drive_link).
We also provide our BigVGAN checkpoint there.

Download:
maa1_full.ckpt and put it into ./useful_ckpts  
BigVGAN vocoder and put it into ./useful_ckpts  
The directory structure should be:
```
useful_ckpts/
├── bigvgan
│   ├── args.yml
│   └── best_netG.pt
└── maa1_full.ckpt
```


## Run the model
After downloading the pretrained models, Run:
```
python gen_wav.py --prompt "a bird chirps" --ddim_steps 100 --duration 10 --scale 3 --n_samples 1 --save_name "save_audio"
```

# TODO
- [ ] Release Training code
- [ ] Release Evaluation code
- [ ] Make Make-An-Audio available on Diffuser
## Cite this work

If you found this tool useful, please consider citing
```bibtex
@misc{huang2023makeanaudio,
      title={Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models}, 
      author={Rongjie Huang and Jiawei Huang and Dongchao Yang and Yi Ren and Luping Liu and Mingze Li and Zhenhui Ye and Jinglin Liu and Xiang Yin and Zhou Zhao},
      year={2023},
      eprint={2301.12661},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```