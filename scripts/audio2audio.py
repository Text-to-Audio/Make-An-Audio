"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
import librosa
# from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import math
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from vocoder.bigvgan.models import VocoderBigVGAN
# from ldm.data.extract_mel_spectrogram import TRANSFORMS_22050,TRANSFORMS_16000
from preprocess.NAT_mel import MelNet
import soundfile

batch_max_length = 624
SAMPLE_RATE= 16000

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=True):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_audio(path,transform,sr=16000,batch_max_length=624):# load wav and return mel
    wav,_ = librosa.load(path,sr=sr)

    audio = transform(wav) # (1,melbins,T)
    if audio.shape[2] <= batch_max_length:
        n_repeat = math.ceil((batch_max_length + 1) / audio.shape[1])
        audio = audio.repeat(1,1, n_repeat)

    audio = audio[..., :batch_max_length].unsqueeze(0) # shape [1,1,80,batch_max_length]
    return audio

def load_img(path):# load mel
    audio = np.load(path)
    if audio.shape[1] <= batch_max_length:
        n_repeat = math.ceil((batch_max_length + 1) / audio.shape[1])
        audio = np.tile(audio, reps=(1, n_repeat))

    audio = audio[:, :batch_max_length]
    audio = torch.FloatTensor(audio)[None, None, :, :] # [1,1,80,batch_max_length]
    return audio

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a bird chirping",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-audio",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/audio2audio-samples"
    )


    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-v",
        "--vocoder_ckpt",
        type=str,
        help="resume from vocoder checkpoint",
        default="",
    )
    return parser.parse_args()

def main():
    opt = parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    hparams = {
        'audio_sample_rate': SAMPLE_RATE,
        'audio_num_mel_bins':80,
        'fft_size': 1024,
        'win_size': 1024,
        'hop_size': 256,
        'fmin': 0,
        'fmax': 8000,
        'batch_max_length': 1248, 
        'mode': 'pad', # pad,none,
    }
    melnet = MelNet(hparams)
    sampler = DDIMSampler(model)
    vocoder = VocoderBigVGAN(opt.vocoder_ckpt,device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples # 一个prompt产生n_samples个结果
    if not opt.from_file: # load prompts from this file
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    assert os.path.isfile(opt.init_audio)
    init_image = load_audio(opt.init_audio,transform=melnet).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0: # default=5.0
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device)) # [B, channel, c, h]
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=uc,)

                    x_samples = model.decode_first_stage(samples)
                    print(x_samples.shape)
                    for x_sample in x_samples:
                        spec = x_sample[0].cpu().numpy()
                        spec_ori = init_image[0][0].cpu().numpy()
                        print(x_sample.shape,spec.shape,init_image.shape)
                        wav = vocoder.vocode(spec)
                        wav_ori = vocoder.vocode(spec_ori)
                        soundfile.write(os.path.join(outpath, f'{prompt.replace(" ", "-")}.wav'), wav, SAMPLE_RATE, 'FLOAT')
                        soundfile.write(os.path.join(outpath, f'{prompt.replace(" ", "-")}_ori.wav'), wav_ori, SAMPLE_RATE, 'FLOAT')
                        base_count += 1
                    all_samples.append(x_samples)


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
