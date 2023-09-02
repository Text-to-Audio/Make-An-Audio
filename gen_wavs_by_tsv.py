import os,sys
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import soundfile
device = 'cuda' # change to 'cpu‘ if you do not have gpu. generating with cpu is very slow.
SAMPLE_RATE = 16000

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tsv_path",
        type=str,
        help="the tsv contains name and caption"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help="the directory contains wavs"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="audio duration, seconds",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=3.0, # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    ) 

    parser.add_argument(
        "--save_name",
        type=str,
        default='test', 
        help="audio path name for saving",
    ) 
    return parser.parse_args()

def initialize_model(config, ckpt,device=device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device,device,model.cond_stage_model.device)
    sampler = DDIMSampler(model)

    return sampler

def dur_to_size(duration):
    latent_width = int(duration * 7.8)
    if latent_width % 4 != 0:
        latent_width = (latent_width // 4) * 4
    return latent_width

def build_name2caption(tsv_path):
    df = pd.read_csv(tsv_path,sep='\t')
    name2cap = {}
    name_count = {}
    for t in df.itertuples():
        name = getattr(t,'name')
        caption = getattr(t,'caption')
        if name not in name_count:
            name_count[name] = 0 
        else:
            name_count[name]+=1
        name2cap[name+f'_{name_count[name]}'] = caption

    return name2cap

def gen_wav(sampler,vocoder,prompt,ddim_steps,scale,duration,n_samples):
    latent_width = dur_to_size(duration)
    start_code = torch.randn(n_samples, sampler.model.first_stage_model.embed_dim, 10, latent_width).to(device=device, dtype=torch.float32)
    
    uc = None
    if scale != 1.0:
        uc = sampler.model.get_learned_conditioning(n_samples * [""])
    c = sampler.model.get_learned_conditioning(n_samples * [prompt])
    shape = [sampler.model.first_stage_model.embed_dim, 10, latent_width]  # 10 is latent height 
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        x_T=start_code)

    x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)

    wav_list = []
    for idx,spec in enumerate(x_samples_ddim):
        wav = vocoder.vocode(spec)
        if len(wav) < SAMPLE_RATE * duration:
            wav = np.pad(wav,SAMPLE_RATE*duration-len(wav),mode='constant',constant_values=0)
        wav_list.append(wav)
    return wav_list

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)

    sampler = initialize_model('configs/text_to_audio/txt2audio_args.yaml', 'useful_ckpts/maa1_full.ckpt')
    vocoder = VocoderBigVGAN('useful_ckpts/bigvnat',device=device)
    print("Generating audios, it may takes a long time depending on your gpu performance")
    name2cap = build_name2caption(args.tsv_path)
    result = {'caption':[],'audio_path':[]}
    for name,caption in tqdm(name2cap.items()):
        wav_list = gen_wav(sampler,vocoder,prompt=caption,ddim_steps=args.ddim_steps,scale=args.scale,duration=args.duration,n_samples=1)
        for idx,wav in enumerate(wav_list):
            audio_path = f'{save_dir}/{name}.wav'
            soundfile.write(audio_path,wav,samplerate=SAMPLE_RATE)
            result['caption'].append(caption)
            result['audio_path'].append(audio_path)
    result = pd.DataFrame(result)
    result.to_csv(f'{save_dir}/result.tsv',sep='\t',index=False)