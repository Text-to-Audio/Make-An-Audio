import torch
from audioldm_eval import EvaluationHelper
import argparse

device = torch.device(f"cuda:{0}")

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_wavsdir',type=str)
    parser.add_argument('--gt_wavsdir',type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    generation_result_path = args.pred_wavsdir
    target_audio_path = args.gt_wavsdir

    evaluator = EvaluationHelper(16000, device)

    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
    )
