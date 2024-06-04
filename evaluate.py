import time
import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd

from flowmse.data_module import SpecsDataModule
from flowmse.odes import OTFLOW
from flowmse.model import VFModel
import pdb
import os
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver, get_black_box_solver

# GPU 2번과 3번만 사용하도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


from utils import energy_ratios, ensure_dir, print_mean_std

import pdb

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--folder_destination", type=str, help="Name of destination folder.")
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--odesolver_type", type=str, choices=("white", "black"), default="white",
                        help="Specify the sampler type")
    parser.add_argument("--odesolver", type=str,
                        default="euler", help="Predictor class for the PC sampler.")
    parser.add_argument("--reverse_starting_point", type=float, default=None, help="Starting point for the reverse SDE.")
    parser.add_argument("--reverse_end_point", type=float, default=None)
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for the ODE sampler")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for the ODE sampler")
    
    

    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    checkpoint_file = args.ckpt
    
    #please change this directory 
        #please change this directory 
    target_dir = "/data/{}/".format(
        args.folder_destination)
    
    
    #"/export/home/lay/PycharmProjects/ncsnpp/enhanced/{}/".format(args.destination_folder)

    ensure_dir(target_dir + "files/")

    # Settings
    sr = 16000
    odesolver_type = args.odesolver_type
    odesolver = args.odesolver
    N = args.N
    
    
    atol = args.atol
    rtol = args.rtol




    # Load score model
    model = VFModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    
    if args.reverse_starting_point == None:
        reverse_starting_point = model.T_rev
    else:
        reverse_starting_point = args.reverse_starting_point
        
    model.ode.T = reverse_starting_point
        
    if args.reverse_end_point == None:
        reverse_end_point = model.t_eps
    else:
        reverse_end_point = args.reverse_end_point
        
        
    # print(reverse_starting_point)
    # print(reverse_end_point)
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    




    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)

        #pdb.set_trace()        

         
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        
        if odesolver_type == "white":
            sampler = get_white_box_solver(odesolver, model.ode, model, Y.cuda(), T_rev=reverse_starting_point, t_eps=reverse_end_point,N=N)
        elif odesolver_type == "black":
            sampler = get_black_box_solver(model.ode, model, Y.cuda(),  rtol=1e-5, atol=1e-5,  T_rev=1.0, t_eps=0.03, N=30,  method='RK45', device='cuda')
        
        else:
            print("{} is not a valid sampler type!".format(odesolver_type))
        sample, nfe = sampler()
        
        
        sample = sample.squeeze()

        
        x_hat = model.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        
        
        # x_hat = model.enhance(y, sampler_type=sampler_type, predictor=predictor, 
        #         corrector=corrector, corrector_steps=corrector_steps, N=N, snr=snr,
        #         atol=atol, rtol=rtol, timestep_type=timestep_type, correct_stepsize=correct_stepsize)


        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("odesolver_type: {}\n".format(odesolver_type))
        file.write("odesolver: {}\n".format(odesolver))
        
        file.write("N: {}\n".format(N))
        
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Reverse end point: {}\n".format(reverse_end_point))
        
        
        
        if odesolver_type == "black":
            file.write("atol: {}\n".format(atol))
            file.write("rtol: {}\n".format(rtol))
