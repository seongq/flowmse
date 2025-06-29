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
from flowmse.model import VFModel
import pdb
import os
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver, get_black_box_solver



from utils import energy_ratios, ensure_dir, print_mean_std

import pdb

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--odesolver_type", type=str, choices=("white"), default="white",
                        help="Specify the sampler type, we used white")
    parser.add_argument("--odesolver", type=str, choices=('euler'),default="euler", help="Numerical integrator")
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the ODE.")
    parser.add_argument("--last_eval_point", type=float, default=0.03)
    
    
    parser.add_argument('--folder_destination', type=str, required=True, help="Destination path of inference results. Absolute path is required.")
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--N", type=int, default=5, help="Number of time steps")
    
    parser.add_argument("--N_mid", type=int, default=0, help="It is not related to FlowSE")
    

    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    checkpoint_file = args.ckpt
    
    # Settings
    sr = 16000
    odesolver_type = args.odesolver_type
    odesolver = args.odesolver
    N = args.N
    N_mid = args.N_mid
   




    # Load vector field model
    model = VFModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=8, num_workers=4, kwargs=dict(gpu=False)
    )
    
    reverse_starting_point = args.reverse_starting_point
    reverse_end_point = args.last_eval_point
        
    model.eval(no_ema=False)
    model.cuda()
    import re
    match = re.search(r'epoch=(\d+)', checkpoint_file)
    if match:
        epoch = match.group(1)  # 숫자만 추출
        print(f"Extracted epoch: {epoch}")
    else:
        print("Epoch not found in the path.")
        
    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    odename = "FLOWMATCHING"
    sigma_min = model.ode.sigma_min
    sigma_max =model.ode.sigma_max
        
       
    # print(tr_dataset)
    # print(tst_dataset)
    
    folder_destination = args.folder_destination
    target_dir = f"{folder_destination}/"
    
    ensure_dir(target_dir + "files/")
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
        
        if N_mid ==0:
            if odesolver_type == "white":
                sampler = get_white_box_solver(odesolver, model.ode, model, Y=Y.cuda(), Y_prior=Y.cuda(), T_rev=reverse_starting_point, t_eps=reverse_end_point,N=N)
        
            else:
                print("{} is not a valid sampler type!".format(odesolver_type))
            
       
        else:
            raise(f"N_mid should be 0.")
       
        
        sample, _ = sampler()
        sample = sample.squeeze()

        
        x_hat = model.to_audio(sample, T_orig)
        
        y = y * norm_factor
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        
        

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
        file.write(f"epoch: {epoch}"+"\n")
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("odesolver_type: {}\n".format(odesolver_type))
        file.write("odesolver: {}\n".format(odesolver))
        
        
        
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Last evaluated point: {}\n".format(reverse_end_point))
        
        file.write("data: {}\n".format(args.test_dir))
        file.write("ode: {}\n".format(odename))
    
        file.write(f"sigma_min: {sigma_min}"+"\n")
        file.write(f"sigma_max: {sigma_max}"+"\n")
    
    
        file.write("N: {}\n".format(N))
