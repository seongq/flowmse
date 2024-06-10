import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel, VFModel_finetuning


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     parser = ArgumentParser()
     parser.add_argument("--batch_size", type=int, default=8,  help="During training take at least N_min reverse steps")
     parser.add_argument("--N_min", type=int, default= 1,  help="During training take at least N_min reverse steps")
     parser.add_argument("--N_max", type=int, default= 1,  help="During training take at most N_max reverse steps")
     parser.add_argument("--t_eps_min", type=float, default = 0.03,  help="During training take at least N_min reverse steps")
     parser.add_argument("--t_eps_max", type=float, default = 0.03,  help="During training take at most N_max reverse steps")
     parser.add_argument("--pre_ckpt", type=str,  help="Load ckpt")     
     parser.add_argument("--base_dir", type=str)
     parser.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")

     
     parser.add_argument("--inference_N", type=int, default=1,  required=True, help="inference N")
     


     args = parser.parse_args()
     checkpoint_file = args.pre_ckpt

    # Load score model
     model = VFModel_finetuning.load_from_checkpoint(
        checkpoint_file, base_dir=args.base_dir,
        batch_size=args.batch_size, num_workers=0, kwargs=dict(gpu=False)
    )
     model.add_para(args.N_min, args.N_max, args.t_eps_min, args.t_eps_max, 
                    args.batch_size, args.inference_N)
     
    
     
     # model.to('cuda:2')
     # print(model.ode.__class__.__name__)
     # Set up logger configuration
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     else:
          if model.ode.__class__.__name__ == "OTFLOW":
               logger = WandbLogger(project="OTFLOW_FINETUNING", log_model=True, save_dir="logs", name=f"otflow")
          elif model.ode.__class__.__name__ == "OTFLOW_DET":
               logger = WandbLogger(project="OTFLOW_FINETUNING", log_model=True, save_dir="logs", name=f"otflow")
          elif model.ode.__class__.__name__ == "STRAIGHTCFM":
               logger = WandbLogger(project="STRAIGHTCFM_FINETUNING", log_model=True, save_dir="logs", name=f"STRAIGHT_CFM")
          elif model.ode.__class__.__name__ == "STOCHASTICINTERPOLANT":
               logger = WandbLogger(project="STOCHASTICINTERPOLANT_FINETUNING", log_model=True, save_dir="logs", name=f"STOCHASTICINTERPOLANT")
          elif model.ode.__class__.__name__ == "SCHRODINGERBRIDGE":
               assert args.T_rev < 1
               logger = WandbLogger(project="SCHRODINGERBRIDGE_FINETUNING", log_model=True, save_dir="logs", name=f"SCHRODINGERBRIDGE_sigma")
          elif model.ode.__class__.__name__ == "FLOWMATCHING":
               logger = WandbLogger(project=f"{model.ode.__class__.__name__}_FINETUNING", log_model=True, save_dir="logs", name=f"{model.ode.__class__.__name__}_N_min_{args.N_min}_{args.N_max}")
          elif model.ode.__class__.__name__ == "FLOWMATCHING_LIN_VAR":
               logger = WandbLogger(project=f"{model.ode.__class__.__name__}_FINETUNING", log_model=True, save_dir="logs", name=f"{model.ode.__class__.__name__}")
          elif model.ode.__class__.__name__ == "FLOWMATCHING_QUAD_VAR":
               logger = WandbLogger(project=f"{model.ode.__class__.__name__}_FINETUNING", log_model=True, save_dir="logs", name=f"{model.ode.__class__.__name__}")
          elif model.ode.__class__.__name__ == "BBED":
               logger = WandbLogger(project=f"{model.ode.__class__.__name__}_FINETUNING", log_model=True, save_dir="logs", name=f"{model.ode.__class__.__name__}")        
          
          else:
               raise ValueError(f"{model.ode.__class__.__name__}에 대한 configuration이 만들어지지 않았음")
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"logs/{model.ode.__class__.__name__}_finetuning__N_min_{args.N_min}_{args.N_max}_{logger.version}", save_last=True, filename='{epoch}-last')]
     checkpoint_callback_last = ModelCheckpoint(dirpath=f"logs/{model.ode.__class__.__name__}_finetuning__N_min_{args.N_min}_{args.N_max}_{logger.version}",
          save_last=True, filename='{epoch}-last')
     checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"logs/{model.ode.__class__.__name__}_finetuning__N_min_{args.N_min}_{args.N_max}_{logger.version}", 
          save_top_k=5, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
     checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"logs/{model.ode.__class__.__name__}_finetuning__N_min_{args.N_min}_{args.N_max}_{logger.version}", 
          save_top_k=5, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
     #callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr] 
     callbacks = [checkpoint_callback_last, checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(  accelerator='gpu', strategy=DDPPlugin(find_unused_parameters=False), gpus=[0,1], auto_select_gpus=False, 
          logger=logger, log_every_n_steps=10, num_sanity_val_steps=0, max_epochs=10,
          callbacks=callbacks)

     # Train model
     trainer.fit(model)

   