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
from flowmse.model import VFModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="otflow")    
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for VFModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     ode_class = ODERegistry.get_by_name(temp_args.ode)
     parser = pl.Trainer.add_argparse_args(parser)
     VFModel.add_argparse_args(
          parser.add_argument_group("VFModel", description=VFModel.__name__))
     ode_class.add_argparse_args(
          parser.add_argument_group("ODE", description=ode_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = VFModel(
          backbone=args.backbone, ode=args.ode, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['VFModel']),
               **vars(arg_groups['ODE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )
     # Set up logger configuration
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     else:
          if ode_class.__name__ == "OTFLOW":
               logger = WandbLogger(project="OTFLOW", log_model=True, save_dir="logs", name=f"otflow_sigma-min_{args.sigma_min}_t_eps_{args.t_eps}")
          elif ode_class.__name__ == "OTFLOW_DET":
               logger = WandbLogger(project="OTFLOW", log_model=True, save_dir="logs", name=f"otflow_det_t_eps_{args.t_eps}")
          elif ode_class.__name__ == "STRAIGHTCFM":
               logger = WandbLogger(project="STRAIGHTCFM", log_model=True, save_dir="logs", name=f"STRAIGHT_CFM_{args.sigma_min}_T_rev_{args.T_rev}_t_eps_{args.t_eps}")
          elif ode_class.__name__ == "STOCHASTICINTERPOLANT":
               logger = WandbLogger(project="STOCHASTICINTERPOLANT", log_model=True, save_dir="logs", name=f"STOCHASTICINTERPOLANT_T_rev_{args.T_rev}_t_eps_{args.t_eps}")
          elif ode_class.__name__ == "SCHRODINGERBRIDGE":
               logger = WandbLogger(project="SCHRODINGERBRIDGE", log_model=True, save_dir="logs", name=f"STOCHASTICINTERPOLANT_sigma_{args.sigma}_T_rev_{args.T_rev}_t_eps_{args.t_eps}")
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     #callbacks = [ModelCheckpoint(dirpath=f"logs/{logger.version}", save_last=True, filename='{epoch}-last')]
     if args.num_eval_files:
          checkpoint_callback_last = ModelCheckpoint(dirpath=f"logs/{logger.version}",
               save_last=True, filename='{epoch}-last')
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
               save_top_k=5, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
               save_top_k=5, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          #callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr] 
          callbacks = [checkpoint_callback_last, checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          accelerator='gpu', strategy=DDPPlugin(find_unused_parameters=False), gpus=[2,3], auto_select_gpus=False, 
          logger=logger, log_every_n_steps=10, num_sanity_val_steps=0, max_epochs=250,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model)

   