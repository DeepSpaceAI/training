import torch
import torch.nn as nn
import os

class TrainingCheckpointer:
    def __init__(
            self,
            model,
            optimizer,
            lr_scheduler,
            config,
            **kwargs
        ):
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.__dict__.update(kwargs)

    def load(
            self,
            trainer,
            fpath,
            device
        ):
        torch_data = os.path.join(fpath, "checkpoint.pt")
        save_data = torch.load(torch_data, device)
        self.model.load_state_dict(save_data["model_state"], map_location=device)
        self.optimizer.load_state_dict(save_data["optim_state"], map_location=device)
        self.lr_scheduler.load_state_dict(save_data["lr_schedueler_state"], map_location=device)
        self.config.__dict__.update(save_data["config"])
        trainer.epoch_iter = save_data["epoch_iter"]
    
    def save(
            self,
            epoch_iter,
            fpath
        ):
        model_info = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "config": self.config.__dict__,
            "epoch_iter": epoch_iter
        }

        fpath = os.path.join(fpath, "checkpoint.pt")
        torch.save(model_info, fpath)