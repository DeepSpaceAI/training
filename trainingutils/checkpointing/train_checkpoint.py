import torch
import os

class TrainingCheckpoint:
    def __init__(
            self,
            model,
            optimizer,
            lr_scheduler,
            config
        ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

    @classmethod
    def load(file_path, device):
        save_data = torch.load(file_path, device)
        
        model = save_data["model"]
        learning_rate = save_data["learning_rate"]
        optimizer = save_data["optim"]
        iter_num = save_data["iter_num"]
        epoch_count = save_data["epoch_count"]
        batch_size = save_data["batch_size"]

        return TrainingCheckpoint(
            model,
            learning_rate,
            optimizer,
            iter_num,
            epoch_count,
            batch_size
        )
    
    def save(self, fpath):
        model_info = {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config.__dict__
        }

        fpath = os.path.join(fpath, "checkpoint.pt")
        torch.save(model_info, fpath)