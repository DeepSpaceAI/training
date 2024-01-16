from training import TrainingCheckpoint
from training.utils import Config
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
            self,
            model,
            optim,
            dataset,
            device,
            training_config: Config
        ):
        self.model: nn.Module = model
        self.optimizer = optim
        self.dataset = dataset
        self.device = device
        self.config = training_config
        self.__update_intern_dict(training_config)

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

    def __update_intern_dict(self, config: Config):
        self.__dict__.update(config.__dict__)

    def train(self):
        raise NotImplementedError("Training Function not Implemented")
    
    def _save_checkpoint(self, epoch, losses):
        # Implement Save Checkpointing
        checkpoint_folder = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}")
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
            
        checkpoint = TrainingCheckpoint(
            self.model,
            self.optimizer,
            self.learning_rate_scheduler,
            self.config
        )

        checkpoint.save(checkpoint_folder)

        plt.plot(losses)
        plt.savefig(os.path.join(checkpoint_folder, "losses.png"))
    
    def _load_checkpoint(self):
        # Implement Load Checkpointing
        pass

    def save_results(self, path: str):
        # Implement Final Training Results Checkpoint
        if not os.path.exists(path):
            os.mkdir(path)

        model_stats = {
            "model_weights": self.model.state_dict()
        }

        torch.save(model_stats, os.join(path, "path.pt"))