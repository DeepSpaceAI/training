from trainingutils.checkpointing import TrainingCheckpointer
from trainingutils.utils import Config
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
        self.__update_internal_state(training_config)

        self.checkpointer: TrainingCheckpointer = TrainingCheckpointer(
            self.model,
            self.optimizer,
            self.dataset,
            self.config
        )

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        
        self.epoch_iter = 0

    def __update_internal_state(
            self,
            config: Config
        ):
        
        self.__dict__.update(config.__dict__)

    def train(self):
        raise NotImplementedError("Training Function not Implemented")
    
    def _save(self, epoch, losses):
        # Implement Save Checkpointing
        checkpoint_folder = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}")
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)

        self.checkpointer.save(checkpoint_folder)

        plt.plot(losses)
        plt.savefig(os.path.join(checkpoint_folder, "losses.png"))
    
    def load(self, checkpoint_path: str, device="auto") -> None:
        """
        Load wrapper for internal checkpointer object, if a child
        class needs to save additional items, their states should be
        updated by overloading this function. This is just a simple
        base load function.
        """
        self.checkpointer.load(self, checkpoint_path, device)