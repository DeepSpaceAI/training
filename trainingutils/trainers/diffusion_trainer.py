from training.trainers.trainer import Trainer
from training.utils import Config
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import mse_loss
import tqdm

class DiffusionTrainer(Trainer):
    def __init__(
            self,
            model,
            optim,
            dataset,
            scheduler,
            lr_scheduler,
            device,
            training_config: Config
        ):
        super().__init__(
            model,
            optim,
            dataset,
            device,
            training_config
        )

        self.scheduler: DDPMScheduler = scheduler
        self.learning_rate_scheduler = lr_scheduler

    @classmethod
    def get_default_config(self) -> Config:
        # Default Diffusion parameters
        max_timesteps: int = 1000

        # Default checkpointing parameters
        checkpoint: bool = True
        checkpoint_path: str = "./checkpoints/"
        checkpoint_iter: int = 50

        # Default training parameters
        learning_rate: float = 1e-4
        learning_rate_warmup_steps: int = 500
        epochs: int = 1000
        batch_size: int = 25
        shuffle: bool = True

        # Acceleration Parameters
        mixed_precision = "fp16"
        gradient_accum_steps=1
        output_dir="ddpm-craters"
        
        return Config(
            max_timesteps=max_timesteps,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_iter=checkpoint_iter,
            learning_rate=learning_rate,
            learning_rate_warmup_steps=learning_rate_warmup_steps,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            mixed_precision=mixed_precision,
            gradient_accum_steps=gradient_accum_steps,
            output_dir=output_dir
        )

    def train(self) -> None:
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True
        )

        losses = []

        for epoch in tqdm.tqdm(range(self.epochs), desc="Epoch"):
            loss_sum = 0
            for pbc, image in tqdm.tqdm(dataloader, desc="Batch", leave=False):
                self.optimizer.zero_grad()

                dims = image.size()
                noise = torch.randn(dims)
                timesteps = torch.randint(1000, (dims[0],), dtype=torch.int64)
                noisey_batch = self.scheduler.add_noise(image, noise, timesteps)

                noisey_batch = noisey_batch.to(self.device)
                noise = noise.to(self.device)
                timesteps = timesteps.to(self.device)

                predicted_noise = self.model(noisey_batch, timesteps, return_dict=False)[0]
                loss = mse_loss(noise, predicted_noise)

                loss.backward()
                self.optimizer.step()
                self.learning_rate_scheduler.step()
                loss_sum += loss.item()
            
            losses.append(loss_sum / len(dataloader))
            if epoch % self.checkpoint_iter == 0 and self.checkpoint and bool(epoch):
                self._save_checkpoint(epoch, losses)
