import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
from utils import load_config, training_gan, plot_training_progress, set_all_seeds
from gan.generator import Generator
from gan.discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    CONFIG_PATH = "./config"
    config = load_config(config_path=CONFIG_PATH)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data
    batch_size = config['General']['BATCH_SIZE']
    transform = transforms.ToTensor()
    train_dataset = SVHN(root='~/datasets', split='train', transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Generator and Discriminator
    set_all_seeds(config['General']['SEED'])
    generator = Generator(
        z_dim=config['Hyperparameters']['Z_DIM'], 
        label_embed_size=config['Hyperparameters']['EMBED_SIZE'], 
        conv_dim=config['Hyperparameters']['CONV_DIM']
    ).to(device)
    discriminator = Discriminator(
        conv_dim=config['Hyperparameters']['CONV_DIM'],
        image_size=config['General']['IMAGE_SIZE']
    ).to(device)

    # Define criterion
    criterion = nn.BCELoss()
    
    # Define optimizers
    optimizer_gen = torch.optim.Adam(
        generator.parameters(), 
        lr=config['Hyperparameters']['LR_GEN'], 
        betas=config['Hyperparameters']['BETAS_ADAM_GEN']
    )
    optimizer_disc = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config['Hyperparameters']['LR_DISC'], 
        betas=config['Hyperparameters']['BETAS_ADAM_DISC']
    )

    # Alternatively provide a lr-scheduler
    scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_gen, gamma=0.99
    )
    scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_disc, gamma=0.99
    )

    # Define summary writer for using Tensorboard to keep track of progress
    summary_writer = SummaryWriter("logs/gan")
    
    # train the cgan
    img_list = training_gan(
        generator=generator,
        discriminator=discriminator,
        optimizer_gen=optimizer_gen,
        optimizer_disc=optimizer_disc,
        scheduler_gen=scheduler_gen,
        scheduler_disc=scheduler_disc,
        num_epochs=config['Hyperparameters']['NUM_EPOCHS'],
        train_loader=train_loader,
        criterion=criterion,
        summary_writer=summary_writer,
        checkpoint_path=config['Weights']['SAVE_WEIGHTS_PATH']
    )

    # Visualization
    plot_training_progress(img_lst=img_list)