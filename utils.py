import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import yaml
import time
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import accuracy_score
from classifier import Classifier
from gan.generator import Generator


# ---------------------------------------------------------------------------- #
#                  GAN Initialization and Reproducibility                      #
# ---------------------------------------------------------------------------- #
def weights_init(m: nn.Module):
    """Custom weight initialization function for the Generator and Discriminator.

    Args:
        m (nn.Module): Module to initialize.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


# see https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
# for more information aboutreproducibility in pytorch
def set_all_seeds(seed: int):
    """Set all possible seeds to ensure reproducibility and to avoid randomness
    involved in GPU computations.

    Args:
        seed (int): Seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------- #
#                         Model Saving and Loading                             #
# ---------------------------------------------------------------------------- #
# checkpoint_path = 'weights/' #'/content/gdrive/MyDrive/hodl/'
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,
    print_save: bool = False,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    directory = os.path.dirname(checkpoint_path)
    if checkpoint_path:
        os.makedirs(directory, exist_ok=True)

    checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
    checkpoint_filepath = checkpoint_path + checkpoint_filename
    torch.save(checkpoint, checkpoint_filepath)
    if print_save:
        print(f"Checkpoint saved at: {checkpoint_filepath}")


def load_checkpoint(
    model: nn.Module, 
    checkpoint_path: str='./weights/',
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> nn.Module:
    """
    Loads model's state dictionary from the provided checkpoint file.

    Args:
        model (nn.Module): The model for which the state dictionary will be loaded.
        checkpoint_path (str): Path to the saved checkpoint file. Defaults to './weights/'
        device (torch.device, optional): Device where the model should be loaded. Defaults to GPU if available, else CPU.

    Returns:
        nn.Module: The model with loaded state dictionary.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the model's state dictionary from the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Print confirmation message
    print(f"Checkpoint loaded from: {checkpoint_path}\n")
    
    return model


# ---------------------------------------------------------------------------- #
#                           Training and Evaluation                            #
# ---------------------------------------------------------------------------- #
#!!! This train loop was inspired by:
#!!! https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def training_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_gen: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    scheduler_gen: torch.optim.lr_scheduler,
    scheduler_disc: torch.optim.lr_scheduler,
    num_epochs: int,
    train_loader: DataLoader,
    criterion: nn.Module = nn.BCELoss(),
    z_dim: int = 10,
    summary_writer: SummaryWriter = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_path: str = "./weights/",
) -> list[torch.Tensor]:
    """Train a conditional GAN model.

    Args:
        generator (nn.Module): A generator model.
        discriminator (nn.Module): A discriminator model.
        optimizer_gen (torch.optim): Optimizer of the generator.
        optimizer_disc (torch.optim): Optimizer of the discriminator.
        scheduler_gen (torch.optim.lr_scheduler): Scheduler of the generator.
        scheduler_disc (torch.optim.lr_scheduler): Scheduler of the discriminator.
        num_epochs (int): Number of epochs to train.
        train_loader (DataLoader): DataLoader of the training data.
        criterion (nn.Module, optional): Loss function. Defaults to nn.BCELoss().
        z_dim (int, optional): Dimension of the latent space. Defaults to 10.
        summary_writer (SummaryWriter, optional): A SummaryWriter object to log
            training progress to tensorboard. Defaults to None.
        device (torch.device, optional): Device to run the code on. Defaults to device.
        checkpoint_path (str, optional): . Defaults to "weights/gan_phile/".

    Returns:
        list[torch.Tensor]: List of images to visualize training progress.
    """
    iters = 0
    img_list = []
    fixed_noise = torch.randn(40, z_dim, 1, 1, device=device)
    fixed_label = torch.arange(0, 10).repeat(4).to(device)
    with trange(num_epochs) as t:
        for epoch in t:
            D_x = 0.0
            D_G_z1 = 0.0
            D_G_z2 = 0.0

            disc_loss_epoch = 0.0
            gen_loss_epoch = 0.0

            for i, (imgs, label) in enumerate(train_loader):

                label = label.to(device)

                dct = {}

                real_images = imgs.to(device)
                real_labels = torch.ones(
                    (real_images.shape[0],), dtype=torch.float, device=device
                )

                # simple way to truncate tails of standard normal distribution
                # avoids too much variety in generated images during training
                noise = torch.randn(real_images.size(0), z_dim, 1, 1).to(device)
                fake_images = generator(noise, label)
                fake_labels = torch.zeros(
                    (real_images.shape[0],), dtype=torch.float, device=device
                )

                # ------------------------------------------------------#
                # Train Discriminator: max log(D(x)) + log(1 - D(G(z))) #
                # ------------------------------------------------------#
                optimizer_disc.zero_grad()
                real_output = discriminator(real_images, label)
                disc_real_loss = criterion(real_output, real_labels)

                fake_output = discriminator(fake_images.detach(), label)
                disc_fake_loss = criterion(fake_output, fake_labels)

                disc_loss = (disc_real_loss + disc_fake_loss) / 2.0

                disc_loss.backward()

                optimizer_disc.step()

                D_x += real_output.mean().item()
                D_G_z1 += fake_output.mean().item()

                disc_loss_epoch += disc_loss.item()

                # --------------------------------------#
                # Train Generator: max log(D(G(z)))     #
                # --------------------------------------#
                optimizer_gen.zero_grad()
                output = discriminator(fake_images, label)
                gen_loss = criterion(output, real_labels)
                gen_loss.backward()
                optimizer_gen.step()

                D_G_z2 += output.mean().item()

                gen_loss_epoch += gen_loss.detach().item()

                if (iters % 5000 == 0) or (
                    (epoch == num_epochs - 1) and (i == len(train_loader) - 1)
                ):
                    with torch.no_grad():
                        fake = generator(fixed_noise, fixed_label).detach().cpu()
                    img_list.append(
                        torchvision.utils.make_grid(fake, padding=2, normalize=True)
                    )

                iters += 1

            dct["D_x"] = D_x / len(train_loader)
            dct["D_G_z1"] = D_G_z1 / len(train_loader)
            dct["D_G_z2"] = D_G_z2 / len(train_loader)
            dct["disc_loss"] = disc_loss_epoch / len(train_loader)
            dct["gen_loss"] = gen_loss_epoch / len(train_loader)

            if summary_writer is not None:
                summary_writer.add_scalar("Generator Loss", dct["gen_loss"], epoch)
                summary_writer.add_scalar("Discriminator Loss", dct["disc_loss"], epoch)
                summary_writer.add_scalar("D_x", dct["D_x"], epoch)
                summary_writer.add_scalar("D_G_z1", dct["D_G_z1"], epoch)
                summary_writer.add_scalar("D_G_z2", dct["D_G_z2"], epoch)

            if scheduler_gen:
                scheduler_gen.step()
            if scheduler_disc:
                scheduler_disc.step()
            save_checkpoint(
                generator, optimizer_gen, epoch, dct["gen_loss"], checkpoint_path
            )

            # update progressbar
            t.set_postfix(dct)
    return img_list


def training_classifier_one_epoch(
    classifier: Classifier,
    generator: Generator,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss=nn.CrossEntropyLoss(),
    num_iters: int=100,
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple[float, float]:
    """
    Train the classifier for one epoch using generated data from the generator. For each iteration of training, 
    the generator generates 1000 images. 

    Args:
        classifier (Classifier): The classifier model.
        generator (Generator): The generator model.
        optimizer (Optimizer): The optimizer for updating the classifier's parameters.
        criterion (_Loss, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        num_iters (int, optional): Number of iterations for training. Defaults to 100.
        device (torch.device, optional): Device to perform training on. Defaults to GPU if available, else CPU.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    # Set the classifier to training mode
    classifier.train()

    # Initialize dictionary to store training losses and accuracies
    results = {'train_losses': [], 'train_accuracies': []}

    # Iterate over the specified number of iterations
    for _ in trange(num_iters, desc='Training'):
        # Generate data using the generator
        images, labels = generator.generate()
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = classifier(images)
        loss = criterion(logits, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Append loss and accuracy to the results dictionary
        results['train_losses'].append(loss.item())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        results['train_accuracies'].append(accuracy_score(preds, labels))

    # Calculate average loss and accuracy for the epoch
    train_loss_epoch = sum(results['train_losses']) / len(results['train_losses'])
    train_acc_epoch = sum(results['train_accuracies']) / len(results['train_accuracies'])

    return train_loss_epoch, train_acc_epoch


def evaluate(
    classifier: Classifier,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss=nn.CrossEntropyLoss(),
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple[list, list, float, float]:
    """
    Evaluate the classifier on the test set.

    Args:
        classifier (Classifier): The classifier model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (_Loss): Loss function. Defaults to nn.CrossEntropyLoss(),
        device (torch.device, optional): Device to perform evaluation on. Defaults to GPU if available, else CPU.

    Returns:
        tuple: A tuple containing true values, predicted values, average loss, and accuracy.
    """
    # Set the classifier to evaluation mode
    classifier.eval()

    # Initialize dictionary to store true values, predicted values, and losses
    results = {'true_values': [], 'pred_values': [], 'losses': []}

    # Iterate over batches in the test loader
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Perform forward pass without gradient computation
        with torch.no_grad():
            logits = classifier(images)
            loss = criterion(logits, labels)

        # Append loss to the losses list
        results['losses'].append(loss.item())

        # Convert logits and labels to numpy arrays and append to respective lists
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        results['true_values'].extend(labels.tolist())
        results['pred_values'].extend(preds.tolist())

    # Calculate accuracy and average loss
    accuracy = accuracy_score(results['true_values'], results['pred_values'])
    avg_loss = sum(results['losses']) / len(results['losses'])

    # Return true values, predicted values, average loss, and accuracy
    return results['true_values'], results['pred_values'], avg_loss, accuracy


def run_experiment(
    test_loader: DataLoader,
    generator: Generator,
    num_iters: int=100,
    lr: float=0.001,
    seed: int=42,
    num_epochs: int=10,
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> float:
    """
    Runs an experiment with the given generator and test loader, training a classifier and evaluating its performance.

    Args:
        test_loader (DataLoader): DataLoader containing the test dataset.
        generator (Generator): The generator model used for generating samples.
        num_iters (int, optional): Number of training iterations per epoch for Classifier. Default is 100.
        lr (float, optional): Learning rate for optimizer of Classifier. Default is 0.001.
        seed (int, optional): Random seed for reproducibility. Default is 42.
        num_epochs (int, optional): Number of training epochs of Classifier. Default is 10.
        device (torch.device, optional): Device to run the experiment on (CPU or GPU). Default is automatically determined based on GPU availability.

    Returns:
        float: The accuracy of the classifier on the validation set after training.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize classifier, criterion, and optimizer
    classifier = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Container to store training and validation metrics
    dct = {'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs':[]}

    # Loop through epochs
    for epoch in range(1, num_epochs + 1):
        
        # Training
        train_loss_epoch, train_acc_epoch = training_classifier_one_epoch(
            classifier=classifier,
            generator=generator,
            optimizer=optimizer,
            criterion=criterion,
            num_iters=num_iters,
            device=device
        )

        # Validation
        _, _, val_loss_epoch, val_acc_epoch = evaluate(
            classifier=classifier,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        # Print and store metrics for this epoch
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss_epoch:.3f}, Accuracy: {train_acc_epoch * 100:.2f}% | Validation Loss: {val_loss_epoch:.3f}, Accuracy: {val_acc_epoch * 100:.2f}%")
        dct['train_losses'].append(train_loss_epoch)
        dct['train_accs'].append(train_acc_epoch)
        dct['val_losses'].append(val_loss_epoch)
        dct['val_accs'].append(val_acc_epoch)

    # plot training progress
    plot_training_curves(dct['train_losses'], dct['train_accs'], dct['val_losses'], dct['val_accs'])

    # Return the accuracy on the validation set after the last epoch
    return dct['val_accs'][-1]


def speed_test(
    generator: Generator,
    n_runs: int=10,
    max_time: int=3,
    n_samples: int=1000
) -> None:
    """
    Tests the speed of the generator by generating samples multiple times and measuring the average time per run.

    Args:
        generator (Generator): An instance of the Generator model.
        n_runs (int, optional): Number of times the generator is used to generate samples. Defaults to 10.
        max_time (int, optional): Maximum average time allowed for generating samples (in seconds). Defaults to 3.
        n_samples (int, optional): Number of samples the generator should generate. Defaults to 1000.

    Raises:
        AssertionError: If the average time exceeds the maximum allowed time.
    """
    start_time = time.time()

    # Generate n_samples a total of n_runs times
    for _ in range(n_runs):
        generator.generate(n_samples=n_samples)

    end_time = time.time()

    # Calculate average time it took to generate n_samples
    average_time = (end_time - start_time) / n_runs

    print(f"Average time per execution: {average_time:.6f} seconds")

    assert average_time <= max_time, "Your generator is too slow"


# ---------------------------------------------------------------------------- #
#                                Visualization                                 #
# ---------------------------------------------------------------------------- #
def plot_training_progress(
    img_lst: list[torch.Tensor],
    path: str='animations/gan_training_progress.mp4',
    figsize: tuple[int, int]=(10, 10),
) -> None:
    """Plot the training progress of the GAN model.

    Args:
        path (str, optional): Path to save the animation. Defaults to "animations/gan_training_progress.mp4".
        img_lst (list[torch.Tensor]): List of images to visualize training progress.
        figsize (tuple[int,int], optional): Size of the figure. Defaults to (10,10).
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_lst]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )

    # make sure to have ffmpeg installed
    # e.g. using sudo apt install ffmpeg
    ani.save(path, writer="ffmpeg", fps=1)


def plot_generator_sample(
    generator: Generator,
    path: str='./img/generator_sample.png',
    figsize: tuple[int, int]=(10, 10),
    n_samples: int=50,
    nrows_ncols: tuple[int, int]=(5, 10)
) -> None:
    """
    Plot a grid of generated images along with their labels.

    Args:
        generator (Generator): An instance of the Generator model.
        path (str, optional): Path to save the plot. Defaults to "./img/generator_sample.png".
        figsize (tuple[int,int], optional): Size of the figure. Defaults to (10,10).
        n_samples (int): Number of samples the generator should generate.
        nrows_ncols (tuple[int,int], optional): Number of rows and columns for the image grid.

    Returns:
        None
    """    
    # Create the directory to save the plot
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    # Generate n_samples pictures
    images, labels = generator.generate(n_samples=n_samples)

    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.3)

    # Plot each image with its label
    for ax, im, label in zip(grid, images, labels):
        im = im.permute(1, 2, 0)  # Reshape image tensor for display
        im = im.detach().cpu().numpy()  # Convert tensor to numpy array

        ax.imshow(im)
        ax.axis('off')  # Turn off axis
        ax.set_title(f'{label.item()}')  # Set title with image label

    # Adjust layout and display plot
    plt.tight_layout() 
    plt.savefig(path) 
    plt.show()


def plot_training_curves(
    train_losses: list, 
    train_accs: list,
    val_losses: list,
    val_accs: list,
    figsize: tuple[int, int]=(10, 10)
) -> None:
    """
    Plot training and validation curves for loss and accuracy when training the classifier.

    Args:
        train_losses (list): List of training losses for each epoch.
        train_accs (list): List of training accuracies for each epoch.
        val_losses (list): List of validation losses for each epoch.
        val_accs (list): List of validation accuracies for each epoch.
        figsize (tuple[int, int], optional): Figure size (width, height) in inches. Defaults to (10, 10).

    Returns:
        None
    """
    # Create figure and axes
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # Plot training and validation losses
    axs[0].plot(np.arange(len(train_losses)) + 1, train_losses, label="train")
    axs[0].plot(np.arange(len(val_losses)) + 1, val_losses, label="valid")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="best")

    # Plot training and validation accuracies
    axs[1].plot(np.arange(len(train_accs)) + 1, train_accs, label="train")
    axs[1].plot(np.arange(len(val_accs)) + 1, val_accs, label="valid")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="best")

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------- #
#                                 Load config                                  #
# ---------------------------------------------------------------------------- #
def load_config(
    config_path: str='./config/',
    config_file: str='config.yaml'
) -> dict:
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str, optional): The path to the directory containing the configuration file. Defaults to './config/'.
        config_file (str, optional): The name of the configuration file. Defaults to 'config.yaml'.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    # Open and read the YAML file
    with open(os.path.join(config_path, config_file)) as f:
        config = yaml.safe_load(f)

    return config