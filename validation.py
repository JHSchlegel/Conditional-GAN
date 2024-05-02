import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import SVHN
from gan.generator import Generator
from utils import load_config, load_checkpoint, run_experiment, speed_test, plot_generator_sample

if __name__ == '__main__':
    # Path to the configuration file
    CONFIG_PATH = "./config"
    
    # Load configuration settings
    config = load_config(config_path=CONFIG_PATH)
    
    # Define device to be GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load testing data
    batch_size = config['Testing']['BATCH_SIZE']
    transform = transforms.ToTensor()
    test_dataset = SVHN(root='~/datasets', split='test', transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Create generator instance and reinstate the latest checkpoint
    generator = Generator(z_dim=config['Hyperparameters']['Z_DIM'], 
                          label_embed_size=config['Hyperparameters']['EMBED_SIZE'], 
                          conv_dim=config['Hyperparameters']['CONV_DIM'])
    
    generator = load_checkpoint(generator, config['Weights']['LOAD_WEIGHTS_PATH'], device).to(device)

    # Check whether generator passes speed test
    print(f"Checking whether generator passes speed test\n") # logging
    
    speed_test(
        generator=generator,
        n_runs=config['Testing']['SPEED_RUNS'], 
        max_time=config['Testing']['SPEED_MAX'], 
        n_samples=config['Testing']['SPEED_SAMPLES']
    )

    # Visualizing samples from the model
    plot_generator_sample(
        generator=generator,
        path='./img/svhn_gan_generated_example_images.png',
        figsize=(10, 10),
        n_samples=50,
        nrows_ncols=(5, 10)
    )

    # Run the experiments with different seeds
    seeds = config['Testing']['SEEDS']
    val_accuracies = [run_experiment(
        generator=generator, 
        test_loader=test_loader, 
        num_iters=config['Testing']['NUM_ITERS'], 
        lr=config['Testing']['LR_CLASSIFIER'], 
        seed=seed,
        num_epochs=config['Testing']['NUM_EPOCHS'],
        device=device) for seed in seeds]
    
    # Print the final results
    print(f"Final score: validation accuracy over {len(seeds)} seeds | mean: {np.mean(val_accuracies) * 100:.4f}% | std: {np.std(val_accuracies) * 100:.4f}%")
