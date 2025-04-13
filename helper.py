import os 
import torch
import glob
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime
from tools.metrics import Loss, GeneratorLoss
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tools.dataloader import ImagePathDataset
from tools.models import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter
from tools.metrics import compute_errors, RunningAverage, RunningAverageDict


def load_model(args):
    Dis = Discriminator()
    Gen = Generator(3 if args.RGB else 1)
    return Dis, Gen

def load_optimizers_and_schedulers(args, models, steps_per_epoch=100):
    optimizer_name = args.optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)
    generator, discriminator = models
    
    optimizer_gen = optimizer_class(generator.get_parameters(), weight_decay=args.wd, lr=args.lr)
    scheduler_gen = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_gen, args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            cycle_momentum=True, three_phase=False, base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
            div_factor=args.epochs * 100, anneal_strategy='linear', final_div_factor=args.epochs)
    
    optimizer_dis = optimizer_class(discriminator.get_parameters(), weight_decay=args.wd, lr=args.lr)
    scheduler_dis = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_dis, args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            cycle_momentum=True, three_phase=False, base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
            div_factor=args.epochs * 100, anneal_strategy='linear', final_div_factor=args.epochs)

    return (optimizer_gen, scheduler_gen), (optimizer_dis, scheduler_dis)

def load_summary_writer(args):
    """
    Load the tensorboard summary writer for the specified experiment name.

    Args:
        args (Namespace): The parsed command line arguments containing experiment configurations.
        name (str): The name of the experiment.

    Returns:
        SummaryWriter: The tensorboard summary writer.
    """
    channels = 3 if args.RGB else 1
    name = f"{args.loss}_channels-{channels}_{datetime.now().strftime('%m_%d_%H_%M')}"

    # Create the directory for storing tensorboard logs if it doesn't exist
    os.makedirs(f"./checkpoints/{name}",  exist_ok=True)

    # Initialize the tensorboard summary writer
    writer = SummaryWriter(f"./runs/{name}")

    return writer

def gan_trainer(epoch, args, dataloader, G, D, adversarial_criterion, content_criterion, gen_opts, dis_opts, writer, device):
    G = G.to(device)
    D = D.to(device)

    optimizer_G, scheduler_G = gen_opts
    optimizer_D, scheduler_D = dis_opts

    global_step = 0
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
    for i, batch in pbar:
        hr = batch["HR"].to(device)
        lr = batch["LR"].to(device)

        # Train Discriminator
        D.zero_grad()
        output_real = D(hr)
        labels_real = torch.ones_like(output_real, device=device)
        loss_D_real = adversarial_criterion(output_real, labels_real)

        sr = G(lr)
        output_fake = D(sr.detach())
        labels_fake = torch.zeros_like(output_fake, device=device)
        loss_D_fake = adversarial_criterion(output_fake, labels_fake)

        loss_D = 5. * (loss_D_real + loss_D_fake)
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        G.zero_grad()
        output_fake_for_G = D(sr)
        labels_real_for_G = torch.ones_like(output_fake_for_G, device=device)
        loss_G_adv = adversarial_criterion(output_fake_for_G, labels_real_for_G)

        loss_G_content = content_criterion(sr, hr)
        loss_G = loss_G_content + 1e-3 * loss_G_adv
        loss_G.backward()
        optimizer_G.step()

        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()

        writer.add_scalar("Loss/Generator", loss_G.item(), global_step)
        writer.add_scalar("Loss/Discriminator", loss_D.item(), global_step)
        writer.add_scalar("LR/Generator", scheduler_G.get_last_lr()[0], global_step)
        writer.add_scalar("LR/Discriminator", scheduler_D.get_last_lr()[0], global_step)
        global_step += 1

        pbar.set_postfix({"G_loss": f"{loss_G.item():.4f}", "D_loss": f"{loss_D.item():.4f}"})
        scheduler_G.step()
        scheduler_D.step()

    print(f" Epoch Avg. Gen. loss {(epoch_G_loss/len(dataloader)):.4f} \n Epoch Avg. Dis. loss {(epoch_D_loss/len(dataloader)):.4f}")

    return G, D, (optimizer_G, scheduler_G), (optimizer_D, scheduler_D), writer

def srgan_trainer(epoch, args, dataloader, G, D, adversarial_criterion, generator_criterion, gen_opts, dis_opts, writer, device):
    G = G.to(device)
    D = D.to(device)

    optimizer_G, scheduler_G = gen_opts
    optimizer_D, scheduler_D = dis_opts

    global_step = 0
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0

    G.train()
    D.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
    for i, batch in pbar:
        hr = batch["HR"].to(device)
        lr = batch["LR"].to(device)

        # Generate fake high-res images
        sr = G(lr)

        # Train Generator
        optimizer_G.zero_grad()
        output_fake_for_G = D(sr).mean()
        loss_G = generator_criterion(output_fake_for_G, sr, hr)
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        output_real = D(hr).mean()
        output_fake = D(sr.detach()).mean()
        loss_D = 1 - output_real + output_fake
        loss_D.backward()
        optimizer_D.step()

        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()

        writer.add_scalar("Loss/Generator", loss_G.item(), global_step)
        writer.add_scalar("Loss/Discriminator", loss_D.item(), global_step)
        writer.add_scalar("LR/Generator", scheduler_G.get_last_lr()[0], global_step)
        writer.add_scalar("LR/Discriminator", scheduler_D.get_last_lr()[0], global_step)
        global_step += 1

        pbar.set_postfix({"G_loss": f"{loss_G.item():.4f}", "D_loss": f"{loss_D.item():.4f}"})
        scheduler_G.step()
        scheduler_D.step()

    #print(f" Epoch Avg. Gen. loss {(epoch_G_loss/len(dataloader)):.4f} \n Epoch Avg. Dis. loss {(epoch_D_loss/len(dataloader)):.4f}")

    return G, D, (optimizer_G, scheduler_G), (optimizer_D, scheduler_D), writer

def load_losses(args):
    if args.loss == 'L1' or args.loss == 'SIlog':
        adversarial_criterion = nn.BCEWithLogitsLoss()
        content_criterion = Loss(args)
        return adversarial_criterion, content_criterion
    else : 
        return None, GeneratorLoss()

def load_dl(args): 

    images = glob.glob(f'{args.data_path}/**/**/*.jpg')[:]
    total = len(images)
    split_size = int(total * 0.8)
    train_list = random.sample(images, split_size)
    test_list  = [item for item in images if item not in train_list]

    train = ImagePathDataset(train_list, shape = (args.height, args.width), scales = [1, 8], compress = [60, 90], blur = [0.2, 2], RGB = args.RGB)
    dataloader_tr = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=0)

    test = ImagePathDataset(test_list, shape = (args.height, args.width), scales = [1, 8], compress = [60, 90], blur = [0.2, 2], RGB = args.RGB)
    dataloader_ts = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    return dataloader_tr, dataloader_ts

def log_sr_images(writer, lr_tensor, sr_tensor, hr_tensor, step, tag='SR_Comparison'):
    """
    Logs LR, SR (output), and HR images to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        lr_tensor (torch.Tensor): Low-res input tensor (B, C, H, W), B=1.
        sr_tensor (torch.Tensor): Super-res output tensor (B, C, H, W), B=1.
        hr_tensor (torch.Tensor): High-res ground truth tensor (B, C, H, W), B=1.
        step (int): Global step or epoch to associate with the images.
        tag (str): Main tag name for the image group.
    """
    # Remove batch dimension
    lr = lr_tensor[0].detach().cpu()
    sr = sr_tensor[0].detach().cpu()
    hr = hr_tensor[0].detach().cpu()

    # Optional: Resize lr to match hr for visual alignment (optional)
    if lr.shape[1:] != hr.shape[1:]:
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), size=hr.shape[1:], mode='bilinear', align_corners=False)[0]

    # Stack them horizontally: [LR | SR | HR]
    grid = make_grid([lr, sr, hr], nrow=3)

    writer.add_image(tag, grid, global_step=step)

def log_sr_images_vertical(writer, lr_tensor, sr_tensor, hr_tensor, step, sample_id=0, tag_prefix='SR_Comparison'):
    """
    Logs LR, SR (output), and HR images to TensorBoard vertically.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        lr_tensor (torch.Tensor): Low-res input tensor (B, C, H, W), B=1.
        sr_tensor (torch.Tensor): Super-res output tensor (B, C, H, W), B=1.
        hr_tensor (torch.Tensor): High-res ground truth tensor (B, C, H, W), B=1.
        step (int): Epoch or training step for TensorBoard.
        sample_id (int): Unique identifier for multiple samples per step.
        tag_prefix (str): Tag prefix for grouping in TensorBoard.
    """
    # Remove batch dimension
    lr = lr_tensor[0].detach().cpu()
    sr = sr_tensor[0].detach().cpu()
    hr = hr_tensor[0].detach().cpu()

    # Resize LR to match HR for visual alignment (optional but helpful)
    if lr.shape[1:] != hr.shape[1:]:
        lr = F.interpolate(lr.unsqueeze(0), size=hr.shape[1:], mode='bilinear', align_corners=False)[0]

    # Stack vertically: [LR]
    #                   [SR]
    #                   [HR]
    vertical = torch.cat([lr, sr, hr], dim=1)  # Concatenate along height

    # Add image to TensorBoard with unique tag
    tag = f"{tag_prefix}/Sample_{sample_id}"
    writer.add_image(tag, vertical, global_step=step)

def evaluator(args, model, test_loader, epoch, device, writer):
    """
    Evaluate the model on a dataset and log comparisons.
    
    Args:
        args (Namespace): Command line arguments containing evaluation configuration.
        model (nn.Module): Model to evaluate.
        test_loader (DataLoader): Data loader for test dataset.
        epoch (int): Current epoch number.
        device (torch.device): Device for evaluation.
        writer (SummaryWriter): TensorBoard writer.
    
    Returns:
        tuple: (metrics_value, val_si, writer)
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        metrics = RunningAverageDict()
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{args.epochs} Validation")):
            # Get ground truth (hr) and input image (lr) as torch tensors.
            hr = batch['HR'].to(device)  # Ground truth
            lr = batch['LR'].to(device)  # Input image for the model
            
            # Forward pass to predict depth
            pred_ = model(lr)
            # Resize prediction to match ground truth (if needed)
            pred_ = torch.nn.functional.interpolate(pred_, hr.shape[-2:], mode='bilinear', align_corners=True)
            
            # Remove extra dimensions and convert to numpy array
            pred = pred_.squeeze().cpu().numpy()  # may be (H, W) or (C, H, W)
            
            # Correct abnormal values in prediction
            pred[pred < 1e-3] = 1e-3
            pred[pred > 1.] = 1.
            pred[np.isinf(pred)] = 1.
            pred[np.isnan(pred)] = 1e-3
            
            # Prepare ground truth depth
            gt_depth = hr.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth < 1.)
            
            # Log images every 150 batches; using lr as input and hr as depth.
            if batch_idx % 150 == 0:
                log_sr_images_vertical(writer, lr, pred_, hr, epoch, batch_idx, tag_prefix=f'Epoch_{epoch + 1}')   
            #    writer = log_images_to_tensorboard(lr, hr, pred, batch_idx, epoch, writer, spacing=10)
            
            # Update metrics for valid regions
            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))
        
        # Format metric values and log to TensorBoard
        values = {key: float(f"{value:.5f}") for key, value in metrics.get_value().items()}
        writer.add_scalar('Metrics/rmse', values['rmse'], epoch + 1)
        #writer.add_scalar('Metrics/sq_rel', values['sq_rel'], epoch + 1)
        writer.add_scalar('Metrics/acc1', values['a1'], epoch + 1)
        writer.add_scalar('Metrics/acc2', values['a2'], epoch + 1)
        writer.add_scalar('Metrics/acc3', values['a3'], epoch + 1)
        writer.add_scalar('Metrics/abs_rel', values['abs_rel'], epoch + 1)
        
        return values, writer

def save_ckpt(args, name, generator, metrics, epoch):
    """
    Save the model checkpoint.

    Args:
        args (Namespace): The parsed command line arguments.
        model (nn.Module): The model to be saved.
        metrics (dict): A dictionary containing the evaluation metrics.
        name (str): The name of the experiment.
        epoch (int): The current epoch number.
    """
    # Construct the checkpoint filename using relevant metrics and epoch number
    checkpoint_filename = os.path.join(
        './checkpoints',
        f"{name}",
        f"epoch-{epoch+1}_abs_rel-{metrics['abs_rel']}_A1-{metrics['a1']}_best.pt"
    )
    
    # Save the model state dictionary to the specified file
    torch.save(generator.state_dict(), checkpoint_filename)
    