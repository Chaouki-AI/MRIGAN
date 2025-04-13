import torch
import argparse
from datetime import datetime
from helper import load_model, load_losses,  load_dl, load_optimizers_and_schedulers, gan_trainer, srgan_trainer, load_summary_writer, evaluator, save_ckpt




if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # train parameters 
    parser.add_argument("--epochs", default=1, type=int, help='number of epochs')
    parser.add_argument("--bs",     default=2 , type=int, help='batch size for training')

    # image parameters 
    parser.add_argument("--RGB", action="store_false", 
                        help="Choose whether using one channel (Gray image) or three dimensional image (RGB)") 
    parser.add_argument("--height", type=int, default=512, help='image input height')
    parser.add_argument("--width" , type=int, default=512, help='image input width')

    #Loss choice 
    parser.add_argument("--loss", default="SILog", type=str, help="loss function to be used for training",
                        choices=['L1', 'SIlog', 'SRLOSS'])

    parser.add_argument("--data_path", default="./DATA/DATA_1/", type=str, help='path of images')
    
    
    # Optimizer parameters
    parser.add_argument("--optimizer", default="AdamW", type=str, help="name of the optimizer to be used for training",
                        choices=['AdamW', 'RMSprop'])
    parser.add_argument("--lr", "--learning-rate", default=0.000357, type=float, 
                        help='max learning rate')
    parser.add_argument("--wd", "--weight-decay", default=0.1, type=float, 
                        help='weight decay')
    
    args = parser.parse_args()
   
        
    name = f"{args.loss}_channels-{3 if args.RGB else 1}_{datetime.now().strftime('%m_%d_%H_%M')}"
    #Load Summary Writer 
    writer = load_summary_writer(args)


    #Load The models
    discriminator, generator = load_model(args)

    #Create The dataloader 
    dataloader_tr, dataloader_ts = load_dl(args)
    #print(len(dataloader_tr))
    #Create the Optimizer 
    Gen_opts, Dis_opts = load_optimizers_and_schedulers(args, models = [generator, discriminator], steps_per_epoch = len(dataloader_tr))

    #Get the Devices ; 
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    #load losses
    adversarial_criterion, content_criterion = load_losses(args)
    trainer = srgan_trainer if adversarial_criterion is None else gan_trainer
 
    #loop
    for epoch in range(0, args.epochs):
        
        #train
        generator, discriminator, Gen_opts, Dis_opts, writer = trainer(epoch, args, dataloader_tr, generator, discriminator, adversarial_criterion, content_criterion, Gen_opts, Dis_opts, writer, device)
        
        #evaluate
        metrics, writer =  evaluator(args, generator, dataloader_ts, epoch, device, writer)
        
        #save the ckpts
        save_ckpt(args, name, generator, metrics, epoch)
     