import argparse
import os
import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from source.data.qm9 import get_train_val_test
from source.modules.graph_task import KuramotoGNN
from source.modules.gnn_with_embedding import NodeGNN, EdgeGNN
from source.utils import save_checkpoint, save_model, LinearWarmupScheduler
from transformers import get_cosine_schedule_with_warmup

import accelerate
from accelerate import Accelerator
from tqdm import tqdm
from ema_pytorch import EMA

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=593)
    parser.add_argument("--epochs", type=int, default=500, help="num of epochs")

    parser.add_argument("--model_name", type=str, required=True, help="KuramotoGNN, NodeGNN, EdgeGNN")
    parser.add_argument("--gnn_layer_type", type=str, default="gcn")
    parser.add_argument("--n_gnn_layers", type=int, default=3)

    # parser.add_argument(
    #     "--checkpoint_every",
    #     type=int,
    #     default=50,
    #     help="save checkpoint every specified epochs",
    # )
    parser.add_argument("--warmup_iters", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--beta_ema", type=float, default=0.998)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--L", type=int, default=1, help="num of layers")
    parser.add_argument("--T", type=int, default=8, help="num of recurrence")

    parser.add_argument("--message_passing_type", type=str, default="attn")
    parser.add_argument("--readout_type", type=str, default="gat")
    parser.add_argument(
        "--num_head", type=int, default=8, help="num of heads in self-attention"
    )


    parser.add_argument("--osc_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=4)
    parser.add_argument("--task", type=int, default=0, help="task in QM9 dataset")
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")

    args = parser.parse_args()
    from pprint import pprint

    pprint(vars(args))

    return args

def train_one_epoch(model, ema, optimizer, loader, epoch, task, accelerator,  scheduler):
    model.train()
    epoch_loss = 0.0

    
    loss_function = nn.L1Loss()

    for data in tqdm(loader, desc=f"Epoch {epoch}: "):
        
        outputs, reference, track_osc = model(x = data.x, edge_index = data.edge_index, batch = data.batch)
        targets = data.y[:, task].reshape(-1, 1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        scheduler.step()        

        ema.update()

        epoch_loss += loss.item()
        

    total_loss = epoch_loss / len(loader)
    
    
    return total_loss

@torch.no_grad
def evaluate(model, loader, task, name, device):
    model.eval()
    loss_function = nn.L1Loss()
    total_loss = 0
    for data in tqdm(loader, desc=f"Evaluate {name}: "):
        data = data.to(device)
        outputs, reference, track_osc = model(x = data.x, edge_index = data.edge_index, batch = data.batch)
        targets = data.y[:, task].reshape(-1, 1)
        loss = loss_function(outputs, targets)

        total_loss += loss.item()

    return total_loss / len(loader)



def main():
    args = get_args()
    accelerator = Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.seed + accelerator.process_index)
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    jobdir = f"runs/{args.project_name}/"

    if accelerator.is_main_process:
        if not os.path.exists(jobdir):
            os.makedirs(jobdir)  # Make results folder (holds all experiment subfolders)
            logger = create_logger(jobdir)
            logger.info(f"Experiment directory created at {jobdir}")
        else:
            logger = create_logger(jobdir)
    import json
    if accelerator.is_main_process:
        logger.info(json.dumps(vars(args), indent=4, ensure_ascii=False))
    train_loader, val_loader, test_loader = get_train_val_test(args.data_path, args.batchsize, num_workers=args.num_workers)

    if accelerator.is_main_process:
        writer = SummaryWriter(jobdir)

    if args.model_name == "KuramotoGNN":


        model = KuramotoGNN(
            in_dim=11, 
            osc_dim=args.osc_dim, 
            final_dim=1, 
            L=args.L, 
            T=args.T, 
            hidden_dim=args.hidden_dim,
            message_passing_type = args.message_passing_type, 
            num_head=args.num_head, 
            gamma=args.gamma, 
            edge_dim=0, 
            readout_type=args.readout_type
        )
    elif args.model_name == "NodeGNN":
        model  = NodeGNN(
            in_dim=11, 
            osc_dim=args.osc_dim, 
            final_dim=1, 
            L=args.L, 
            T=args.T, 
            hidden_dim=args.hidden_dim,
            n_gnn_layers= args.n_gnn_layers,
            gnn_layer_type= args.gnn_layer_type,
            message_passing_type = args.message_passing_type, 
            num_head=args.num_head, 
            gamma=args.gamma, 
            edge_dim=0, 
            readout_type=args.readout_type
        )
    elif args.model_name == "EdgeGNN":
        model  = EdgeGNN(
                    in_dim=11, 
                    osc_dim=args.osc_dim, 
                    final_dim=1, 
                    L=args.L, 
                    T=args.T, 
                    hidden_dim=args.hidden_dim,
                    n_gnn_layers= args.n_gnn_layers,
                    gnn_layer_type= args.gnn_layer_type,
                    message_passing_type = args.message_passing_type, 
                    num_head=args.num_head, 
                    gamma=args.gamma, 
                    edge_dim=0, 
                    readout_type=args.readout_type
                )

    else:
        raise NotImplementedError

    if accelerator.is_main_process:
        logger.info(f"{model}")
        # print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of basemodel parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    ema = EMA(model, beta=args.beta_ema, update_every=10, update_after_step=200)
    ema.to(device)
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")

    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters, num_training_steps=args.epochs * len(train_loader))
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    scheduler = LinearWarmupScheduler(optimizer=optimizer, warmup_iters=args.warmup_iters)


    for epoch in range(0, args.epochs):
        total_loss = train_one_epoch(model, ema, optimizer, train_loader, epoch,  
                                     task=args.task, accelerator=accelerator, scheduler=scheduler)
        if accelerator.is_main_process:
                logger.info(f"[Epoch {epoch}] Training Loss: {total_loss:.4f}")
        
        if accelerator.is_main_process:
            writer.add_scalar("training loss", total_loss, epoch)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # save_checkpoint(
            #     accelerator.unwrap_model(model),
            #     optimizer,
            #     epoch,
            #     total_loss,
            #     checkpoint_dir=jobdir,
            # )
            # save_model(ema, epoch, checkpoint_dir=jobdir, prefix="ema")
            val_loss = evaluate(model=model, loader=val_loader, task=args.task, name="validation", device=device)
            writer.add_scalar("validation loss", val_loss, epoch)
            logger.info(f"[Epoch {epoch}] validation Loss: {val_loss:.4f}")
                        
    if accelerator.is_main_process:
        test_loss = evaluate(model=model, loader=test_loader, task=args.task, name="test", device=device)
        writer.add_scalar("test loss", test_loss, epoch)
        logger.info(f"test Loss: {val_loss:.4f}")
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(jobdir, f"model.pth"),
        )
        torch.save(ema.state_dict(), os.path.join(jobdir, f"ema_model.pth"))


if __name__ == "__main__":
    main()
