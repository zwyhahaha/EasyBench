import wandb
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.utils import get_dataset
from tests.utils import get_optimizer, get_scheduler, get_network_data, get_data_info, set_seed
from models.network import LogReg, get_op_model

def test_op(config, seed=42, warmup=False):
    epochs = config.epochs
    task = config.task # ['function', 'network', 'llm']
    model = config.model # {'function': ['rosenbrock', 'rastrigin', 'least_squares'], 'network': ['mlp','vgg','resnet'], 'llm': ['llm']}
    optimizer_name = config.optimizer
    scheduler_name = config.scheduler
    lr_decay = config.lr_decay

    assert task == 'op'
    assert model == 'logreg'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(seed)

    wandb.run.name = f"{optimizer_name}_{config.learning_rate}_seed_{seed}"

    train_set = get_dataset(dataset_name=config.dataset,
                                     train_flag=True,
                                     datadir='./data',)

    batch_size = config.batch_size
    if batch_size == "full":
        batch_size =  len(train_set)
        
    train_loader = torch.utils.data.DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              batch_size=batch_size)

    # val set
    val_set = get_dataset(dataset_name=config.dataset,
                                   train_flag=False,
                                   datadir='./data',)
    
    def logistic_loss(model, images, labels, backwards=False):
        logits = model(images)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        loss = criterion(logits.view(-1), labels.view(-1))

        if backwards and loss.requires_grad:
            loss.backward()

        return loss
    
    def logistic_accuracy(model, images, labels):
        logits = torch.sigmoid(model(images)).view(-1)
        pred_labels = (logits > 0.5).float().view(-1)
        acc = (pred_labels == labels).float().mean()

        return acc
    
    @torch.no_grad()
    def compute_metric_on_dataset(model, dataset, name="loss"):
        if name == "loss":
            metric_function = logistic_loss
        elif name == "accuracy":
            metric_function = logistic_accuracy
        
        model.eval()
        loader = torch.utils.data.DataLoader(dataset, drop_last=False, batch_size=128)
        score_sum = 0.
        for batch in loader:
            images, labels = batch["images"].cuda(), batch["labels"].cuda()
            score_sum += metric_function(model, images, labels).item() * images.shape[0] 
        score = float(score_sum / len(loader.dataset))
        return score
    
    loss_function = logistic_loss
    model = get_op_model(train_set=train_set).cuda()

    optimizer = get_optimizer(optimizer_name, model.parameters(), config)
    scheduler = get_scheduler(optimizer, scheduler_name, lr_decay)

    for epoch in range(epochs):
        model.train()
        train_loss = compute_metric_on_dataset(model, train_set)
        valid_acc = compute_metric_on_dataset(model, val_set, name="accuracy")

        if np.isnan(train_loss):
            print("Train loss is NaN.")
            wandb.finish()
            break

        wandb.log({'train_loss': train_loss,
                'valid_acc': valid_acc,})
        print(f"Epoch {epoch}: train_loss: {train_loss}, valid_acc: {valid_acc}")

        for batch in train_loader:
            images, labels = batch["images"].cuda(), batch["labels"].cuda()
            optimizer.zero_grad()
            def closure():
                return loss_function(model, images, labels, backwards=False)
            loss = closure()
            loss.backward()

            if optimizer_name in ['OSMM','OSGM',"OSMM2","OSMM3"]:
                optimizer.step(closure,epoch=epoch,monotone_step=epochs)
            else:
                optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        