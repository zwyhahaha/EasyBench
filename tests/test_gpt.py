import wandb
import torch
from torch.nn import functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gpt import GPT, GPTConfig
from tests.utils import set_seed, get_optimizer, get_scheduler

def test_gpt(config, seed=42, warmup=False):
    set_seed(seed)

    epochs = config.epochs
    task = config.task # ['function', 'network', 'llm', 'gpt']
    optimizer_name = config.optimizer

    assert task == 'gpt'
    wandb.run.name = f"{optimizer_name}_{config.learning_rate}_seed_{seed}"

    # vocab size is 2, so we only have two possible tokens: 0,1
    vocab_size = 2
    # context length is 3, so we take 3 bits to predict the next bit probability
    context_length = 3

    gpt_config = GPTConfig(
        block_size = context_length,
        vocab_size = vocab_size,
        n_layer = 4,
        n_head = 4,
        n_embd = 16,
        bias = False,
    )

    model = GPT(gpt_config)

    seq = list(map(int, "111101111011110"))

    # convert the sequence to a tensor holding all the individual examples in that sequence
    X, Y = [], []
    # iterate over the sequence and grab every consecutive 3 bits
    # the correct label for what's next is the next bit at each position
    for i in range(len(seq) - context_length):
        X.append(seq[i:i+context_length])
        Y.append(seq[i+context_length])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    
    optimizer = get_optimizer(optimizer_name, model.parameters(), config)

    # train the GPT for some number of iterations
    for i in range(epochs):

        def closure():
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
            return loss
        loss = closure()
        loss.backward()

        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if optimizer_name in ['OSMM','OSGM',"OSMM2"]:
            optimizer.step(closure)
        else:
            optimizer.step()
        optimizer.zero_grad()
        wandb.log({'train_loss': loss.item(), 'epoch': i, 'grad_norm': total_norm})