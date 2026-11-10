import os
import torch
from torch import nn
from torch.utils.data import DataLoader

def fit_pretrain():
    pass

def fit_sft(model, dataset, config, output):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    os.makedirs(os.path.join(output, 'log'), exist_ok=True)
    model.train()
    with open(os.path.join(output, 'log', 'train.log'), 'w', buffering=1) as log:
        for epoch in range(config['epochs']):
            total, count = 0.0, 0
            for x, y in dataloader:
                loss = criterion(model(x.float()), y.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.detach().item() * len(y)
                count += len(y)
            message = f'epoch={epoch + 1} loss={total / count:.6f}'
            log.write(message + '\n')
            print(message, flush=True)
    torch.save(model.state_dict(), os.path.join(output, 'result', config['name'] + '.pth'))
