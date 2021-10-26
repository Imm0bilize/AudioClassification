import wandb
import torch
from tqdm import tqdm


def train_step(model,
               optimizer,
               data_loader,
               ds_len,
               loss_fn,
               device):

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        total_loss += loss.item()
        total_acc += (torch.argmax(predictions, 1) == torch.argmax(labels, 1)).float().sum()
        loss.backward()
        optimizer.step()
        print(total_acc / (i + 1))
    return total_loss / ds_len, total_acc / ds_len


@torch.no_grad()
def validation_step(model,
                    data_loader,
                    ds_len,
                    loss_fn,
                    device):

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    for (images, labels) in data_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        total_loss += loss_fn(predictions, labels).items()
        total_acc += (torch.argmax(predictions, 1) == torch.argmax(labels, 1)).float().sum()
    return total_loss / ds_len, total_acc / ds_len


def train_loop(model,
               optimizer,
               scheduler,
               loss_fn,
               train_data_loader,
               train_ds_length,
               val_data_loader,
               val_ds_length,
               num_epochs,
               device):

    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(
            model, optimizer, train_data_loader, train_ds_length, loss_fn, device
        )
        val_loss, val_acc = validation_step(
            model, val_data_loader, val_ds_length, loss_fn, device
        )

        print(f'Epoch: {epoch + 1}\tLR:{scheduler.get_last_lr()[0]}\n'
              f'{train_loss=:0.3f} {train_acc=:0.3f} ; {val_loss=:0.3f} {val_acc=:0.3f}')
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc,
                   'learning_rate': scheduler.get_last_lr()[0]})

        scheduler.step()
