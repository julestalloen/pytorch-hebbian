from ignite.utils import convert_tensor
from torch.utils.data import random_split


def split_dataset(dataset, val_split):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))
