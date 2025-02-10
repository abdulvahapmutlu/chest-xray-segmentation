import datetime
from torch.utils.data import DataLoader

def format_time(seconds):
    """Convert seconds to hh:mm:ss string."""
    return str(datetime.timedelta(seconds=int(seconds)))

def create_dataloaders(train_ds, val_ds, test_ds, batch_size=4, num_workers=0):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
