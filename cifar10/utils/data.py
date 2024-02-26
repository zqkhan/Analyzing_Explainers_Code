import torch

def get_data_loader(x, y, batch_size=32, shuffle=True):
    """Fetches a DataLoader, which is built into PyTorch, and provides a
    convenient (and efficient) method for sampling.
    :param x: (torch.Tensor) inputs
    :param y: (torch.Tensor) labels
    :param batch_size: (int)
    """

    dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size)

    return data_loader

def get_data_loader_batched(x, y, d, batch_size=32, shuffle=True):
    """Fetches a DataLoader, which is built into PyTorch, and provides a
    convenient (and efficient) method for sampling.
    :param x: (torch.Tensor) inputs
    :param y: (torch.Tensor) labels
    :param batch_size: (int)
    """

    dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y), torch.Tensor(d))
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size)

    return data_loader