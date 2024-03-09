import importlib

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, mode):
        dataset_module = importlib.import_module("data." + args.dataset_name)
        soccernet_dataset = getattr(dataset_module, 'SoccerNet')
        self.dataset = soccernet_dataset()
        

    def __len__(self):
        return self.dataset.__len__()


    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)