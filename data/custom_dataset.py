import 

class CustomDataset(Dataset):
    def __init__(self, args, mode):
        self.dataset = None

        dataset_module = importlib.import_module("data." + args.dataset_name)
        bahave_dataset = getattr(dataset_module, 'BehaveDataset')
        # if args.dataset_name == "behave_v1":
        #     from data.behave_dataset_v1 import BehaveDataset as custom_dataset
        # else:
        #     from data.behave_dataset_v2 import BehaveDataset as custom_dataset

        self.dataset = bahave_dataset(args, mode)
        
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)