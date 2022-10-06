from torchvision import transforms, datasets
import torch
import pdb
from opts import parse_opts

def get_transforms(mode):
    assert mode == 'train' or mode == 'val'
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif mode == 'val':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


def get_dataloader(args):
    # Define the train & test datasets
    #train_dataset = datasets.CIFAR10(args.train_data_folder,
    #                                train=True,
    #                                download=True,
    #                                transform=get_transforms('train'))


    train_dataset = datasets.ImageFolder(args.train_data_folder,transform=get_transforms('train'))
    print(train_dataset.classes)


    #val_dataset = datasets.CIFAR10(args.test_data_folder,
    #                               train=False,
    #                               transform=get_transforms('val'))

    val_dataset = datasets.ImageFolder(args.test_data_folder, transform=get_transforms('val'))
    
    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=False,
                                               num_workers=args.num_workers, 
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               drop_last=True,
                                               num_workers=args.num_workers, 
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=16, 
                                              shuffle=True)
    return train_loader, valid_loader, test_loader
