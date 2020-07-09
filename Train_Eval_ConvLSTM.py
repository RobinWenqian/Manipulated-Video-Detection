import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import tqdm
import numpy as np
from torchvision import datasets, models, transforms
from torchsummary import summary
from ConvLSTM_Models import ConvLSTM
from ConvLSTM_Dataloader import RandomCrop, GeneralVideoDataset

def main():
    args = parse.parse_args()
    experiment_name = args.experiment_name
    train_path = args.train_path
    val_path = args.val_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    gpu_avi = args.gpu_avi

    output_path = os.path.join('../output', experiment_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.backends.cudnn.benchmark = gpu_avi

	#creat train and val dataloader
    Train_Dataset_Config = {
        'clips_list_file' : train_path,
        'root_dir' : '../Celeb-DF-v2',
        'channels' : 3,
        'time_depth' : 64,
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225),
        'cut_size' : 256,
        'transformflag' : True
    }

    Valid_Dataset_Config = {
        'clips_list_file' : val_path,
        'root_dir' : '../Celeb-DF-v2',
        'channels' : 3,
        'time_depth' : 64,
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225),
        'cut_size' : 256,
        'transformflag' : True
    }

    ConvLSTM_Config = {
        'channels' : 3,
        'hidden_dim' : [128,128,256],
        'kernel_size' : (3, 3),
        'num_layers' : 3,
        'batch_first' : True,
        'bias' : True,
        'return_all_layers' : False
    }

    transform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(Train_Dataset_Config['mean'], Train_Dataset_Config['std']),
        ])
    
    train_set = GeneralVideoDataset(
        Train_Dataset_Config['clips_list_file'],
        Train_Dataset_Config['root_dir'],
        Train_Dataset_Config['channels'],
        Train_Dataset_Config['time_depth'],
        Train_Dataset_Config['mean'],
        Train_Dataset_Config['std'],
        Train_Dataset_Config['cut_size'],
        Train_Dataset_Config['transformflag'],
        transform
        )
    
    val_set = GeneralVideoDataset(
        Valid_Dataset_Config['clips_list_file'],
        Valid_Dataset_Config['root_dir'],
        Valid_Dataset_Config['channels'],
        Valid_Dataset_Config['time_depth'],
        Valid_Dataset_Config['mean'],
        Valid_Dataset_Config['std'],
        Valid_Dataset_Config['cut_size'],
        Valid_Dataset_Config['transformflag'],
        transform
        )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    train_dataset_size = len(train_set)
    val_dataset_size = len(val_set)

	#Creat the model
    model = ConvLSTM(
        ConvLSTM_Config['channels'],
        ConvLSTM_Config['hidden_dim'],
        ConvLSTM_Config['kernel_size'],
        ConvLSTM_Config['num_layers'],
        ConvLSTM_Config['batch_first'],
        ConvLSTM_Config['bias'],
        ConvLSTM_Config['return_all_layers']
    )
    
    summary(model, input_size = 
        (batch_size,Train_Dataset_Config['time_depth'],
        Train_Dataset_Config['cut_size'],
        Train_Dataset_Config['cut_size']))
    
    if continue_train:
        model.load_state_dict(torch.load(model_path))
    
    if gpu_avi:
        model = model.cuda()

    criterion_CE = nn.CrossEntropyLoss()
    criterion_Hu = nn.SmoothL1Loss()

	#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)
        model=model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        for (image,labels,failflag) in train_loader:
            labels = np.array(labels)
            labels = torch.from_numpy(labels)
            #print(labels)
            iter_loss = 0.0
            iter_corrects = 0.0
            if gpu_avi:
                image = image.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion_CE(outputs, labels)
            #loss = criterion_Hu(preds,labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not(iteration % 10):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        #Evaluation
        model.eval()
        with torch.no_grad():
            for (image, labels, failflag) in val_loader:
                print(image.shape)
                if gpu_avi:
                    image = image.cuda()
                    labels = labels.cuda()
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion_CE(outputs, labels)
                #loss = criterion_Hu(preds,labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        if not (epoch % 10):
		#Save the model trained with multiple gpu
		#torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    #torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--experiment_name', '-n', type=str, default='Mesonet')
    parse.add_argument('--train_root','-tr',default = '../Celeb-DF-v2')
    parse.add_argument('--train_path', '-tp' , type=str, default = '../Celeb-DF-v2/List_of_training_videos.pkl')
    parse.add_argument('--val_path', '-vp' , type=str, default = '../Celeb-DF-v2/List_of_testing_videos.pkl')
    parse.add_argument('--batch_size', '-bz', type=int, default=1)
    parse.add_argument('--epoches', '-e', type=int, default='50')
    parse.add_argument('--model_name', '-mn', type=str, default='meso4.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='../output/Mesonet/best.pkl')
    parse.add_argument('--cut_size', '-cs', type=str, default=256)
    parse.add_argument('--gpu_avi', type = bool, default = False)

    main()