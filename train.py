import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from torch.autograd import Variable
from PIL import Image
import model
import random
import cv2
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class my_dataset(Dataset):
    def __init__(self, list_data, transforms=None):
        self.list_data = list_data
        self.transforms = transforms
    def __getitem__(self, item):


        temp_dir = '/data/data_cloud/train/'+ self.list_data[item][0]
        img_temp_RGB = Image.open(temp_dir)
        if img_temp_RGB.mode != 'RGB':
            img_temp_RGB = img_temp_RGB.convert('RGB')
        if transforms != None:
            img_tensor = self.transforms(img_temp_RGB)
        
        
        temp_data = self.list_data[item][1]
        temp_data = temp_data.split(';')

        if len(temp_data) == 5:
            list_labels = [int(temp_data[0]), int(temp_data[1]), int(temp_data[2]), int(temp_data[3]), int(temp_data[4])]
            label_tensors = torch.from_numpy(np.array(list_labels, dtype=np.float32))
            
            list_scores = [1.0, 1.0, 1.0, 1.0, 1.0]
            label_scores = torch.from_numpy(np.array(list_scores, dtype=np.float32))
        elif len(temp_data) == 4:
            list_labels = [int(temp_data[0]), int(temp_data[1]), int(temp_data[2]), int(temp_data[3]), 0.0]
            label_tensors = torch.from_numpy(np.array(list_labels, dtype=np.float32))
            
            list_scores = [1.0, 1.0, 1.0, 1.0, 0.0]
            label_scores = torch.from_numpy(np.array(list_scores, dtype=np.float32))
        elif len(temp_data) == 3:
            list_labels = [int(temp_data[0]), int(temp_data[1]), int(temp_data[2]), 0.0, 0.0]
            label_tensors = torch.from_numpy(np.array(list_labels, dtype=np.float32))
            
            list_scores = [1.0, 1.0, 1.0, 0.0, 0.0]
            label_scores = torch.from_numpy(np.array(list_scores, dtype=np.float32))
        elif len(temp_data) == 2:
            list_labels = [int(temp_data[0]), int(temp_data[1]), 0.0, 0.0, 0.0]
            label_tensors = torch.from_numpy(np.array(list_labels, dtype=np.float32))
            
            list_scores = [1.0, 1.0, 0.0, 0.0, 0.0]
            label_scores = torch.from_numpy(np.array(list_scores, dtype=np.float32))
        elif len(temp_data) == 1:
            list_labels = [int(temp_data[0]), 0.0, 0.0, 0.0, 0.0]
            label_tensors = torch.from_numpy(np.array(list_labels, dtype=np.float32))
            
            list_scores = [1.0, 0.0, 0.0, 0.0, 0.0]
            label_scores = torch.from_numpy(np.array(list_scores, dtype=np.float32))

        return img_tensor, label_tensors.long(), label_scores
    def __len__(self):
        return len(self.list_data)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def loss_func(outputs_label, outputs_score, labels, scores, label_criterion, score_criterion):
    # outputs_label: batch_size*150
    # outputs_score: batch_size*5
    # labels:        batch_size*5
    # scores:        batch_size*5


    outputs_score[outputs_score < 0.25] = 0.0
    # outputs_score[outputs_score > 0.50] = 1.0
    loss_scores = score_criterion(outputs_score, scores)
    loss_labels = []
    for i in range(5):
        temp_label = labels[:, i]
        temp_label = temp_label.squeeze()
        output_label = outputs_label[:, 30*i:30*i+30]
        temp_loss = label_criterion(output_label, temp_label)
        loss_labels.append(temp_loss)
    
    loss_labels_sum = sum(loss_labels)
    loss = loss_scores + loss_labels_sum * 2.0
    return loss

def get_num_correct(outputs_label, outputs_score, labels):
    # outputs_label: batch_size*150
    # outputs_score: batch_size*5
    # labels:        batch_size*5
    num_correct = 0
    np_predict = np.zeros((int(outputs_label.size(0)),5), dtype=np.float32) # batch_size*1
    for i in range(5):
        temp_label = outputs_label[:, 30*i:30*i+30] # batch_size*30
        temp_score = outputs_score[:, i] # batch_size*1
        for j in range(temp_score.shape[0]):
            if temp_score[j].item() > 0.75:
                temp = temp_label[j:j+1, :]
                _, np_predict[j][i] = torch.max(temp, 1)
        
    tensor_predict = torch.from_numpy(np_predict)
    for i in range(tensor_predict.shape[0]):
        temp_1 = tensor_predict[i]
        temp_2 = labels[i].cpu().float()
        if torch.equal(temp_1, temp_2):
            num_correct += 1
    return num_correct


if __name__ == '__main__':

    batch_size = 64
    n_workers = 16
    learn_rate = 1e-4
    train_dir = './Train_label.csv'
    data_dir = '/data/data_cloud/'
    flag_mixup = False
    flag_pretrain = False
    input_size = 128
    crop_sieze = 512
    iter_num_multiple = 6.

    max_epoch = int(50. * iter_num_multiple)

    list_rate_train = []
    list_rate_val = []


    model = model.model_baseline_inception().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    if flag_pretrain:
        model.load_state_dict(torch.load('./model_final' + '.pb'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=1e-5)

    transforms_train = transforms.Compose([
        transforms.RandomRotation(30),
        # transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25),
        transforms.RandomCrop(crop_sieze, pad_if_needed=True),
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    transforms_valid = transforms.Compose([
        transforms.CenterCrop(crop_sieze),
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    file_reader = csv.reader(open(train_dir, encoding='utf-8'))
    list_data = []
    list_data_valid = []
    list_data_train = []

    label_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.MSELoss()
        
    list_data_all = []
    for i, temp_info in enumerate(file_reader):
        if i > 0:
            if (i+1) % 50 == 0:
                list_data_valid.append(temp_info)
            else:
                list_data_train.append(temp_info)
            list_data_all.append(temp_info)

    dataset_train = my_dataset(list_data_train, transforms=transforms_train)
    dataset_valid = my_dataset(list_data_valid, transforms=transforms_valid)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=n_workers)
    current_lr = learn_rate
    max_rate_val = 0
    for epoch in range(max_epoch):
        time_start = time.time()
        model = model.train()
        total = 0
        num_correct = 0
        for i ,(imgs, labels, scores) in enumerate(loader_train):
            imgs = imgs.to(device)
            labels = labels.to(device)
            scores = scores.to(device)
            optimizer.zero_grad()

            outputs_label, outputs_score = model(imgs)
            loss = loss_func(outputs_label, outputs_score, labels, scores, label_criterion, score_criterion)
            loss.backward()
            optimizer.step()

            # _, predicts = torch.max(outputs.data, 1)py
            total += labels.size(0)
            temp_correct = get_num_correct(outputs_label, outputs_score, labels)
            num_correct = num_correct + temp_correct
            if (i+1) % 10 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], loss:{:.6f}, lr:{:.6f}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss.item(), current_lr))
            
        acc_train = 100.0 * float(num_correct)/float(total)
        list_rate_train.append(acc_train)
        
        model = model.eval()
        total = 0
        num_correct = 0
        for (imgs, labels, scores) in loader_valid:
            imgs = imgs.to(device)
            labels = labels.to(device)
            scores = scores.to(device)
            optimizer.zero_grad()
            outputs_label, outputs_score = model(imgs)

            total += labels.size(0)
            temp_correct = get_num_correct(outputs_label, outputs_score, labels)
            num_correct = num_correct + temp_correct
        acc = 100.0 * float(num_correct)/float(total)
        if acc >= max_rate_val:
            max_rate_val = acc
            torch.save(model.state_dict(), './model_best' + '.pb' )
            flag_epoch = epoch + 1
        list_rate_val.append(acc)
        time_cost = time.time() - time_start
        print('Epoch:{}, Val_rate:{:.2f}%, train_rate:{:.2f}%, time_cost:{:.2f}s'.format(epoch+1, acc, acc_train, time_cost))

        # if epoch+1 == 150 or epoch+1 == 250:
        #     current_lr = current_lr * 0.1
        #     update_lr(optimizer, current_lr)
        # if epoch+1 == 200:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-5)
        if epoch+1 == int(20 * iter_num_multiple) or epoch+1 == int(35 * iter_num_multiple) or epoch+1 == int(45 * iter_num_multiple):
                current_lr = current_lr * 0.5
                update_lr(optimizer, current_lr)

    model = model.eval()
    torch.save(model.state_dict(), './model_final' + '.pb' )
    print('The most best rate_val:{:.2f}% , the best iter epoch is:{}'.format(max_rate_val, flag_epoch))
    log_train = open('./log.txt', 'w')
    for i in range(len(list_rate_train)):
        temp_line = str(i+1) + '  ' + str(round(list_rate_train[i], 2)) + '%' + '  ' + str(round(list_rate_val[i], 2)) + '%' + '\n'
        log_train.write(temp_line)
    log_train.close()

