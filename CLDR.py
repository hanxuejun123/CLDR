import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils import data
# from easydict import EasyDict
from dataloaders.base import MNIST, CIFAR10, CIFAR100
from dataloaders.datasetGen import SplitGen, PermutedGen
import copy
import argparse
import time


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='s-mnist / p-mnist / cifar10 / cifar100')
parser.add_argument('--runs', type=int, default=5, help='How many experiments to repeat')
parser.add_argument('--replay', type=int, default=1000, help='total memory size')
parser.add_argument('--epochs', type=int, default=80, help='training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='training learning-rate')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer: Adam / SGD')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes per task')
parser.add_argument('--ntasks', type=int, default=5, help='total number of tasks to learn')
parser.add_argument('--n_permutation', type=int, default=0, help='n-PermutedMnist. Default 0-SplitedMnist/SplitedCifar')
parser.add_argument('--filename', type=str, default='result_CLDR.txt', help='results saved filename')
parser.add_argument('--O1N1', type=bool, default=True, help='mini-batch:: old:new=1:1')
parser.add_argument('--criterion_mixup', type=str, default='CE', help='criterion_mixup')
parser.add_argument('--mixup_within_class', type=bool, default=False, help='mixup within the same class')
parser.add_argument('--alpha0', type=float, default=3, help='alpha0 - mixup beta distribution')
parser.add_argument('--alpha', type=float, default=1, help='alpha - weight for KD')
parser.add_argument('--beta', type=float, default=1, help='beta - weight for MM')
parser.add_argument('--gamma', type=float, default=0.001,  help='gamma - weight for FS')

args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(args.device)

args.temp = 2
args.criterion = nn.CrossEntropyLoss()

if args.dataset == 's-mnist':
    args.epochs = 4
    args.batch_size = 128
    args.lr = 1e-3
    args.num_classes = 2
    args.ntasks = 5
    args.feature_dim = 400
    args.net = 'MLP'
elif args.dataset == 'p-mnist':
    args.epochs = 10
    args.batch_size = 128
    args.lr = 1e-4
    args.num_classes = 10
    args.ntasks = 10
    args.feature_dim = 1000
    args.n_permutation = 10
    args.net = 'MLP'
elif args.dataset == 'cifar10':
    args.num_classes = 2
    args.ntasks = 5
    args.net = 'CNN'
elif args.dataset == 'cifar100':
    args.num_classes = 10
    args.ntasks = 10
    args.net = 'CNN'
else:
    print('- dataset error -')


class MLP(nn.Module):
    def __init__(self, in_features=32*32, ndf=1000, out_features=10):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, ndf)
        self.fc2 = nn.Linear(ndf, ndf)
        self.last = nn.Linear(ndf, out_features)  

    def features(self, x):
        x = self.fc1(x.view(-1, self.in_features))
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


class CifarNet(nn.Module):
    def __init__(self, in_channels=3, out_dim=10):
        super(type(self), self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
        self.linear_block = nn.Sequential(
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5) 
        )
        self.last = nn.Linear(512, out_dim)


    def weight_init(self):
        nn.init.constant_(self.last.weight, 0)
        nn.init.constant_(self.last.bias, 0)

    def features(self,x):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        return o

    def forward(self, x):
        o = self.features(x)
        o = self.last(o)
        return o



def mixup_data(x, y, x_r, y_r, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.alpha0 > 0:
        lam = np.random.beta(args.alpha0, args.alpha0)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x_r
    y_a, y_b = y, y_r
    return mixed_x, y_a, y_b, lam


def mixup_data_within_class(x, y, x_r, y_r, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.alpha0 > 0:
        lam = np.random.beta(args.alpha0, args.alpha0)
    else:
        lam = 1
    
    for c in range(args.num_classes):
        xc = x[torch.where(y==c)]
        xc_r = x_r[torch.where(y_r==c)]
        yc = y[torch.where(y==c)]               
        lenc = min(len(xc), len(xc_r))
        mixed_xc = lam * xc[:lenc] + (1-lam) * xc_r[:lenc]
        yc = yc[:lenc]

        if c==0:
            mixed_x = mixed_xc
            label = yc
        else:
            mixed_x = torch.cat((mixed_x, mixed_xc))
            label = torch.cat((label, yc))

    return mixed_x, label, label, lam


def mixup_criterion(pred, y_a, y_b, lam, loss_fn='MSE'):
    if loss_fn == 'MSE':
        y_a, y_b = F.one_hot(y_a, 2).float(), F.one_hot(y_b, 2).float()
        y = lam * y_a + (1 - lam) * y_b
        pred = F.softmax(pred)
        # loss = lam * nn.MSELoss()(pred, y_a) + (1 - lam) * nn.MSELoss()(pred, y_b)
        loss = nn.MSELoss()(pred, y)
    else: # CE
        loss = lam * nn.CrossEntropyLoss()(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss()(pred, y_b)
    return loss


def PCC(x, y):
    vx = x - torch.mean(x,(1,2),keepdim=True)
    vy = y - torch.mean(y)
    rho = torch.sum(vx*vy,(1,2),keepdim=True) / (torch.sqrt(torch.sum(vx**2,(1,2),keepdim=True)+1e-15) * torch.sqrt(torch.sum(vy ** 2)))    #size: dim*1*1
    return rho

def regularizer_l1(x, y, x_r, y_r, net, args):
    n, d = x.size()
    n_r, _ = x_r.size()
    x, x_r = x.permute(1,0).reshape(d,n,1), x_r.permute(1,0).reshape(d,n_r,1)
    y, y_r = F.one_hot(y,args.num_classes).float(), F.one_hot(y_r,args.num_classes).float()

    w = list(net.last.parameters())[0] 
    w = w.permute(1,0).reshape(d,1,args.num_classes)

    pred, pred_r = x * w, x_r * w
    pred, pred_r = F.softmax(pred), F.softmax(pred)         
    Delta = 1 - PCC(pred,y) * PCC(pred_r,y_r)
    L1_reg = torch.sum(Delta * torch.norm(w,1,dim=(1,2),keepdim=True))

    return L1_reg

def train_task1(train_loader, args):
    if args.net == 'CNN':
        net = CifarNet(out_dim=args.num_classes).to(args.device)
    elif args.net == 'MLP':
        net = MLP(ndf = args.feature_dim, out_features=args.num_classes).to(args.device)

    if args.optim == 'Adam':
        optimizer = optim.Adam(list(net.parameters()), lr=args.lr)
    else: 
        optimizer = optim.SGD(list(net.parameters()), lr=args.lr) #, momentum = 0.9, weight_decay=0.0001

    # training
    net.train()

    loss_lst = []

    for epoch in range(args.epochs):
        for _, (sample, label, _) in enumerate(train_loader):
            sample, label = sample.to(args.device), label.to(args.device)

            output_data = net(sample)
            loss_data = args.criterion(output_data, label)
            loss = loss_data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lst.append(loss.item())
        print(f'epoch: {epoch}/{args.epochs}, loss: {np.mean(loss_lst)}')

    traindata = iter(train_loader)
    batch = next(traindata)
    # all_sample = torch.empty(0,3,32,32)
    # all_label = torch.empty(0).long()
    all_sample, all_label, _ = batch
    for _ in range(int(args.replay/args.batch_size)):
        batch = next(traindata)
        all_sample = torch.cat((all_sample,batch[0]),0)
        all_label = torch.cat((all_label,batch[1]),0)
    id_replay = torch.randperm(len(all_sample))[:args.replay].to(dtype=torch.long, device=args.device)

    return net, ({'1': all_sample[id_replay]}, {'1': all_label[id_replay]})


def train_task2(train_loader,replay_set_old, net_old, args, num_tasks):
    net = copy.deepcopy(net_old)

    if args.optim == 'Adam':
        optimizer = optim.Adam(list(net.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(list(net.parameters()), lr=args.lr, momentum = 0.9, weight_decay=0.0001)

    # training
    net_old.eval()
    net.train()

    loss_lst = []
    loss_data_lst = []
    loss_data_replay_lst = []
    loss_mix_lst = []
    loss_lwf_data_lst = []
    loss_lwf_data_replay_lst = []

    all_replay_sample, all_replay_label = replay_set_old

    for epoch in range(args.epochs):
        for _, (sample, label, _) in enumerate(train_loader):
            sample, label = sample.to(args.device), label.to(args.device)
            sample_r_lst = []
            label_r_lst = []

            if args.O1N1:
                r_per_task = [len(sample) // (num_tasks-1) + (1 if x < len(sample) % (num_tasks-1) else 0) for x in range (num_tasks-1)]
            # [num // div + (1 if x < num % div else 0)  for x in range (div)]
            else:    
                r_per_task = [len(sample)]*(num_tasks-1)

            for task in range(1,num_tasks):
                sample_r, label_r = all_replay_sample[f'{task}'].to(args.device), all_replay_label[f'{task}'].to(args.device)
                if len(sample_r)<r_per_task[task-1]:
                    c = int(r_per_task[task-1]/len(sample_r)+1)
                    sample_r = torch.cat([sample_r] * c)
                    label_r = torch.cat([label_r] * c)
                idx = torch.randperm(len(sample_r))[:r_per_task[task-1]].to(dtype=torch.long, device=args.device)
                sample_r_lst.append(sample_r[idx])
                label_r_lst.append(label_r[idx])
            replay_sample, replay_label = torch.cat(sample_r_lst).to(args.device), torch.cat(label_r_lst).to(args.device)
            
            feature = net.features(sample)
            feature_replay = net.features(replay_sample)
            idx = torch.randperm(len(replay_sample))[:len(sample)].to(dtype=torch.long, device=args.device)

            if args.mixup_within_class:
                feature_mix, label_a, label_b, lam = mixup_data_within_class(feature, label, feature_replay[idx],replay_label[idx], args)
            else:
                feature_mix, label_a, label_b, lam = mixup_data(feature, label, feature_replay[idx],replay_label[idx], args)

            output = net.last(feature)
            output_replay = net.last(feature_replay)
            output_mix = net.last(feature_mix)

            output_old = net_old(sample)
            output_replay_old = net_old(replay_sample)

            loss_data = args.criterion(output, label)
            loss_data_replay = args.criterion(output_replay, replay_label)
            loss_mix = mixup_criterion(output_mix, label_a, label_b, lam, args.criterion_mixup)

            loss_lwf_data = - torch.sum(torch.mean(F.softmax(output_old/args.temp) * F.log_softmax(output/args.temp), 0),0)
            loss_lwf_data_replay = - torch.sum(torch.mean(F.softmax(output_replay_old/args.temp) * F.log_softmax(output_replay/args.temp), 0),0)
            
            reg = regularizer_l1(feature,label,feature_replay,replay_label,net,args)
            loss = (loss_data + loss_data_replay) + args.alpha*(loss_lwf_data + loss_lwf_data_replay) + args.beta*loss_mix + args.gamma * reg  

            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
            loss_lst.append(loss.item())
            loss_data_lst.append(loss_data.item())
            loss_data_replay_lst.append(loss_data_replay.item())
            loss_mix_lst.append(loss_mix.item())
            loss_lwf_data_lst.append(loss_lwf_data.item())
            loss_lwf_data_replay_lst.append(loss_lwf_data.item())
        print(f'epoch: {epoch}/{args.epochs}, loss: {np.mean(loss_lst)}, loss_data: {np.mean(loss_data_lst)}, loss_data_replay: {np.mean(loss_data_replay_lst)}, loss_mix: {np.mean(loss_mix_lst)}, loss_lwf_data: {np.mean(loss_lwf_data_lst)}, loss_lwf_data_replay: {np.mean(loss_lwf_data_replay_lst)} ')


    # num_per_task = args.replay//num_tasks
    num_per_task = [args.replay // num_tasks + (1 if x < args.replay % num_tasks else 0)  for x in range (num_tasks)]

    traindata = iter(train_loader)
    batch = next(traindata)
    all_sample, all_label, _ = batch
    for _ in range(int(num_per_task[num_tasks-1]/args.batch_size)):
        batch = next(traindata)
        all_sample = torch.cat((all_sample,batch[0]),0)
        all_label = torch.cat((all_label,batch[1]),0)

    idx = torch.randperm(len(all_sample))[:num_per_task[num_tasks-1]].to(dtype=torch.long, device=args.device)
    all_replay_sample[f'{num_tasks}'] = all_sample[idx]
    all_replay_label[f'{num_tasks}'] = all_label[idx]

    for task in range(1, num_tasks):
        idx_keep = torch.randperm(len(all_replay_sample[f'{task}']))[:num_per_task[task-1]].to(dtype=torch.long, device=args.device)
        all_replay_sample[f'{task}'] = all_replay_sample[f'{task}'][idx_keep]
        all_replay_label[f'{task}'] = all_replay_label[f'{task}'][idx_keep]
    
    print(all_replay_sample.keys())
    print(all_replay_label.keys())

    return net, (all_replay_sample, all_replay_label)


def test(val_loader, net, args):
    net.eval()
    correct = 0
    with torch.no_grad():
        for _, (sample, label, _) in enumerate(val_loader):
            sample, label = sample.to(args.device), label.to(args.device)
            pred = net(sample).argmax(dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()   
    acc = correct / len(val_loader.dataset)
    print(acc)
    return acc


def main(args):
    if args.dataset == 's-mnist':
        train_dataset, val_dataset = MNIST('data')
    elif args.dataset == 'p-mnist':
        train_dataset, val_dataset = MNIST('data')
    elif args.dataset == 'cifar10':
        train_dataset, val_dataset = CIFAR10('data')
    elif args.dataset == 'cifar100':
        train_dataset, val_dataset = CIFAR100('data')
    else:
        print('- dataset error -')

    if args.n_permutation > 0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                n_permute=args.n_permutation,
                                                                                remap_class=False)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                        first_split_sz=args.num_classes,
                                                                        other_split_sz=args.num_classes,
                                                                        rand_split=False,
                                                                        remap_class=True)
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)

    val_loaders = []
    acc_ls = []
    for i in range(args.ntasks):
        train_name = task_names[i]
        print('======================',train_name,'=======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=4)
        val_loaders.append(val_loader)
        if i == 0:
            # first task
            net,replay_set = train_task1(train_loader, args)
            acc = test(val_loader, net, args)
            acc_ls.append(acc)
        else:
            net,replay_set = train_task2(train_loader, replay_set, net, args, i+1)
            acc_ave = 0
            for val in val_loaders:
                acc = test(val, net, args)
                acc_ave += acc
            acc_ave/=(i+1)
            print('average acc: ', acc_ave)
            acc_ls.append(acc_ave)
    return acc_ls

    
if __name__ == '__main__':
    filename = args.filename
    file = open(filename, "a") 
    if args.dataset == 's-mnist':
        file.write('\n - - - Split_MNIST_CLDR(mlp) - - - \n')
    elif args.dataset == 'p-mnist':
        file.write('\n - - - Permuted_MNIST_CLDR (mlp) - - - \n')
    elif args.dataset == 'cifar10':
        file.write('\n - - - Split_CIFAR10_CLDR (cnn) - - - \n')
    elif args.dataset == 'cifar100':
        file.write('\n - - - Split_CIFAR100_CLDR (cnn) - - -  \n')
    file.write('**** same # of replay samples per task ***\n')
    if args.mixup_within_class:
        file.write(' - - - Mixup - Within - Same Class - - - \n')
    file.write(f'args.replay = {args.replay}\n')
    file.write(f'args.epochs = {args.epochs}\n')
    file.write(f'args.batch_size = {args.batch_size}\n')  
    file.write(f'args.lr = {args.lr}\n')
    file.write(f'args.optim = {args.optim}\n')  
    file.write(f'args.O1N1 = {args.O1N1}\n')
    file.write(f'args.criterion_mixup = {args.criterion_mixup}\n')
    file.write(f'args.mixup_within_class = {args.mixup_within_class}\n')
    file.write(f'args.num_classes = {args.num_classes}\n')
    file.write(f'args.ntasks = {args.ntasks}\n')    
    file.close()
    file = open(filename, "a") 
    file.write(f'\n alpha = {args.alpha}')
    file.write(f'\n beta = {args.beta}')
    file.write(f'\n gamma = {args.gamma}\n')
    file.write(f'\n alpha0 = {args.alpha0}')
    file.write('\n acc = ')
    acc_lst = []
    time1 = time.time()
    for runs_seed in range(args.runs):
        seed_torch(runs_seed)
        acc = main(args)
        acc_lst.append(acc)
        file.write(f'{acc}, ')
    time2 = time.time()
    time_run = (time2-time1)/60
    file.write(f'\n average_acc = {np.mean(acc_lst,0)}+-{np.std(acc_lst,0)}\n')
    file.write(f' Total running time is {time_run} minutes \n')
    file.close()



    