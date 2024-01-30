import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
import datetime
import json
import collections
import pathlib
import copy
from tqdm import tqdm
from utils import *
from load_dataset import *
import os
from wideresnet import WideResNet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description='SAFE-STUDENT')
parser.add_argument('--lr', default=0.128, type=float, help='learning rate')
parser.add_argument('--warm_up', default=1000, type=int, help='number of epochs before main training starts')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset CIFAR10')
parser.add_argument('--outdir', default='results/', type=str, help='Output directory')
parser.add_argument('--model', default='WideResnet', type=str, help='WideResnet')
parser.add_argument('--batch_size', default=256, type=int, help='Training batch size.')
parser.add_argument('--ts_iteration', default=3, type=int, help='number of student to teacher switch iterations')
parser.add_argument('--n_labels', type=int, default=2400)
parser.add_argument('--n_unlabels', type=int, default=20000)
parser.add_argument('--n_valid', type=int, default=5000)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--ratio', type=float, default=0.6)
parser.add_argument('--name', default='SAFE-STUDENT', type=str, help='Name of the experiment')

def create_model(model_name):
    print('==> Building model..')
    if model_name == 'WideResnet':
        model = WideResNet(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
    return model

def warmup(epoch, model, trainloader):
    model.train()

    wqk_train_loss= []
    correct = 0
    total = 0
    trainloader = tqdm(trainloader)

    trainloader.set_description('[%s %04d/%04d]' % ('warmup', epoch, args.warm_up))



    for batch_idx, (inputs, inputs_noaug, target, dataset_index) in enumerate(trainloader):

        inputs, target = inputs.to(device), target.long().to(device)
        optimizer_teacher.zero_grad()
        outputs1 = model(inputs)
        loss_1 = criterion(outputs1, target)
        loss_1.backward()
        optimizer_teacher.step()


        wqk_train_loss.append(loss_1.item())
        _, predicted = outputs1.max(1)

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        total_acc = correct / total

        postfix = {}
        postfix['loss'] = sum(wqk_train_loss)/len(wqk_train_loss)
        postfix['acc'] = total_acc

        postfix['lr'] = optimizer_teacher.param_groups[0]['lr']
        trainloader.set_postfix(postfix)

    total_loss = sum(wqk_train_loss)/len(wqk_train_loss)
    total_acc = correct / total

    log = collections.OrderedDict({
        'epoch': epoch,
        'train':
            collections.OrderedDict({
                'loss': total_loss,
                'accuracy': total_acc,
            }),
    })
    return log

def test(epoch, model, testloader, total_epoch):
    global best_acc
    model.eval()

    wqk_test_loss = []
    correct = 0
    total = 0
    testloader = tqdm(testloader)
    testloader.set_description('[%s %04d/%04d]' % ('*test', epoch, total_epoch))

    with torch.no_grad():
        for batch_idx, (inputs, target, data_index) in enumerate(testloader):
            inputs, target = inputs.to(device), target.long().to(device)
            outputs1 = model(inputs)
            loss1 = criterion(outputs1, target)
            wqk_test_loss.append(loss1.item())

            _, predicted = outputs1.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            total_acc = correct / total

            postfix = {}
            postfix['loss'] = sum(wqk_test_loss) / len(wqk_test_loss)
            postfix['acc'] = total_acc
            testloader.set_postfix(postfix)

    total_loss = sum(wqk_test_loss) / len(wqk_test_loss)
    total_acc = correct / total
    log = collections.OrderedDict({
        'epoch': epoch,
        'test':
            collections.OrderedDict({
                'loss': total_loss,
                'accuracy': total_acc,
            }),
    })
    return log, total_acc

if __name__ == "__main__":
    print("this is start")
    args = parser.parse_args()
    dataset_name = args.dataset.lower()
    print(args.__dict__)

    args.outdir = args.outdir + args.name + '/'


    outdir = pathlib.Path(args.outdir + '_'.join(s for s in [args.model, args.dataset]))


    outdir.mkdir(exist_ok=True, parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0

    l_train_set, u_train_set, validation_set, test_set= get_dataloaders(dataset=args.dataset,
                                                                                  n_labels=args.n_labels,
                                                                                  n_unlabels=args.n_unlabels,
                                                                                  n_valid=args.n_valid,
                                                                                  tot_class=args.n_class,
                                                                                  ratio=args.ratio)


    labeled_loader = torch.utils.data.DataLoader(l_train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = torch.utils.data.DataLoader(u_train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=2, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              drop_last=False)

    model_list = []
    batch_sizes = []

    for i in range(args.ts_iteration):
        model_list.append(args.model)# WideResnet
        batch_sizes.append(args.batch_size)# 256


    model_teacher = create_model(model_list[0])
    model_student = create_model(model_list[1])


    print(model_list[:args.ts_iteration + 1])
    start_date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d")

    model_teacher = model_teacher.to(device)
    model_student = model_student.to(device)


    cudnn.benchmark = True

    optimizer_teacher = optim.SGD(model_teacher.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                  nesterov=True,
                                  dampening=0)
    scheduler_teacher = torch.optim.lr_scheduler.StepLR(optimizer_teacher, step_size=5, gamma=0.97)


    optimizer_student = optim.SGD(model_student.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                  nesterov=True,
                                  dampening=0)
    scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=5, gamma=0.97)

    criterion = nn.CrossEntropyLoss()

    exp_logs = []
    exp_info = collections.OrderedDict({
        'model': model_list,
        'type': 'default',
        'arguments': args.__dict__,
    })

    exp_log = exp_info.copy()
    exp_logs.append(exp_log)
    save_json_file_withname(outdir, args.name, exp_logs)

    for epoch in range(args.warm_up):
        train_log = warmup(epoch, model_teacher, labeled_loader)
        exp_log = train_log.copy()
        if epoch % 10 == 0 and epoch != 0:
            test_log, acc = test(epoch, model_teacher, validation_loader, args.warm_up)
            exp_log.update(test_log)
            if (acc > best_acc):
                best_acc = acc
                torch.save(model_teacher.state_dict(), os.path.join("./save_model/", f"pretrain_teacher_WideResnet_{dataset_name}.pth"))
        scheduler_teacher.step()
        exp_logs.append(exp_log)
        save_json_file_withname(outdir, args.name, exp_logs)
