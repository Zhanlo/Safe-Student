import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import logging
import time

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

from matplotlib import pyplot as plt
from tqdm import tqdm


from utils import *
from load_dataset import *
from wideresnet import WideResNet

parser = argparse.ArgumentParser(description='SAFE-STUDENT')
parser.add_argument('--lr', default=0.128, type=float, help='learning rate')
parser.add_argument('--epochs', default=400, type=int, help='Total number of epochs')
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
parser.add_argument('--tau_1', type=float, default=0.6)
parser.add_argument('--tau_2', type=float, default=0.1)
parser.add_argument('--lambda_1', type=float, default=0.3)
parser.add_argument('--lambda_2', type=float, default=0.05)



def create_model(model_name):
    logging.info('==> Building model..')

    if model_name == 'WideResnet':
        model = WideResNet(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
    return model

def kl_div(p_output, get_softmax=False):

    if get_softmax:
        p_output = F.softmax(p_output)
    uniform_output = p_output*0+(1/args.n_class)
    uniform_output = F.softmax(uniform_output,dim=-1)

    return F.kl_div(p_output, uniform_output, reduction='none')

def calculate_natigative_logit(list_val):
    T = 10
    total = 0
    for ele in range(0, len(list_val)):
        total = total + list_val[ele]
    return T * np.log(total)

def calculate_natigative_logit2(list_val):
    T = 10
    total = 0
    for ele in range(0, len(list_val)):
        total = total + list_val[ele]
    maxe = np.max(list_val)
    total = total-maxe
    return T * np.log(total)

# Training
def train_student(epoch, model_student, model_teacher, labeled_loader, unlabeled_loader):
    model_student.train()
    model_teacher.eval()

    wqk_train_loss = []
    correct = 0
    total = 0
    labeled_loader = tqdm(labeled_loader)
    labeled_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.epochs))


    iter_u = iter(unlabeled_loader)

    for batch_idx, (inputs, inputs_noaug, target, dataset_index) in enumerate(labeled_loader):
        try:
            inputs_u, inputs_noaug_u, target_u, index_u = next(iter_u)
        except StopIteration:
            iter_u = iter(unlabeled_loader)
            inputs_u, inputs_noaug_u, target_u, index_u = next(iter_u)

        inputs_u = inputs_u.to(device)
        inputs_noaug_u = inputs_noaug_u.to(device)

        with torch.no_grad():
            pseudo_logit = model_teacher(inputs_noaug_u)
            pseudo_label = F.softmax(pseudo_logit, dim=1).detach()
            confidence, targets_u = torch.max(pseudo_label, dim=-1)

            logit_energy = pseudo_logit.detach().cpu().numpy()
            T = 10
            list_logit_ori = [np.exp(x / T) for _, x in enumerate(logit_energy)]


            energy_ori = [calculate_natigative_logit(yi) for _, yi in enumerate(list_logit_ori)]


            energy_upd = [calculate_natigative_logit2(yi) for _, yi in enumerate(list_logit_ori)]


            energy_gain = energy_ori
            for i in range(len(energy_upd)):
                energy_gain[i] = energy_ori[i] - energy_upd[i]



            threshold1_energy_gain = np.percentile(energy_gain, args.tau_1 * 100)
            threshold2_energy_gain = np.percentile(energy_gain, args.tau_2 * 100)


            energy_gain = torch.Tensor(energy_gain).cpu().cuda()
            gt_mask = (energy_gain > threshold1_energy_gain).float()


            gt = gt_mask[:, None]
            mask = gt >= 1


            new_mask = (energy_gain < threshold2_energy_gain).float()

            new = new_mask[:, None]
            mask2 = new >= 1
            ###################

        outputs00 = model_student(inputs_u)

        outputs0 = F.log_softmax(outputs00, dim=1)


        loss_cbe1 = 0
        for i in range(len(targets_u)):
            loss_cbe1 = loss_cbe1 + F.cross_entropy(outputs00[i:i + 1], targets_u[i:i + 1]) * mask[i]

        loss_cbe1 = loss_cbe1 / torch.sum(mask)

        loss_cbe2 = F.kl_div(outputs0, pseudo_label, reduction='none')

        loss_cbe2 = torch.mean(torch.sum(loss_cbe2, dim=1) * mask)


        loss_cbe = loss_cbe1 + loss_cbe2

        inputs, target = inputs.to(device), target.long().to(device)
        optimizer_student.zero_grad()

        outputs1 = model_student(inputs)
        loss_ce = criterion(outputs1, target)

        loss_ucd = kl_div(outputs0)

        loss_ucd = torch.mean(torch.sum(loss_ucd, dim=1) * mask2)


        loss = loss_ce + loss_cbe * args.lambda_1 + loss_ucd * args.lambda_2

        loss.backward()
        optimizer_student.step()

        wqk_train_loss.append(loss.item())

        _, predicted = outputs1.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        total_acc = correct / total
        postfix = {}
        loss = sum(wqk_train_loss)/len(wqk_train_loss)
        postfix['loss'] = loss
        postfix['acc'] = total_acc
        lr = optimizer_student.param_groups[0]['lr']
        postfix['lr'] = lr
        labeled_loader.set_postfix(postfix)
        logging.info('[Training] iter-epoch-batch={}-{}-{}, loss={:.4f}, acc={:.4f}, lr={:.4f}' \
                     .format(i, epoch, batch_idx, loss, total_acc, lr))

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

def test_teacher(iter, total_iter, model_teacher, test_loader):
    model_teacher.eval()

    wqk_test_loss = []
    correct = 0
    total = 0
    testloader = tqdm(test_loader)
    testloader.set_description('[%s %04d/%04d]' % ('*test', iter, total_iter))

    with torch.no_grad():
        for batch_idx, (inputs, target, dataset_index) in enumerate(test_loader):
            inputs, target = inputs.to(device), target.long().to(device)
            outputs1 = model_teacher(inputs)
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

    total_loss_Multi_classification = sum(wqk_test_loss) / len(wqk_test_loss)
    total_acc_Multi_classification = correct / total
    log_Multi_classification = collections.OrderedDict({
        'iter': iter,
        'test_teacher':
            collections.OrderedDict({
                'loss': total_loss_Multi_classification,
                'accuracy': total_acc_Multi_classification,
            }),
    })
    return log_Multi_classification, total_acc_Multi_classification





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
    logging.info('[Testing] epoch/total_epoch={}/{}, loss={:.4f}, acc={:.4f}' \
                 .format(epoch, total_epoch, total_loss, total_acc))

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
    start_time = time.time()
    start_time_formatted = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    file_name = f'results/log/{start_time_formatted}_info.log'

    # 获取 root logger
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=file_name, level=logging.DEBUG)
    print("logging")
    logging.info('logging')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f'start_time: {start_time_formatted}')


    args = parser.parse_args()
    dataset_name = args.dataset.lower()
    args.file_name = file_name
    logging.info(args.__dict__)
    print(args.__dict__)

    args.outdir = args.outdir + args.name + '/'


    outdir = pathlib.Path(args.outdir + '_'.join(s for s in [args.model, args.dataset]))

    outdir.mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    test_acc = 0

    l_train_set, u_train_set, validation_set, test_set = get_dataloaders(dataset=args.dataset,
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


    model_list = ['WideResnet', 'WideResnet', 'WideResnet', 'WideResnet']
    batch_sizes = [args.batch_size, args.batch_size, args.batch_size, args.batch_size]


    model_teacher = create_model(model_list[0])
    model_student = create_model(model_list[1])

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

    logging.info('load the pretrain model')
    model_teacher.load_state_dict(torch.load(os.path.join("./save_model/", f"pretrain_teacher_WideResnet_{dataset_name}.pth")), strict=True)
    model_student.load_state_dict(torch.load(os.path.join("./save_model/", f"pretrain_teacher_WideResnet_{dataset_name}.pth")), strict=True)
    # 加载预训练模型
    logging.info(
        f'pretrain model load_path{os.path.join("./save_model/", f"pretrain_teacher_WideResnet_{dataset_name}.pth")}')


    train_loss_list = []
    test_loss_list = []
    valid_loss_list = []

    train_acc_list = []
    test_acc_list = []
    test_epoch_list = []
    valid_acc_list = []

    test_teacher_acc = []



    for i in range(args.ts_iteration):
        logging.info('\n[{}/{}] iterative training on student'.format(i + 1, args.ts_iteration))

        for epoch in range(args.epochs):
            train_log = train_student(epoch, model_student, model_teacher, labeled_loader, unlabeled_loader)
            train_loss_list.append(train_log['train']['loss'])
            train_acc_list.append(train_log['train']['accuracy'])

            exp_log = train_log.copy()


            if epoch % 10 == 0 and epoch != 0:
                valid_log, acc = test(epoch, model_student, validation_loader, args.epochs)
                valid_loss_list.append(valid_log['test']['loss'])
                valid_acc_list.append(valid_log['test']['accuracy'])

                if (acc > best_acc):
                    best_acc = acc
                    test_log, test_acc = test(epoch, model_student, test_loader, args.epochs)
                    test_loss_list.append(test_log['test']['loss'])
                    test_acc_list.append(test_log['test']['accuracy'])
                    test_epoch_list.append(epoch)

                    save_path = f'{dataset_name}_student.pth'
                    torch.save(model_student.state_dict(), os.path.join("./save_model/", save_path))
                exp_log.update(valid_log)
                exp_log.update(test_log)
            scheduler_student.step()
            exp_logs.append(exp_log)
            save_json_file_withname(outdir, args.name, exp_logs)


        if i!=args.ts_iteration:
            log, acc = test_teacher(i + 1, args.ts_iteration + 1, model_teacher, test_loader)
            exp_logs.append(log)
            save_json_file_withname(outdir, args.name, exp_logs)
            test_teacher_acc.append(acc)


        if i != args.ts_iteration - 1:
            model_teacher = create_model('WideResnet')

            cudnn.benchmark = True
            save_path = f'{dataset_name}_student.pth'
            model_teacher.load_state_dict(
                torch.load(os.path.join("./save_model/", save_path)))
            model_teacher = model_teacher.to(device)

            optimizer_student = optim.SGD(model_student.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                          nesterov=True, dampening=0)
            scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=5, gamma=0.97)


    lists = [train_loss_list, test_loss_list, valid_loss_list,
             train_acc_list,  valid_acc_list, test_teacher_acc]
    labels = ['train_loss', 'test_loss', 'valid_loss',
              'train_acc',  'valid_acc', 'test_teacher_acc']

    for lst, label in zip(lists, labels):
        plt.figure()
        plt.plot(lst, label=label, marker='o')

        max_value = max(lst)
        max_index = lst.index(max_value)
        min_value = min(lst)
        min_index = lst.index(min_value)
        plt.annotate(round(max_value, 4), (max_index, max_value))
        plt.annotate(round(min_value, 4), (min_index, min_value))

        plt.ylabel(label)
        if label == 'test_teacher_acc':
            plt.xlabel('iter_test_teacher_acc')
        else:
            plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/log/{int(start_time)}_{label}.png')
        plt.close()


    lists = [test_epoch_list, test_acc_list]
    labels = ['test_epoch', 'test_acc']

    for lst, label in zip(lists, labels):
        plt.figure()
        plt.plot(lst, label=label, marker='o')
        for i, value in enumerate(lst):
            plt.annotate(round(value, 4), (i, value))

        plt.ylabel(label)
        plt.xlabel('iter')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/log/{int(start_time)}_{label}.png')
        plt.close()
        logging.info(label)
        logging.info(lst)

    save_json_file_withname(outdir, args.name, exp_logs)

    name = [str(start_time_formatted)]
    setting = [f'ratio: {args.ratio}', dataset_name, f'iteration: {args.ts_iteration}']
    lists = [test_teacher_acc, test_acc_list[-4:], setting, name]
    labels = ['test_teacher_acc', 'test_acc', 'setting', 'start_time_formatted']


    data_dict = dict(zip(labels, lists))
    for key in data_dict:
        data_dict[key] = [str(data_dict[key])]

    df_single_element = pd.DataFrame(data_dict)
    csv_path = "results/log/safe-student.csv"

    if os.path.exists(csv_path):
        df_single_element.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_single_element.to_csv(csv_path, index=False)

    end_time = time.time()
    logging.info("total time: " % (end_time - start_time))