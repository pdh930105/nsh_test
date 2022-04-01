import os, sys, shutil, time, random
import argparse


# ----------torch library load ------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import create_loader
import pandas as pd
import numpy as np
import pickle

## import quantized module

from utils import WarmUpLR

import models
from options import Option
from log_utils import make_logger, AverageMeter

import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(model, train_loader, optimizer, epoch, device, logger, warmup_scheduler=None):
    model.train()
    total_loss = AverageMeter()
    total_acc = AverageMeter()
    total_top5_acc = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        prec1, prec5 = accuracy(output.cpu().detach(), target.cpu().detach(), (1, 5))
        total_loss.update(loss)
        total_acc.update(prec1)
        total_top5_acc.update(prec5)
        loss.backward()
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        if batch_idx % 100 == 0:
            result_text= 'Train Epoch: [{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()) 
            logger.info(result_text)
    logger.info('Train Epoch: [{}]\t Average Loss: {:.6f}\t Total Acc : {:.4f}\t Total Top5 Acc : {:.4f}'.format(
                epoch, total_loss.avg, total_acc.avg, total_top5_acc.avg))
    print("===="*10)
    
    return total_acc.avg, total_top5_acc.avg, total_loss.avg


def test(model, test_loader, epoch, device, logger):
    model.eval()
    total_test_loss = AverageMeter()
    total_top5_acc = AverageMeter()
    total_top1_acc = AverageMeter()
    correct = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.cross_entropy(output, target)
            total_test_loss.update(test_loss)
            prec1, prec5 = accuracy(output.cpu().detach(), target.cpu().detach(), (1, 5))
            total_top1_acc.update(prec1)
            total_top5_acc.update(prec5)
            
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct.update((pred.eq(target.view_as(pred)).sum().item()))

    logger.info('\nEpoch [{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {:.4f}%\n'.format(
        epoch, total_test_loss.avg, correct.sum, len(test_loader.dataset),
        total_top1_acc.avg, total_top5_acc.avg))
    print("===="*10)
    return total_top1_acc.avg, total_top5_acc.avg, total_test_loss.avg

class Hook():
    def __init__(self, module, forward=True):
        if forward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


def batchnorm_hook_result(model, input_data, input_label, optimizer, option):
    optimizer.zero_grad()
    model.train()
    count = 0
    forward_hook_list = []
    backward_hook_list = []
    
    for i, (name, module) in enumerate(model.named_modules()):
        if (isinstance(module, nn.BatchNorm2d)) and "shortcut" not in name:
            if count in option.activation_index:
                temp_fwd_hook = Hook(module)
                temp_bwd_hook = Hook(module, forward=False)
                name_idx = f"idx_{count}_{name}"
                forward_hook_list.append((name_idx, temp_fwd_hook))
                backward_hook_list.append((name_idx, temp_bwd_hook))
            count+=1
    
    predict = model(input_data)
    loss = F.cross_entropy(predict, input_label)
    loss.backward()
    return forward_hook_list, backward_hook_list

def get_batchnorm_param_dict(net, epoch):
    count = 0
    save_bn_dict = {}
    save_bn_dict['epoch'] = epoch

    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'shortcut' not in name:
            save_bn_dict[f"{count}_{name}_alpha"] = m.weight.cpu().detach().numpy()
            save_bn_dict[f"{count}_{name}_beta"] = m.bias.cpu().detach().numpy()
            save_bn_dict[f"{count}_{name}_avg"] = m.running_mean.cpu().detach().numpy()
            save_bn_dict[f"{count}_{name}_var"] = m.running_var.cpu().detach().numpy()
            count+=1
    return save_bn_dict




def main():
    parser = argparse.ArgumentParser(description="resnet test")
    parser.add_argument("--conf_path", type=str, help="hocon config path")
    parser.add_argument("--resume", action="store_true", dest='resume', default=False, help="load pkt and using retraining")
    parser.add_argument("--gpu_num", type=int, default=0, help="select gpu num")

    args = parser.parse_args()

    option = Option(args.conf_path, args)
    torch.manual_seed(option.seed)
    torch.cuda.manual_seed(option.seed)
    np.random.seed(option.seed)

    
    if option.dataset.lower() == "cifar100":
        cifar100_path = os.path.join(option.data_path, "CIFAR100")
        train_loader, test_loader, n_classes, image_size = create_loader(option.batch_size, cifar100_path, option.dataset)
    elif option.dataset.lower() == "imagenet":
        train_loader ,test_loader, n_classes, image_size = create_loader(option.batch_size, option.data_path, option.dataset)
    
    else : 
        AssertionError("please select dataset cifar100|imagenet")
    

    if option.model_name.lower() in model_names:
        net = models.__dict__[option.model_name.lower()](num_classes=n_classes)

    else:
        print(option.model_name)
        raise AssertionError("This test only using resnet18im")

    device = torch.device(f'cuda:{args.gpu_num}')
    net = net.to(device)

    
    if option.optimizer.lower() == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=option.lr, momentum=option.momentum, nesterov=option.nesterov)
    
    elif option.optimizer.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr= option.lr)
    
    if option.scheduler.lower() == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=option.ml_step, gamma=option.lr_gamma)
    
    if option.warmup > 0:
        warmup_scheduler = WarmUpLR(optimizer, total_iters=option.warmup*len(train_loader))

    if args.resume:
        option.log_override = False
        option.set_save_path()
        checkpoint = torch.load(os.path.join(option.save_path, "last_checkpoint.pth"))
        start_epoch = checkpoint['end_epoch']+1
        
        if start_epoch < 5:
            print("re-train for using warmup train")
            option.log_override = True
            option.set_save_path()
            start_epoch = 0   
        else :
            print(f"load pretrained model : epoch {start_epoch}")
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print(f"train {option.model_name}")
        start_epoch = 0
        option.log_override = True
        option.set_save_path()

    logger = make_logger("train", os.path.join(option.save_path, "train.log"))
    writer = SummaryWriter(os.path.join(option.save_path, "tfboard_result"))


    random_sampler= torch.utils.data.RandomSampler(test_loader.dataset)
    sample_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=128, sampler=random_sampler)

    check_data, check_label = next(iter(sample_loader))
    check_data = check_data.to(device)
    check_label = check_label.to(device)

    ones_shape = [option.batch_size, 3, 224, 224] if option.dataset.lower() == "imagenet" else [option.batch_size, 3, 32, 32]
    dummy_input = torch.ones(ones_shape).to(device) * 0.1


    with torch.no_grad():
        net.eval()
        writer.add_graph(net, dummy_input)

    #del dummy_input
    #del ones_shape

    csv_path = os.path.join(option.save_path, "batchnorm_param.csv")

    batchnorm_df = pd.DataFrame()

    logger.info(f"-------start batchnorm param logging -----------\n")

    save_bn_dict = get_batchnorm_param_dict(net, epoch=-1)
    batchnorm_df = batchnorm_df.append(save_bn_dict, ignore_index=True)
    batchnorm_df.to_csv(csv_path)

    logger.info(f"-------end batchnorm param logging -----------\n")

    best_test_acc = 0
    best_epoch = 0
    save_best_acc_path = os.path.join(option.save_path, "best_checkpoint.pth")

    if -1 in option.activation_index:
        idx_count = 0
        for name, module in net.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "shortcut" not in name:
                idx_count+=1
        option.activation_index = range(idx_count)


    print(option.print_parameters())
    for epoch in range(start_epoch, option.epochs):
        logger.info(f"-------{epoch} epoch start-----------")

        if epoch in option.activation_step:
            logger.info(f"-------- logging {epoch} batch layer input tensor ------------------")
            result_fwd_hook_list, result_bwd_hook_list = batchnorm_hook_result(net, check_data, check_label, optimizer, option)

            for hook in result_fwd_hook_list:
                name = hook[0]
                batch_input = hook[1].input[0].cpu().detach()
                batch_output = hook[1].output.cpu().detach()
                print(f"batch_input shape : {batch_input.shape}")
                print(f"batch_output shape : {batch_output.shape}")
                save_input_pkl_path = os.path.join(option.save_path, name, f"{epoch}_fwd_input.pkl")
                if not os.path.exists(os.path.join(option.save_path, name)):
                    os.makedirs(os.path.join(option.save_path, name))
                save_output_pkl_path = os.path.join(option.save_path, name, f"{epoch}_fwd_act.pkl")                
                
                with open(save_input_pkl_path, "wb") as fw:
                    pickle.dump(batch_input, fw)
                #with open(save_output_pkl_path, "wb") as bw:
                #    pickle.dump(batch_output, bw)
                hook[1].close()
            
            for hook in result_bwd_hook_list:
                name = hook[0]
                #batch_input = hook[1].input[0].cpu().detach()
                batch_output = hook[1].output[0].cpu().detach()
                #print(f"batch_grad_input shape : {batch_input.shape}")
                #print(f"batch_grad_input tuples shape : {[len(f) for f in hook[1].input]}")
                print(f"batch_grad_output shape : {batch_output.shape}")
                print(f"batch_grad_output tuples shape : {[len(f) for f in hook[1].output]}")
                
                save_output_pkl_path = os.path.join(option.save_path, name, f"{epoch}_act_grad.pkl")                
                #with open(save_input_pkl_path, "wb") as fw:
                #    pickle.dump(batch_input, fw)
                with open(save_output_pkl_path, "wb") as bw:
                    pickle.dump(batch_output, bw)
                hook[1].close()
            logger.info(f"-------- logging end {epoch} --------------------")        
        
            if option.get_weight_param:
                count=0
                for i, (name, module) in enumerate(net.named_modules()):
                    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "shortcut" not in name:
                        if count in option.activation_index:
                            name_idx = f"idx_{count}_{name}"
                            save_weight_pkl_path = os.path.join(option.save_path, name_idx, f"{epoch}_weight.pkl")
                            with open(save_weight_pkl_path, "wb") as bw:
                                pickle.dump(module.weight.cpu().detach(), bw)
                            count+=1
            
            if option.get_weight_grad_param:
                count=0
                for i, (name, module) in enumerate(net.named_modules()):
                    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "shortcut" not in name:
                        if count in option.activation_index:
                            name_idx = f"idx_{count}_{name}"
                            save_weight_pkl_path = os.path.join(option.save_path, name_idx, f"{epoch}_weight_grad.pkl")
                            with open(save_weight_pkl_path, "wb") as bw:
                                pickle.dump(module.weight.grad.cpu().detach(), bw)
                            count+=1
        

        if epoch < option.warmup:
            train_acc, train_top5_acc, train_loss = train(net, train_loader, optimizer, epoch, device, logger, warmup_scheduler)
        else:
            train_acc, train_top5_acc, train_loss = train(net, train_loader, optimizer, epoch, device, logger)
        print(f"-------{epoch} epoch end  -----------\n")

        scheduler.step()
        logger.info(f"-------{epoch} epoch end-----------")
        
        print("----- test and print accuracy ------------------")
        test_acc, test_top5_acc, test_loss=test(net, test_loader, epoch, device, logger)
        writer.add_scalars("Loss", {"train_loss" : train_loss, "test_loss" :test_loss}, epoch)
        writer.add_scalars("Accuracy", {"train_Acc_Top1" : train_acc,
                                        "test_Acc_Top1" :test_acc,
                                        "train_Acc_Top5" : train_top5_acc,
                                        "test_Acc_Top5": test_top5_acc
                                        }, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        print("----- test end -------------------------")
        print("\n")
        logger.info(f"save intermediate epoch [{epoch}] result\n\n")
        save_state_dict_path = os.path.join(option.save_path, f"last_checkpoint.pth")
        save_bn_dict = get_batchnorm_param_dict(net, epoch=epoch)
        batchnorm_df = batchnorm_df.append(save_bn_dict, ignore_index=True)
        batchnorm_df.to_csv(csv_path)

        if test_acc > best_test_acc:
            logger.info(f"logging best performance {epoch} epoch")
            print(f"logging best performance {epoch} epoch")
            torch.save(net.state_dict(), save_best_acc_path)
            best_epoch = epoch
            best_test_acc = test_acc
            writer.add_scalar("Best Test Acc", best_test_acc, best_epoch)
            
        torch.save({
            'end_epoch': epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict()
        }, save_state_dict_path)


if __name__ == '__main__':
    main()
