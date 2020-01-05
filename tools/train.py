import torch
import torchvision  
import torch.nn as nn 
import numpy as np
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data 
import torch.optim as optim
from datetime import datetime
import sys
sys.path.insert(0, '../')

import os 
import shutil 
import torchvision.datasets as Data 
import torch.utils.data as TD
import matplotlib.pyplot as plt 
import time 
from os import path 
from importlib import import_module
from utils.data_builder import Dataset_Loader
from utils.Model_builder import Model_builder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parser():
    parser = argparse.ArgumentParser(description='Parser for training')
    parser.add_argument('--cfg',default="configs/train_cifar10.py",help='Configuration file')
    args = parser.parse_args()
    return args

def count_parameters(model):
    total = 0
    trainable = 0
    for param in model.parameters():
        temp = param.numel()
        total += temp
        if param.requires_grad:
            trainable += temp
    print('Total params: {} | Trainable params: {}'.format(
        total, trainable
    ))

def get_run_id():
    """A run ID for a particular training session.
    """
    dt = datetime.now()
    run_id = dt.strftime('%m_%d_%H_%M')
    return run_id

def trainer(configer,model,Train_loader,Val_loader):

    Train_cfg = configer.train_cfg
    Current_cfg = dict()

    # Prepare optimizer
    optim_cfg = Train_cfg['optimizer']
    optim_name = optim_cfg.pop('name')
    print('### SELECTED OPTIMIZER:', optim_name)
    optim_cls = getattr(optim, optim_name)
    optimizer = optim_cls(model.parameters(), **optim_cfg)
    Current_cfg["Optimizer"] = optimizer

    # Prepare loss function(s)
    loss_cfg = Train_cfg.pop('criterion')
    loss_name = loss_cfg.pop('L1')
    print('### SELECTED LOSS FUNCTION:', loss_name)
    loss_cls = getattr(nn, loss_name)
    loss = loss_cls(**loss_cfg)
    Current_cfg["Loss_criterion"] = loss

    # Prepare learning rate scheduler if specified
    scheduler_cfg = Train_cfg.pop('scheduler', None)
    scheduler = None


    if scheduler_cfg is not None:
        scheduler_name = scheduler_cfg.pop('name')
        print('### SELECTED SCHEDULER:{}\n'.format(scheduler_name))
        scheduler_cls = getattr(optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_cls(optimizer, **scheduler_cfg)

    Current_cfg["scheduler"] = scheduler

    # Getting current run_id

    Current_cfg["Run_id"] = get_run_id()
    Current_cfg["Store_root"] = Train_cfg["training_store_root"]
    Current_cfg["DataParallel"] = configer.model["DataParallel"]

    # Loading Training configs 

    Epochs = Train_cfg["epochs"]
    Test_interval = Train_cfg["test_interval"]
    Current_cfg["Plot_Accuracy"] = Train_cfg["plot_accuracy_graphs"]

    # Setting accuracy and losses lists and constants
    Best_Val_accuracy = 0
    Train_accuracies = []
    Train_losses = [] 
    Val_accuracies = []
    Val_losses = [] 

    print ('---------- Starting Training')
    for i in range(Epochs):

        if (scheduler is not None) and (scheduler_name != 'ReduceLROnPlateau'):

            scheduler.step()

        model,Epoch_train_set_accuracy,Epoch_train_set_loss = Train_epoch(configer,model,Train_loader,Current_cfg,i)

        Train_accuracies.append(Epoch_train_set_accuracy)
        Train_losses.append(Epoch_train_set_loss)

        if (i%Test_interval) == 0:

            model,Epoch_Val_set_accuracy,Epoch_Val_set_loss = Val_epoch(configer,model,Val_loader,Current_cfg,i)           

            Val_accuracies.append(Epoch_Val_set_accuracy)
            Val_losses.append(Epoch_Val_set_loss)

            if Epoch_Val_set_accuracy > Best_Val_accuracy:
                print("Best Validation accuracy found uptil now !! Saving model state....")
                Best_Val_accuracy = Epoch_Val_set_accuracy

                Model_State_Saver(model,Current_cfg,Train_accuracies,Train_losses,Val_accuracies,Val_losses,i)

            if (scheduler is not None) and (scheduler_name == 'ReduceLROnPlateau'):

                scheduler.step(Epoch_Val_set_loss)

    Model_State_Saver(model,Current_cfg,Train_accuracies,Train_losses,Val_accuracies,Val_losses,i)


def Train_epoch(configer,model,Train_loader,Current_cfg,i):

    def get_count(outputs, labels):
        """Number of correctly predicted labels.
        """
        pred_labels = torch.argmax(outputs, dim=1)
        count = torch.sum(torch.eq(pred_labels, labels)).item()
        return count

    # Training essentials

    optimizer = Current_cfg['Optimizer']
    criterion = Current_cfg['Loss_criterion']
    Dataset = configer.dataset_cfg['id_cfg']['name']

    # Some useful constants
    num_train_batches = len(Train_loader)
    running_loss = 0
    Total_correct = 0
    Total_count = 0

    print('*' * 20, 'TRAINING', '*' * 20)

    start = time.time()
    model.train()

    for batch_idx, (Input, labels) in enumerate(Train_loader):

        Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
        Input = Input.float().to(device)
        labels = labels.to(device)

        outputs = model(Input)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        Total_correct += get_count(outputs, labels)
        Total_count += len(Input)

        print('Epoch: {} | Iter: {}/{} | Running loss: {:.3f} | Time elapsed: {:.2f}'
                ' mins'.format(i,batch_idx + 1, num_train_batches,
                                running_loss/(batch_idx + 1),
                                (time.time() - start) / 60), end='\r',
                flush=True)

        del Input, labels, outputs

    Epoch_avg_loss = float(running_loss)/num_train_batches
    Epoch_accuracy = (Total_correct/Total_count)*100

    print('\nTraining --> Acc: {:.3f}% | Loss: {:.3f}'.format(Epoch_accuracy, Epoch_avg_loss))

    return model,Epoch_accuracy,Epoch_avg_loss


def Val_epoch(configer,model,Val_loader,Current_cfg,i):  

    def get_count(outputs, labels):
        """Number of correctly predicted labels.
        """
        pred_labels = torch.argmax(outputs, dim=1)
        count = torch.sum(torch.eq(pred_labels, labels)).item()
        return count

    # Training essentials

    criterion = Current_cfg['Loss_criterion']
    Dataset = configer.dataset_cfg['id_cfg']['name']

    # Some useful constants
    num_train_batches = len(Val_loader)
    running_loss = 0
    Total_correct = 0
    Total_count = 0

    print('*' * 20, 'VALIDATING', '*' * 20)

    start = time.time()
    model.eval()

    for batch_idx, (Input, labels) in enumerate(Val_loader):

            Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
            Input = Input.float().to(device)
            labels = labels.to(device)

            outputs = model(Input)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            Total_correct += get_count(outputs, labels)
            Total_count += len(Input)

            print('Epoch: {} | Iter: {}/{} | Running loss: {:.3f} | Time elapsed: {:.2f}'
                  ' mins'.format(i,batch_idx + 1, num_train_batches,
                                 running_loss/(batch_idx + 1),
                                 (time.time() - start) / 60), end='\r',
                  flush=True)

            del Input, labels, outputs

    Epoch_avg_loss = float(running_loss)/num_train_batches
    Epoch_accuracy = (Total_correct/Total_count)*100

    print('\nValidate --> Acc: {:.3f}% | Loss: {:.3f}'.format(Epoch_accuracy, Epoch_avg_loss))

    return model,Epoch_accuracy,Epoch_avg_loss


def Model_State_Saver(model,Current_cfg,Train_accuracies,Train_losses,Val_accuracies,Val_losses,i):

    Store_root = Current_cfg["Store_root"]
    run_id = Current_cfg["Run_id"]

    if not os.path.isdir(os.path.join(Store_root,run_id)):
        os.mkdir(os.path.join(Store_root,run_id))
        os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states'))

        os.mkdir(os.path.join(Store_root,run_id,"Plots"))
        os.mkdir(os.path.join(Store_root,run_id,"Plots",'Combined'))

        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays"))
        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training'))
        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation'))

        shutil.copy('../'+args.cfg , os.path.join(Store_root,run_id,"Train_config.py"))

    # Saving model state dict
    if Current_cfg["DataParallel"]:
        torch.save(model.module.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}.pth".format(i)))
    else:
        torch.save(model.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}.pth".format(i)))

    # Saving Accuracy and loss arrays

    Train_accuracies = np.asarray(Train_accuracies)
    Val_accuracies = np.asarray(Val_accuracies)

    Train_losses = np.asarray(Train_losses)
    Val_losses = np.asarray(Val_losses)

    np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training',"Train_Accuracies.npy"),Train_accuracies)
    np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation',"Valid_Accuracies.npy"),Val_accuracies)

    np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training',"Train_losses.npy"),Train_losses)
    np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation',"Val_losses.npy"),Val_losses)

    # Plotting Training and Validation plots
    assert len(Train_losses) == len(Val_losses), "Loss arrays don't match !"
    Epochs = len(Train_losses)
    A, = plt.plot(np.arange(1,Epochs+1,1),Train_losses)
    B, = plt.plot(np.arange(1,Epochs+1,1),Val_losses)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Epoch Loss")
    plt.legend(["Training set loss","Validation Set loss"])
    plt.title("Train and Val Loss over epochs")
    plt.savefig(os.path.join(Store_root,run_id,"Plots",'Combined',"Loss_comparision.png"))
    plt.clf()

    assert len(Train_accuracies) == len(Val_accuracies), "Accuracy arrays don't match !"
    Epochs = len(Train_accuracies)
    A, = plt.plot(np.arange(1,Epochs+1,1),Train_accuracies)
    B, = plt.plot(np.arange(1,Epochs+1,1),Val_accuracies)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Percent Accuracy")
    plt.legend(["Training set Accuracy","Validation Set Accuracy"])
    plt.title("Train and Val Accuracy over epochs")
    plt.savefig(os.path.join(Store_root,run_id,"Plots",'Combined',"Accuracy_comparision.png"))
    plt.clf()


def main(args):
    filename = args.cfg
    module_name = path.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = path.dirname(filename)
    sys.path.insert(0, "../"+config_dir)
    configer = import_module(module_name)
    sys.path.pop(0)

    # Building Dataloaders
    Train_loader,Val_loader = Dataset_Loader(configer)

    # Building config DL model
    model = Model_builder(configer)
    count_parameters(model)

    # Resuming training from saved checkpoint
    if configer.Train_resume:

        Store_root = configer.train_cfg["training_store_root"]
        Load_run_id = configer.Load_run_id
        Load_Epoch = configer.Load_Epoch

        print("\n### Resuming training from config checkpoint ID {0} and Epoch {1}\n".format(Load_run_id,Load_Epoch))

        checkpoint_weights = torch.load(os.path.join(Store_root,Load_run_id,"Model_saved_states","Epoch_{}.pth".format(Load_Epoch)))

        if configer.model["DataParallel"]:
            model.module.load_state_dict(checkpoint_weights)        
        else:
            model.load_state_dict(checkpoint_weights)  

    elif configer.Validate_only:

        Store_root = configer.train_cfg["training_store_root"]
        Load_run_id = configer.Load_run_id
        Load_Epoch = configer.Load_Epoch
        model_name = configer.model["name"]

        print("\n### Validating Model:{0} from config checkpoint ID {1} and Epoch {2}\n".format(model_name,Load_run_id,Load_Epoch))

        Train_cfg = configer.train_cfg
        loss_cfg = Train_cfg['criterion']
        loss_name = loss_cfg.pop('L1')
        loss_cls = getattr(nn, loss_name)
        loss = loss_cls(**loss_cfg)
        Val_cfg = dict(Loss_criterion= loss)

        Val_epoch(configer,model,Val_loader,Val_cfg,0)      

        return 0  

    # Training the model for config settings

    trainer(configer,model,Train_loader,Val_loader)


if __name__ == "__main__":

    args = parser()
    main(args)
