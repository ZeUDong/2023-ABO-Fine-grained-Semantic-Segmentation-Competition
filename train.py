import torch
import torch.utils.data as data_utils 
import signal 
import sys 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging 
import math 
import json 
import time 
import utils
import utils.workspace as ws
from torch import nn 
import matplotlib.pyplot as plt
import torch.nn.functional as F

HIER_DATA_PATH='./hier'




class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


# def get_learning_rate_schedules(specs):

#     schedule_specs = specs["LearningRateSchedule"]

#     schedules = []

#     for schedule_specs in schedule_specs:

#         if schedule_specs["Type"] == "Step":
#             schedules.append(
#                 StepLearningRateSchedule(
#                     schedule_specs["Initial"],
#                     schedule_specs["Interval"],
#                     schedule_specs["Factor"],
#                 )
#             )
#         elif schedule_specs["Type"] == "Warmup":
#             schedules.append(
#                 WarmupLearningRateSchedule(
#                     schedule_specs["Initial"],
#                     schedule_specs["Final"],
#                     schedule_specs["Length"],
#                 )
#             )
#         elif schedule_specs["Type"] == "Constant":
#             schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

#         else:
#             raise Exception(
#                 'no known learning rate schedule of type "{}"'.format(
#                     schedule_specs["Type"]
#                 )
#             )

#     return schedules

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def init_seeds(seed=0):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def jitter_shape(b_shapes,scale_weight,noise_weight):
    scale = torch.randn(b_shapes.shape[0], b_shapes.shape[1]) * scale_weight + 1 
    noise = torch.randn(b_shapes.shape) * noise_weight 
    a_shapes=(b_shapes * scale.unsqueeze(2).cuda()) + noise.cuda()
    return a_shapes


def labelSmooth(one_hot, label_smooth):
    return one_hot*(1-label_smooth)+label_smooth/one_hot.shape[1]

class MyCrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=0, class_weight=None):
        super().__init__()
        self.class_weight = class_weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7
        
    def forward(self, x, y):

        x = x.permute(0,2,1)
        # print(y.shape,y)
        y_mask = torch.ones_like(y)
        non_index =(y==-1)
        y_mask[non_index] = 0
        #print(y_mask.shape,y_mask)

        y[y==-1] = 0
        y = F.one_hot(y, x.shape[2])
        # print(y.shape, y_mask.shape)
        if self.label_smooth:
            y = labelSmooth(y, self.label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        x_softmax = F.softmax(x, 1)
        # print(x_softmax.shape)
        x_softmax = torch.clamp(x_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        x_softmaxlog = torch.log(x_softmax)

        # original CE loss
        loss = -y * x_softmaxlog

        if self.class_weight:
            loss = loss*self.class_weight

         # if gamma:
         #    loss = loss*(y*(1-x_sigmoid)**gamma+(1-y)*(x_sigmoid)**gamma)
        #print(loss.shape, y_mask.shape)
        loss_all = 0
        for i in range(x.shape[0]):

	        y_mask0 = y_mask[i].unsqueeze(-1)
	        loss0 = loss[i]
	        loss_all += torch.sum(y_mask0*loss0) / torch.sum(y_mask0)
        # = y_mask*loss
        #loss = torch.mean(torch.mean(loss, -1))
        return loss_all/x.shape[0]




def main_function(experiment_directory, category, stop_threshold=250):
    init_seeds()
    specs=ws.load_experiment_specifications(experiment_directory)
    
    data_source=specs["DataSource"]
    arch=__import__("networks."+specs["NetworkArch"], fromlist=["DGCNNModel"])
    print('using network,',specs["NetworkArch"])
    checkpoints=list(range(specs["SnapshotFrequency"], specs["NumEpochs"] + 1, specs["SnapshotFrequency"]))

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)

    checkpoints.sort()
    # lr_schedules=get_learning_rate_schedules(specs)
    

    def save_checkpoints(epoch):
        ws.save_model_parameters(experiment_directory,str(epoch)+".pth",encoder,optimizer_all,epoch)
    
    def save_checkpoints_best(epoch):
        ws.save_model_parameters(experiment_directory,"pretrain.pth",encoder,optimizer_all,epoch)
    
    def signal_handler(sig,frame):
        logging.info("Stopping early...")
        sys.exit(0)

    # def adjust_learning_rate(lr_schedules,optimizer,epoch):
    #     for i,param_group in enumerate(optimizer.param_groups):
    #         param_group["Lr"]= lr_schedules[0].get_learning_rate(epoch)
    
    signal.signal(signal.SIGINT,signal_handler)

    cate_path=os.path.join(HIER_DATA_PATH,category)
    label_file=f'{cate_path}/level-2.txt'
    file1=open(label_file,'r')
    lines = file1.readlines()
    max_label= len(lines)
    max_size=150

    scale_weight=0.1
    noise_weight=0.01
    print(f' max part size, {max_size}, max_label {max_label}, scale_weight {scale_weight}, noise_weight {noise_weight}')
    encoder = arch.DGCNNModel(output_channels=max_label,d_model=512,dropout=0.4,max_part_n=max_size,input_channels=6).cuda()
        
    num_epochs = specs["NumEpochs"]
    data_source=os.path.join(data_source,category)
    train_dataset = utils.dataloader.GTSamples_abo(data_source, 'train', f'abo_{category}_train.npz', max_label, max_size)

    scene_per_batch_train = 2

    if len(train_dataset) < 6:
        scene_per_batch_train= len(train_dataset)
    
    train_loader = data_utils.DataLoader(train_dataset,
        batch_size=scene_per_batch_train, 
        shuffle=True, 
        num_workers=8, 
        drop_last=True)

    num_scenes= len(train_dataset)
    logging.info("There are {} shapes".format(num_scenes))

    #logging.debug(encoder)
    # optimizer_all=torch.optim.Adam(
    #     [
    #     {"params":encoder.parameters(),
    #     "lr": lr_schedules[0].get_learning_rate(0), 
    #     "betas":(0.5,0.999)}
    #     ]
    # )
    optimizer_all=torch.optim.Adam(
        [
        {"params":encoder.parameters(),
        "lr": 0.001, 
        "weight_decay":0.0001,
        "betas":(0.5,0.999)}
        ]
    )
    lr_schedules = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_all,
                                                                 T_0=250, 
                                                                T_mult=1)
    start_epoch=0
    logging.info("starting from epoch {}".format(start_epoch))
    
    start_time = time.time()
    last_epoch_time = 0 
    best_loss=999 
    best_epoch = 0
    #best_acc=0
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    #loss = MyCrossEntropyLoss()

    ave_log_epoch=1 
    loss_log = []
    epoch_log = []
    stop_count=0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # adjust_learning_rate(lr_schedules,optimizer_all,epoch-start_epoch)
        #print('lr,', lr_schedules[0].get_learning_rate(epoch-start_epoch))
        print('lr,', optimizer_all.param_groups[0]["lr"])

        avarage_total_loss=0
        avarage_num_train=0 

        iters = 0 

        for pcs, labels, masks, segments, part_nums, names in train_loader:

            pcs = pcs.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            segments = segments.cuda()
            
            iters += 1
            optimizer_all.zero_grad()
            pcs[:,:,:3] = jitter_shape(pcs[:,:,:3], scale_weight, noise_weight)

            pred = encoder(pcs, masks, segments)
            # print(pred.shape, labels.shape)
            # print(pred)
            # print(labels)
            total_loss = loss(pred, labels)
            total_loss.backward()

            optimizer_all.step()
            avarage_total_loss += total_loss.detach().item()
            
            avarage_num_train += 1
            
        if (epoch) % ave_log_epoch == 0:
            seconds_elapsed = time.time() - start_time
            ava_epoch_time = (seconds_elapsed - last_epoch_time) / ave_log_epoch 
            left_time = ava_epoch_time * (num_epochs + start_epoch - epoch)/3600 
            last_epoch_time = seconds_elapsed
            t_loss = avarage_total_loss / avarage_num_train
            loss_log.append(t_loss)
            epoch_log.append(epoch-start_epoch)
            print("epoch={}/{},total_loss={:.6f}, 1 epoch time={:.6f}, left time={:.6f}".format(epoch, num_epochs + start_epoch, t_loss,
                ava_epoch_time, left_time))

            if t_loss < best_loss:
                save_checkpoints_best(epoch)
                best_loss = t_loss
                best_epoch = epoch
            plt.figure(figsize=(4, 4))
            plt.plot(epoch_log, loss_log)
            plt.title("loss")
            plt.savefig(os.path.join(experiment_directory, 'loss.png'))

        lr_schedules.step()

    print("best_loss: ",best_loss," best_epoch:",best_epoch)

if __name__=="__main__":
    #python train_nodes.py -e chair_exp -g 0 --cate chair
    import argparse
    arg_parser = argparse.ArgumentParser(description="Train a Network")
    arg_parser.add_argument("--experiment", "-e",
        dest="experiment_directory", 
        required=True,
        help="The experiment directory. This directory should include "+
        "experiment specifications in 'specs,json' + \
        and logging will be "+"done in this directory as well.")
    arg_parser.add_argument("--gpu", 
        "-g", dest="gpu", 
        required=True, 
        help="gpu id")

    arg_parser.add_argument(
        "--cate", dest="category", 
        required=True, help="category")

    utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(int(args.gpu))

    print(f'experiment, {args.experiment_directory}, category: {args.category}, gpu:{int(args.gpu)}')
    
    main_function(args.experiment_directory,args.category)
