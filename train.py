import os
import config
from time import time
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.my_dataset import CTDataLoader
from net.ResUnet import ResUNet, init, StageNet
from common.Visualizer import Visualizer
from common.Metric import Metric
from tqdm import tqdm
import argparse
from loss.Dice import DiceLoss_Focal, DiceLoss

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='VNet', help='model name [default: VNet]')
    parser.add_argument('--tqdm', type=bool, default=False, help='model name [default: True]')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=500, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--pretrain', type=str, default='/data/qinwang/single_yk/epoch_74.pth')#'/data/qinwang/multistage_true_no_down/epoch_80/net_epoch_80.pth',help='Pretrain model path')
    parser.add_argument('--save_dir', type=str, default='/data/qinwang/single_yk_1/',help='Path for saving model')

    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--augment', type=bool,  default=True, help='Whether use data augmentation [default: False]')
    parser.add_argument('--step_size', type=int,  default=50, help='Decay step for lr decay [default: every 200 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.3]')
    parser.add_argument('--viz_name', type=str, default='kkyykk', help='Name for ploting [default: kkyykk]')
    parser.add_argument('--num_workers', type=int, default=4, help='Num of the workers [default: 4]')
    parser.add_argument('--slice_number', type=int, default=32, help='Amount of the slice in z axis [default: 32]')
    return parser.parse_args()

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_for_epoch(epoch, net, viz, metricer, opt):
    if args.tqdm:
        iteration_choice = tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9)
    else:
        iteration_choice = enumerate(train_loader, 0)
    mean_loss = []
    dices = []
    for step, (image, label, glob_cube, start_slice, end_slice, label_glob) in iteration_choice:
        opt.zero_grad()
        image = image.cuda() #[b,1,48,384,240]
        label = label.long().cuda() #[b,48,384,240]
        glob_cube = glob_cube.cuda()
        label_glob = label_glob.long().cuda()
        '''For StageNet'''
        # predictions_stage1, predictions_stage2 = net(image, glob_cube, start_slice, end_slice)  # [b,3,48,384,240]
        # loss = loss_func(predictions_stage1, label_glob)  + loss_func(predictions_stage2, label)#, weight=weight2)
        '''For StageNet'''

        '''For VNet'''
        predictions_stage1 = net(image)  # [b,3,48,384,240]
        loss = loss_func(predictions_stage1, label)#, weight=weight2)
        '''For VNet'''
        mean_loss.append(loss.item())
        loss.backward()
        #if step %4 == 0:#for batch_size = 1 if not, remove it, probably no help
        opt.step()
        # predictions_arg = predictions_stage2.argmax(dim=1).float()  # [b,48,384,240]
        predictions_arg = predictions_stage1.argmax(dim=1).float()  # [b,48,384,240]
        metricer.set_input(label, predictions_arg)
        dices.append(metricer.dice_for_batch())
        if step % 4 is 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time() - start) / 60))

    train_epoch_loss = sum(mean_loss) / len(mean_loss)
    train_epoch_dice = np.concatenate(dices,axis=0).mean(axis=0)
    print('epoch:', epoch, 'train_loss:', train_epoch_loss, 'train_dice: ', train_epoch_dice)
    print('----------------tran end----------------------------')
    # plot train loss
    viz.plot('epoch_loss', train_epoch_loss, x_indice=epoch, name='train')
    viz.plot('epoch_dice_kidney', train_epoch_dice[0], x_indice=epoch, name='train')
    viz.plot('epoch_dice_tumor', train_epoch_dice[1], x_indice=epoch, name='train')

def validatation_for_epoch(epoch, net, viz, metricer):
    val_dices = []
    slice_number = 32  # slice numbers
    interval = 16

    # validation
    with torch.no_grad(): #
        for step, (_, _, cube_glob, _, _, label_glob) in enumerate(val_loader):
            res_tensor = torch.zeros((2, 3, cube_glob.shape[2], 384,
                                      240))  # ,ct_array.shape[-1],ct_array.shape[0],ct_array.shape[1]))
            cube_glob = cube_glob.cuda()
            label_glob = label_glob.cuda()


            cube_glob_tensor = cube_glob
            start_slice = 0
            end_slice = start_slice + slice_number - 1  # 31
            ct_array_list = []

            while end_slice <= cube_glob_tensor.shape[2] - 1:
                ct_array_list.append(cube_glob_tensor[:, :, start_slice:end_slice + 1, :, :])
                start_slice = start_slice + interval  # 4
                end_slice = start_slice + slice_number - 1  # 47

            # 当无法整除的时候反向取最后一个block
            if end_slice is not cube_glob_tensor.shape[2] - 1 + slice_number:
                ct_array_list.append(cube_glob_tensor[:, :, -slice_number:, :, :])
            # print('outputs_list',outputs_list)

            # 进行拼接
            start_slice1 = 0
            end_slice1 = start_slice1 + slice_number - 1  # 31
            ind_1 = 0
            while end_slice1 <= cube_glob_tensor.shape[2] - 1:
                ct_tensor = ct_array_list[ind_1]
                '''StageNet'''
                # outputs1, outputs2 = net(ct_tensor, cube_glob_tensor, start_slice1, end_slice1)  # after softmax
                # res_map1 = F.softmax(outputs1, dim=1).cpu()
                # res_map2 = F.softmax(outputs2, dim=1).cpu()
                '''VNet'''
                outputs1= net(ct_tensor)  # after softmax
                res_map2 = F.softmax(outputs1, dim=1).cpu()

                res_tensor[:, :, start_slice1:end_slice1 + 1, :, :] = res_tensor[:, :, start_slice1:end_slice1 + 1,:,:] + res_map2  # res_map1[:,:,start_slice1:end_slice1 + 1,:,:]
                start_slice1 = start_slice1 + interval  #
                end_slice1 = start_slice1 + slice_number - 1
                ind_1 = ind_1 + 1
                # print(start_slice1,end_slice1,ind)

            # 当无法整除的时候反向取最后一个block
            if end_slice1 is not res_tensor.shape[2] - 1 + slice_number:
                ct_tensor = ct_array_list[-1]
                '''StageNet'''
                # outputs1, outputs2 = net(ct_tensor, cube_glob_tensor, res_tensor.shape[2] - slice_number,
                #                          res_tensor.shape[2])  # after softmax
                # res_map1 = F.softmax(outputs1, dim=1).cpu()
                # res_map2 = F.softmax(outputs2, dim=1).cpu()
                '''VNet'''
                outputs1= net(ct_tensor)  # after softmax
                res_map2 = F.softmax(outputs1, dim=1).cpu()

                res_tensor[:, :, -slice_number:, :, :] = res_map2 + res_tensor[:, :, -slice_number:,:,:]  # res_map1[:,-slice_number:,:,:]


            res_map = np.argmax(res_tensor, axis=1).float()# otuput [bs, ori_z,ori_x,ori_y]
            # print('res map:', res_map.shape)
            # print(label_glob.shape)



            metricer.set_input(label_glob, res_map)
            val_dices.append(metricer.dice_for_batch())
    #print('val_dices',val_dices)
    val_epoch_dice = np.concatenate(val_dices,axis=0).mean(axis=0)
    print('----------------val end---------------------------')
    # plot train loss
    viz.plot('epoch_dice_kidney', val_epoch_dice[0], x_indice=epoch, name='val')
    viz.plot('epoch_dice_tumor', val_epoch_dice[1], x_indice=epoch, name='val')
    print('dice kidney',val_epoch_dice[0])
    print('dice tumor',val_epoch_dice[1])
    print('-------------------val end-------------------------')
    return val_epoch_dice

if __name__ == '__main__':
    args = parse_args()
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = '/data/weihao/pre-KiTS-3mm/'
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    NAME = args.viz_name # the name of visdom. should firstly run 'python -m visdom.server' in Terminal
    cudnn.benchmark = True # the used GPU
    leaing_rate = args.learning_rate
    batch_size = args.batchsize
    num_workers = args.num_workers
    start = time()

    '''LOGGING
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt'%(log_dir,args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    logger.info(args)'''

    '''LOADING MODEL'''
    if args.model == 'StageNet':
        net = StageNet(training=True)
    elif args.model == 'VNet':
        net = ResUNet(training = True, inchannel=1, stage = 1)
        #net2 = ResUNet(training = True, inchannel=1, stage = 1)
    net = torch.nn.DataParallel(net).cuda()
    if args.pretrain is not None:
        print('Use pretrain model %s...'%args.pretrain)
        #logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        kidney_dice = checkpoint['kidney_dice']
        tumor_dice = checkpoint['tumor_dice']
        print('Performance in training')
        print('\r Kidney %s: %f' % 'Instance Accuracy', kidney_dice)
        print('\r Tumor %s: %f' % 'Instance Accuracy', tumor_dice)
        net.load_state_dict(checkpoint)#(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        net = net.apply(weights_init)

    '''OPTIMIZER'''
    if args.optimizer == 'SGD':
        opt = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        opt = torch.optim.Adam(
            net.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    '''EVALUATION DEFINITION'''
    viz = Visualizer(env=NAME)
    metricer = Metric()
    '''DATA LOADER'''
    data_train = CTDataLoader(mode='train', slice_number = args.slice_number, scale=False, rotate=False, flip=False, glob_flag=True,
                              use_weight=True)  # image : [batch, 1, 48, 384, 240]) label: [batch, 48, 384, 240]
    train_loader = DataLoader(data_train, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    data_val = CTDataLoader(mode='val', glob_flag=True)
    val_loader = DataLoader(data_val, batch_size=2, shuffle=False, num_workers=num_workers, drop_last=True)

    '''LOSS FUNCTION'''
    #loss_func = nn.CrossEntropyLoss(weight=torch.Tensor(data_train.labelweights).cuda())
    loss_func = DiceLoss()
    current_dice = 0.5
    start_epoch = 0
    # training process
    for epoch in range(start_epoch, args.epoch):
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        if epoch%args.step_size==0:
            print('Learning rate:%f' % lr)
            #logger.info('Learning rate:%f' ,lr)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01

        train_for_epoch(epoch, net,viz, metricer, opt)
        net.eval()
        val_dice = validatation_for_epoch(epoch, net, viz, metricer)
        net.train()
        # 每10个epoch evaluate 一次 model
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        if val_dice[1] > current_dice: # Focus on the dice of tumor
            current_dice = val_dice[1]
            kidney_dice = val_dice[0]
            tumor_dice = current_dice
            state = {
                'epoch': epoch,
                'kidney_dice': kidney_dice,
                'tumor_dice': tumor_dice,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(net.state_dict(), args.save_dir + 'epoch_{}.pth'.format(epoch))

        
