# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import nvidia_smi
import torch, data, iou
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter


def adjust_lr(optimizer, lr):
    lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("LR =", lr)
    return lr


# Options
use_cuda = torch.cuda.is_available()
print("use_cuda =", use_cuda)

if data.original_full_ds:
    train_dt = data.kits_dataset('./train')
elif data.tissue_kidney_tumor_small_ds:
    train_dt = data.kits_dataset('./kidney_train')

train_data_loader = torch.utils.data.DataLoader(train_dt, batch_size=data.batch_size,
                                                collate_fn=data.train_point_transform,
                                                shuffle=True)
if data.original_full_ds:
    val_dt = data.kits_dataset('./val')
elif data.tissue_kidney_tumor_small_ds:
    val_dt = data.kits_dataset('./kidney_val')

val_data_loader = torch.utils.data.DataLoader(val_dt, batch_size=data.batch_size,
                                              collate_fn=data.val_point_transform,
                                              shuffle=False)
val_size = len(val_dt)
train_size = len(train_dt)
print('training examples =', train_size)
print('validation examples =', val_size)
print('batch size =', data.batch_size)
print('classes =', data.CLASS_LABELS)
print('original_full_ds =', data.original_full_ds)
print('tissue_kidney_full_ds =', data.tissue_kidney_full_ds)
print('tissue_kidney_tumor_small_ds =', data.tissue_kidney_tumor_small_ds)
print('inference_mode =', data.inference_mode)

exp_name = './weights/weights_scale{}_m{}_rep{}_ResidualBlocks{}'. \
    format(data.scale, data.m, data.val_reps, data.residual_blocks)

now = datetime.now()
logdir = "logs/" + now.strftime("%Y%m%d-%H_%M_%S") + "/"
logfile_name = "logs/train_log_" + now.strftime("%Y%m%d-%H_%M_%S") + ".txt"
logfile = open(logfile_name, 'w')
writer = SummaryWriter(logdir=logdir, flush_secs=10)
print('tensorboard --logdir logs')

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # gpu0

unet = data.Model(data.dimension, data.full_scale, data.num_cl)
if use_cuda:
    unet = unet.cuda()

lr = 0.001
training_epochs = 10000
epoch_learning_rates = [5000, 7000]

optimizer = optim.Adam(unet.parameters(), lr=lr)
training_epoch = scn.checkpoint_restore(unet, exp_name, 'unet', use_cuda)
optimizer = scn.optimizer_restore(optimizer, exp_name, 'unet')
print("starting from epoch:", training_epoch)
print('classifier parameters', sum([x.nelement() for x in unet.parameters()]))

# tmp
adjust_lr(optimizer, 0.0001)

num_iterations = train_size // data.batch_size
iteration = (training_epoch - 1) * num_iterations + 1
for epoch in range(training_epoch, training_epochs + 1):
    num_iterations = epoch * int(train_size / data.batch_size)
    if epoch in epoch_learning_rates:
        lr = adjust_lr(optimizer, lr)

    unet.train()
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0
    epoch_start = time.time()
    train_loss = 0

    for i, batch in enumerate(train_data_loader):
        iter_start = time.time()
        optimizer.zero_grad()

        # save input obj

        # xyz = batch['x'][0].cpu().numpy()
        # rgb = batch['x'][1].cpu().numpy()
        # lbl = batch['y'].cpu().numpy()
        # inds = (xyz[:, 3] == 0)  # get 1st scene in the batch
        # xyz = xyz[inds]
        # xyz[:, 0] = (xyz[:, 0].astype(np.float32) / data.scale_z).astype(np.int)
        # rgb = rgb[inds]
        # lbl = lbl[inds]
        # data.save_to_obj(xyz, rgb, lbl, './epoch'+str(epoch)+'_batch'+str(i)+'.obj')

        batch_num_points = batch['y'].shape[0]
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
        predictions = unet(batch['x'])

        # loss = torch.nn.functional.cross_entropy(predictions, batch['y'])
        if data.tissue_kidney_full_ds:
            loss = torch.nn.functional.cross_entropy(predictions, batch['y'],
                                                     weight=torch.tensor([1., 1000.]).cuda())
        elif data.original_full_ds:
            loss = torch.nn.functional.cross_entropy(predictions, batch['y'],
                                                     weight=torch.tensor([1., 100., 10000.]).cuda())
        elif data.tissue_kidney_tumor_small_ds:
            loss = torch.nn.functional.cross_entropy(predictions, batch['y'],
                                                     weight=torch.tensor([1., 1., 100.]).cuda())

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        writer.add_scalar("train_loss", float(loss.item()), iteration)
        writer.add_scalar('nv/gpu_memory', res.used, iteration)

        # get optimizer's lr
        lrs = []
        for param_group in optimizer.param_groups:  # can there be several param_groups?
            lrs.append(param_group["lr"])
        # lrs = optimizer.param_groups[0]["lr"]
        s = "Iteration: {}/{}, train loss = {:.5f}, time = {:4.1f}, lr = {}, points in batch = {}". \
            format(iteration, num_iterations, loss.item(), time.time() - iter_start, lrs, batch_num_points)
        print(s)
        logfile.write(s + '\n')
        iteration += 1
        # del batch

    s = "EPOCH: {}, train loss = {:.5f}, time = {:4.1f}, lr = {}\n".format(epoch, train_loss / (i + 1),
                                                                           time.time() - epoch_start, lrs)
    print(s)
    logfile.write(s + '\n')

    scn.checkpoint_save(unet, optimizer, exp_name, 'unet', epoch, use_cuda=use_cuda, save_frequency=2)
    # validate
    if epoch % 2 == 0:
        with torch.no_grad():
            unet.eval()
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
            save = False
            print("\nEvaluation")
            start = time.time()
            for rep in range(1, 1 + data.val_reps):
                all_pred = np.array([], dtype=int)
                valLabels = np.array([], dtype=int)
                num_batches = len(val_data_loader)
                for i, batch in enumerate(val_data_loader):
                    print(">>>Processing batch: {}/{}".format(i + 1, num_batches))
                    valLabels = np.concatenate((valLabels, batch['y']))
                    if use_cuda:
                        batch['x'][1] = batch['x'][1].cuda()
                        batch['y'] = batch['y'].cuda()
                    predictions = unet(batch['x'])

                    # val loss
                    if data.tissue_kidney_full_ds:
                        loss = torch.nn.functional.cross_entropy(predictions, batch['y'],
                                                                 weight=torch.tensor([1., 1000.]).cuda())
                    elif data.original_full_ds:
                        loss = torch.nn.functional.cross_entropy(predictions, batch['y'],
                                                                 weight=torch.tensor(
                                                                     [1., 100., 10000.]).cuda())
                    elif data.tissue_kidney_tumor_small_ds:
                        loss = torch.nn.functional.cross_entropy(predictions, batch['y'],
                                                                 weight=torch.tensor([1., 1., 100.]).cuda())
                    writer.add_scalar("val_loss", float(loss.item()), iteration)
                    s = "Iteration: {}/{}, val loss = {:.5f}".format(iteration, num_iterations, loss.item())
                    print(s)
                    logfile.write(s + '\n')

                    predictions = predictions.cpu().numpy()
                    predictions = np.argmax(predictions, axis=1)
                    all_pred = np.concatenate((all_pred, predictions))

                    # save predicted obj
                    if save == True:
                        xyz = batch['x'][0].cpu().numpy()
                        rgb = batch['x'][1].cpu().numpy()
                        inds = (xyz[:, 3] == 0)  # get 1st scene in the batch
                        xyz = xyz[inds]
                        xyz[:, 0] = (xyz[:, 0].astype(np.float32) / data.scale_z).astype(np.int)
                        rgb = rgb[inds]
                        pred = predictions[inds]
                        data.save_to_obj(xyz, rgb, pred,
                                         './pred_epoch' + str(epoch) + '_batch' + str(i) + '.obj')

                s = 'EPOCH: {}, validation: time = {:.3f}, timestamp = {}'. \
                    format(epoch, time.time() - start, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print(s)
                class_ious = iou.evaluate(all_pred, valLabels)

                if data.num_cl == 2:
                    ious = str(class_ious['tissue'][0]) + ', ' + \
                           str(class_ious['kidney'][0])
                elif data.num_cl == 3:
                    ious = str(class_ious['tissue'][0]) + ', ' + \
                           str(class_ious['kidney'][0]) + ', ' + \
                           str(class_ious['tumor'][0])
                logfile.write(s + '\nclass ious: ' + ious + '\n')

                writer.add_scalar("iou/tissue", class_ious['tissue'][0], iteration)
                writer.add_scalar("iou/kidney", class_ious['kidney'][0], iteration)
                if data.num_cl == 3:
                    writer.add_scalar("iou/tumor", class_ious['tumor'][0], iteration)

logfile.close()
