import torch as t
from torch import nn, optim
from model import SlowFastNet
import os
from dataloader import make_loader
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = [int(i) for i in CUDA_VISIBLE_DEVICES.split(",")]


epoch = 40
batch_size = 4
init_lr = 0.001
lr_de_rate = 0.1
lr_de_epoch = 10
train_data_root_dir = r"/home/yuyang/data/ucf101"
valid_data_root_dir = r"/home/yuyang/data/ucf101"
num_workers = 4
clip_len = 16
slow_tao = 16
alpha = 8
weight_decay = 0.00001
short_side_size_range = [256, 320]
crop_size = [224, 224]
print_step = 2
class_names = ["ApplyEyeMakeup", "ApplyLipstick", "Archery"]
best_valid_loss = float("inf")
num_classes = len(class_names)
softmax_op = nn.Softmax(dim=1)


def calc_accu(model_output, target):
    accu = (t.argmax(softmax_op(model_output), dim=1) == target).sum().item() / model_output.size()[0]
    return accu


def train_epoch(current_epoch, model, criterion, optimizer, train_loader):
    model.train()
    step = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(device_ids[0])
        l_train_cuda = l_train.cuda(device_ids[0])
        train_output = model(d_train_cuda)
        train_loss = criterion(train_output, l_train_cuda)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_accu = calc_accu(train_output, l_train_cuda)
        if current_step % print_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, train_accu:%.5f" % (current_epoch, epoch, current_step, step, train_loss.item(), train_accu))
        current_step += 1
    print("saving epoch model......")
    t.save(model.state_dict(), "epoch.pth")
    return model


def valid_epoch(current_epoch, criterion, valid_loader, model):
    global best_valid_loss
    model.eval()
    step = len(valid_loader)
    accum_loss = 0
    accum_accu = 0
    for d_valid, l_valid in valid_loader:
        d_valid_cuda = d_valid.cuda(device_ids[0])
        l_valid_cuda = l_valid.cuda(device_ids[0])
        with t.no_grad():
            valid_output = model(d_valid_cuda)
            valid_loss = criterion(valid_output, l_valid_cuda)
            valid_accu = calc_accu(valid_output, l_valid_cuda)
            accum_loss += valid_loss.item()
            accum_accu += valid_accu
    avg_loss = accum_loss / step
    avg_accu = accum_accu / step
    print("##########valid epoch:%d############" % (current_epoch,))
    print("valid_loss:%.5f, valid_accu:%.5f" % (avg_loss, avg_accu))
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        print("saving best model......")
        t.save(model.state_dict(), "best.pth")
    return model


def main():
    model = SlowFastNet(num_classes=num_classes, slow_tao=slow_tao, alpha=alpha)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    criterion = nn.CrossEntropyLoss().cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.StepLR(optimizer, lr_de_epoch, lr_de_rate)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(crop_size, short_side_size_range, clip_len, train_data_root_dir, True, class_names, batch_size, num_workers)
        valid_loader = make_loader(crop_size, short_side_size_range, clip_len, valid_data_root_dir, False, class_names, batch_size, num_workers)
        model = train_epoch(current_epoch, model, criterion, optimizer, train_loader)
        model = valid_epoch(current_epoch, criterion, valid_loader, model)
        lr_sch.step()


if __name__ == "__main__":
    main()