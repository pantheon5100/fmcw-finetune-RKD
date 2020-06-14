import time

import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torchvision

from model.model_shallow import Shollow
from utlis.some_utils import create_root, plot_confusion_matrix, weight_init

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 20
N_EPOCH = 100

LEARNING_RATE = 1e-3
DECAY = LEARNING_RATE / N_EPOCH


def main():
    Data = np.load("TimeFreq9D.npz")
    Label = np.load("TimeFreq9L.npz")
    Train_data = dict()

    for name in Data.files:
        Train_data[name] = [Data[name], Label[name]]

    comment = "shallow_one"
    time_stamp = time.strftime('ex-%m%d%H%M', time.localtime(time.time()))
    writer = SummaryWriter(logdir='runs/{}-ft-{}_resnet18'.format(time_stamp, comment))
    writer.add_text('global_config', 'BATCH_SIZE: {}, N_EPOCH: {}, '
                                     'LEARNING_RATE: {}, DECAY: {}'.format(BATCH_SIZE, N_EPOCH, LEARNING_RATE, DECAY))
    val_cm = open(r"./confusion_matrix/ft-{}-val_cm.txt".format(comment),"a+",encoding="utf-8")
    val_cm.write("\n{}".format(time_stamp))
    test_cm = open(r"./confusion_matrix/ft-{}-test_cm.txt".format(comment),"a+",encoding="utf-8")
    test_cm.write("\n{}".format(time_stamp))

    # release gpu memory
    torch.cuda.empty_cache()
    # initiate model
    # model = ResFT().to(DEVICE).double()
    # model = torchvision.models.resnet18(pretrained=False, num_classes=6).cuda().double()
    model = Shollow().to(DEVICE).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_save_dir = create_root(r"I:\Zak_work\State_of_art\new200501\model_save\ft-{}-state".format(comment)+time_stamp)

    # set recorder and learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Writing training information to writer
    writer.add_text('state', 'model save dir: {}'.format(model_save_dir))
    writer.add_text('Data_info', 'Dataset: {}'.format([(key, Train_data[key][0].shape) for key in Train_data.keys()]))

    # scheduler_state_dict = scheduler.state_dict()
    # writer.add_text('/config/lr', 'learning rate scheduler: {}'.format(scheduler_state_dict))
    # print(scheduler_state_dict)

    # Prepare data
    zgy_data = Train_data['zgy'][0]
    zgy_label = Train_data['zgy'][1]
    index =  int(200*0.8)
    train_label = []
    train_data = []
    validation_label = []
    validation_data = []
    for i in range(6):
        train_data.append(zgy_data[200*i:200*i+index])
        train_label.append(zgy_label[200*i:200*i+index])
        validation_data.append(zgy_data[200*i+index:200*(i+1)])
        validation_label.append(zgy_label[200 * i + index:200 * (i + 1)])
    train_data = np.vstack(train_data)
    train_label = np.vstack(train_label)
    validation_data = np.vstack(validation_data)
    validation_label = np.vstack(validation_label)

    # Shuffle data
    permutation = np.random.permutation(train_data.shape[0])
    train_data = train_data[permutation, :, :, :]
    train_label = train_label[permutation, :]


    for epoch in range(1, N_EPOCH + 1):
        # for every epoch calculate mean loss and acc
        train_loss = []
        train_acc = []

        for times in range(int(train_data.shape[0] / BATCH_SIZE)):
        # for times, data_src_iter in enumerate(data_loader.get(dataset_config['train__'])[0]):
            loss_class = nn.CrossEntropyLoss()
            # It's better for each batch to set model.train()
            model.train()
            # Training model using source data
            s_img, s_label = train_data[BATCH_SIZE * times:BATCH_SIZE * (times + 1)], train_label[
                                                                                  BATCH_SIZE * times:BATCH_SIZE * (
                                                                                              times + 1)]
            s_img, s_label = torch.tensor(s_img).to(DEVICE).double(), torch.tensor(s_label).to(DEVICE).double()

            class_output = model(s_img)
            err = loss_class(class_output, torch.max(s_label, 1)[1])
            train_loss.append(err.item())

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            # Calculate acc
            pred = torch.max(class_output.data, 1)
            n_correct = (pred[1] == torch.max(s_label, 1)[1]).sum().item()
            batch_acc = (n_correct / BATCH_SIZE) * 100
            train_acc.append(batch_acc)

        # loss and training acc scalars
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        print('Train Epoch: [{}/{}]\tLoss: {:.6f}\tBatch Acc: {}'.format(
            epoch, N_EPOCH, train_loss, train_acc))

        print('Val_data test.')
        acc_src, acc_m, f1score = test(model, [validation_data, validation_label], epoch)
        val_cm.write("\n~{}\n{}".format(epoch, acc_m))
        # Validation scalar
        writer.add_scalar('train/val_acc', acc_src, global_step=epoch)
        writer.add_text('train/val_f1score', "{}".format(f1score), epoch)

        if epoch%1==0:
            acc_src1, acc_m1, f1score = test(model, Train_data, epoch, names=["zk", "lt", "dl", "jc", "lg"])
            acc_src2, acc_m2, f1score = test(model, Train_data, epoch, names=["gp", "zwh", "fl"])
            test_cm.write("\n~{}\n#{}\n#{}".format(epoch, acc_m1, acc_m2))
            writer.add_scalars('train/test_acc', {"5p":acc_src1, "3p":acc_src2}, epoch)

            torch.save(model.state_dict(), model_save_dir + '\\aa_epoch_{}-{}'.format(epoch, acc_src2))

    writer.close()
    val_cm.close()
    test_cm.close()


def test(model, dataloader, epoch, names=None):
    """

    """
    model.eval()
    pred_label = []
    true_label = []
    if names:
        n_correct = 0
        total_data = 0
        predict_matrix = np.zeros([6,6])

        for name in names:
            t_imgs = torch.tensor(dataloader[name][0]).double().cuda()
            t_lables = torch.tensor(dataloader[name][1]).double().cuda()

            with torch.no_grad():
                for times in range(int(t_imgs.shape[0] / BATCH_SIZE)):
                    t_img, t_lable = t_imgs[BATCH_SIZE * times:BATCH_SIZE * (times + 1)], t_lables[
                                                                                              BATCH_SIZE * times:BATCH_SIZE * (
                                                                                                      times + 1)]
                    class_output = model(t_img)
                    pred = torch.max(class_output.data, 1)
                    label = torch.max(t_lable, 1)
                    n_correct += (pred[1] == label[1]).sum().item()
                    predict_matrix += confusion_matrix(pred[1].cpu().detach(), label[1].cpu().detach(), labels=[0,1,2,3,4,5])
                    true_label.append(label[1].cpu().detach().numpy())
                    pred_label.append(pred[1].cpu().detach().numpy())

            total_data += len(t_imgs)

    else:
        total_data = len(dataloader[0])
        with torch.no_grad():
            # if type(dataloader) == 'list':
            t_img = torch.tensor(dataloader[0]).double().cuda()
            t_lable = torch.tensor(dataloader[1]).double().cuda()
            class_output = model(t_img)
            pred = torch.max(class_output.data, 1)
            label = torch.max(t_lable, 1)
            n_correct = (pred[1] == label[1]).sum().item()
            predict_matrix = confusion_matrix(pred[1].cpu(), label[1].cpu(), labels=[0,1,2,3,4,5])
            true_label.append(label[1].cpu().detach().numpy())
            pred_label.append(pred[1].cpu().detach().numpy())

    true_label = np.hstack(true_label)
    pred_label = np.hstack(pred_label)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(true_label, pred_label)

    accu = float(n_correct) / total_data * 100
    print('Accuracy dataset: {:.4f}% F1-score: {}'.format(accu, np.mean(f_score)))
    return accu, predict_matrix, [precision, recall, f_score, true_sum]

if __name__ == '__main__':
    main()
