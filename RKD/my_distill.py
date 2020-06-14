import os

import torch
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from metric.utils import recall
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer

from data_load import load_data
from model_o import ResFT
from stu_model import ResNet
from model.model_dense import Dense
from model.model_mobile import Mobile
from parser_argparse import get_parser


def main():
    parser = get_parser()
    arg_distill = ['--base', 'googlenet',
                   '--embedding_size', '6',
                   '--teacher_base', 'resnet18',
                   '--teacher_embedding_size', '6',
                   '--teacher_load', 'teacher/best.pth',
                   '--dist_ratio', '1',
                   '--angle_ratio', '2',
                   '--triplet_ratio', '1',
                   '--save_dir', 'student']

    opts = parser.parse_args(arg_distill)

    def get_normalize():
        google_mean = torch.Tensor([104, 117, 128]).view(1, -1, 1, 1).cuda()
        google_std = torch.Tensor([1, 1, 1]).view(1, -1, 1, 1).cuda()

        zero_mean = torch.Tensor([0, 0, 0]).view(1, -1, 1, 1).cuda()
        zero_std = torch.Tensor([1, 1, 1]).view(1, -1, 1, 1).cuda()

        def othernorm(x):
            # x = (x - zero_mean) / zero_std
            x = x - zero_mean

            return x

        return othernorm

    teacher_normalize = get_normalize()

    dataset_train, dataset_train_eval, dataset_eval = load_data()

    print("Number of images in Training Set: %d" % len(dataset_train))
    print("Number of images in Test set: %d" % len(dataset_eval))

    loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5,
                                                                         iter_per_epoch=opts.iter_per_epoch),
                                     pin_memory=True, num_workers=8)
    loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                                   pin_memory=False, num_workers=8)
    loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                             pin_memory=True, num_workers=8)

    student = ResNet()
    # teacher = ResFT()
    teacher = Dense()
    teacher.load_state_dict(torch.load('./torch_save/dense_net100.pth'))
    student = student.cuda()
    teacher = teacher.cuda()

    optimizer = optim.Adam(student.parameters(), lr=opts.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
    triplet_criterion = L2Triplet(sampler=opts.triplet_sample(), margin=opts.triplet_margin)
    at_criterion = AttentionTransfer()

    # logger
    writer = SummaryWriter(comment='First')


    def train(loader, ep):
        student.train()
        teacher.eval()

        dist_loss_all = []
        angle_loss_all = []
        dark_loss_all = []
        triplet_loss_all = []
        at_loss_all = []
        loss_all = []

        train_iter = tqdm(loader, ascii=True)
        for images, labels in train_iter:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                # t_b1, t_b2, t_b3, t_b4, t_pool, t_e = teacher(teacher_normalize(images), True)
                t_e = teacher(teacher_normalize(images))



            # b1, b2, b3, b4, pool, e = student(teacher_normalize(images), True)
            e = student(teacher_normalize(images), True)

            # at_loss = opts.at_ratio * (at_criterion(b2, t_b2) + at_criterion(b3, t_b3) + at_criterion(b4, t_b4))

            triplet_loss = opts.triplet_ratio * triplet_criterion(e, labels)
            dist_loss = opts.dist_ratio * dist_criterion(e, t_e)
            angle_loss = opts.angle_ratio * angle_criterion(e, t_e)
            dark_loss = opts.dark_ratio * dark_criterion(e, t_e)

            loss = triplet_loss + dist_loss + angle_loss + dark_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            triplet_loss_all.append(triplet_loss.item())
            dist_loss_all.append(dist_loss.item())
            angle_loss_all.append(angle_loss.item())
            dark_loss_all.append(dark_loss.item())
            # at_loss_all.append(at_loss.item())
            loss_all.append(loss.item())

            train_iter.set_description("[Train][Epoch %d] Triplet: %.5f, Dist: %.5f, Angle: %.5f, Dark: %5f," %
                                       (ep, triplet_loss.item(), dist_loss.item(), angle_loss.item(), dark_loss.item()))
        print('[Epoch %d] Loss: %.5f, Triplet: %.5f, Dist: %.5f, Angle: %.5f, Dark: %.5f \n' %\
              (ep, torch.Tensor(loss_all).mean(), torch.Tensor(triplet_loss_all).mean(),
               torch.Tensor(dist_loss_all).mean(), torch.Tensor(angle_loss_all).mean(), torch.Tensor(dark_loss_all).mean(),))
        writer.add_scalar('data/loss_all', torch.Tensor(loss_all).mean(), ep)
        writer.add_scalar('data/triplet_loss_all', torch.Tensor(triplet_loss_all).mean(), ep)
        writer.add_scalar('data/dist_loss_all', torch.Tensor(dist_loss_all).mean(), ep)
        writer.add_scalar('data/angle_loss_all', torch.Tensor(angle_loss_all).mean(), ep)
        writer.add_scalar('data/dark_loss_all', torch.Tensor(dark_loss_all).mean(), ep)
        # writer.add_scalar('data/at_loss_all', torch.Tensor(at_loss_all).mean(), ep)



    def eval(net, normalize, loader, ep, cm=False):
        K = [1]
        net.eval()
        test_iter = tqdm(loader, ascii=True)
        embeddings_all, labels_all = [], []

        with torch.no_grad():
            for images, labels in test_iter:
                images, labels = images.cuda(), labels.cuda()
                output = net(normalize(images))
                embeddings_all.append(output.data)
                labels_all.append(labels.data)
                test_iter.set_description("[Eval][Epoch %d]" % ep)

            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            rec = recall(embeddings_all, labels_all, K=K)

            for k, r in zip(K, rec):
                print('[Epoch %d] Recall@%d: [%.4f]\n' % (ep, k, 100 * r))
        if cm:
            f, ax = plt.subplots()
            C_matrix = confusion_matrix(labels_all.numpy(), torch.argmax(embeddings_all, 1).numpy())
            # C_matrix = np.int16((C_matrix/C_matrix.max()*100))
            sns.heatmap(C_matrix, annot=True, ax=ax, fmt='d')
            ax.set_xlabel('Predict')
            ax.set_ylabel('True')
            ax.set_title('Correct / Total : {} / {}'.format(np.sum(np.diag(C_matrix)), len(labels_all)))
            plt.savefig('eval%d.jpg'%ep)

            return rec[0], f
        return rec[0]


    eval(teacher, teacher_normalize, loader_train_eval, 0, cm=True)
    eval(teacher,teacher_normalize,  loader_eval, 0)
    best_train_rec = eval(student, teacher_normalize, loader_train_eval, 0)
    best_val_rec = eval(student, teacher_normalize, loader_eval, 0)

    for epoch in range(1, opts.epochs+1):
        train(loader_train_sample, epoch)
        lr_scheduler.step()
        train_recall, f1 = eval(student,teacher_normalize, loader_train_eval, epoch, cm=True)
        val_recall, f2 = eval(student,teacher_normalize, loader_eval, epoch, cm=True)
        writer.add_figure('train_eval', f1, epoch)
        writer.add_figure('eval', f2, epoch)

        writer.add_scalar('data/train_recall', train_recall, epoch)
        writer.add_scalar('data/val_recall', val_recall, epoch)

        if best_train_rec < train_recall:
            best_train_rec = train_recall

        if best_val_rec < val_recall:
            best_val_rec = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write('Best Train Recall@1: %.4f\n' % (best_train_rec * 100))
                f.write("Best Test Recall@1: %.4f\n" % (best_val_rec * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))

        print("Best Train Recall: %.4f" % best_train_rec)
        print("Best Eval Recall: %.4f" % best_val_rec)


if __name__ == '__main__':
    main()
