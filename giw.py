import os
import os.path
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloader import ColorMNIST
from model import Net
from kmm import kmm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--step', type=float, default=100, help='period of learning rate decay')
parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
parser.add_argument('--wd', type=float, default=0.002, help='weight decay')
parser.add_argument('--bs', type=int, default=256, help='batch size for training data')
parser.add_argument('--num_epoch', type=int, default=400, help='total number of training epoch')
parser.add_argument('--seed', type=int, default=99, help='random seed')

args = parser.parse_args()


def set_seed(seed=args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model():
    net = Net()

    if torch.cuda.is_available():
        net.cuda()

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)

    return net, opt, scheduler


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def main():
    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])

    train_dataset = ColorMNIST(root='./data/',
                               download=True,
                               train=True,
                               val=False,
                               transform=transform
                               )

    test_dataset = ColorMNIST(root='./data/',
                              download=True,
                              train=False,
                              transform=transform
                              )

    val_dataset = ColorMNIST(root='./data/',
                             download=True,
                             train=True,
                             val=True,
                             transform=transform
                             )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               num_workers=0,
                                               drop_last=False,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs,
                                              num_workers=0,
                                              drop_last=False,
                                              shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.bs,
                                             num_workers=0,
                                             drop_last=False,
                                             shuffle=True)

    # define the model, optimizer, and lr decay scheduler
    net, opt, scheduler = build_model()

    print("pre-training starts")

    # pre-train the model
    for epoch in tqdm(range(10)):
        test_acc_tmp = []

        for i, (image, labels, _) in enumerate(train_loader):
            net.train()
            image, labels = to_cuda(image), to_cuda(labels)
            out_train = net(image)
            l_tr = F.cross_entropy(out_train, labels.squeeze())

            opt.zero_grad()
            l_tr.backward()
            opt.step()

        net.eval()
        # test acc
        for itr, (test_img, test_label, __) in enumerate(test_loader):
            test_img, test_label = to_cuda(test_img), to_cuda(test_label)
            test_correct = 0
            test_total = 0
            out_test = net(test_img)
            _, predicted = torch.max(out_test.data, 1)
            test_total += test_label.size(0)
            test_correct += (predicted == test_label.squeeze()).sum().item()
            test_accuracy = test_correct / test_total

            test_acc_tmp.append(test_accuracy)

        test_accuracy_mean = np.mean(test_acc_tmp)
        print("test accuracy mean is", test_accuracy_mean)

    # retrieve transformed features & estimate alpha
    net.fc1.register_forward_hook(get_activation('fc1'))
    fe_tr, fe_val, index_val = get_feature(net, train_loader, val_loader)
    print("training osvm starts")
    val_dic, alpha = val_split(fe_tr, fe_val, index_val)
    print("alpha is estimated as", alpha)

    # train the model
    test_acc = []

    for epoch in tqdm(range(args.num_epoch)):
        train_acc_tmp = []
        test_acc_tmp = []

        for i, (image, labels, _) in enumerate(train_loader):
            # weight estimation (we) step
            net.cuda()
            net.eval()
            image, labels = to_cuda(image), to_cuda(labels)

            out_train = net(image)
            l_tr = F.cross_entropy(out_train, labels.squeeze(), reduction='none').reshape(-1, 1)

            val_image, val_labels, val__ = next(iter(val_loader))
            val_image, val_labels = to_cuda(val_image), to_cuda(val_labels)

            split_labels = [bool(val_dic[i.item()]) for i in val__]

            val1_image, val1_labels = val_image[split_labels], val_labels[split_labels]
            val2_image, val2_labels = val_image[np.invert(split_labels)], val_labels[np.invert(split_labels)]

            out_val1 = net(val1_image)

            l_val1 = F.cross_entropy(out_val1, val1_labels.squeeze(), reduction='none').reshape(-1, 1)

            n_batch = len(_)
            dist = torch.cdist(l_tr, l_tr)[torch.tril_indices(n_batch, n_batch, offset=-1).unbind()]
            kernel_width = torch.quantile(dist, q=0.5).item()

            l_tr_cpu, l_val_cpu = np.array(l_tr.detach().cpu()), np.array(l_val1.detach().cpu())
            coef = kmm(l_tr_cpu, l_val_cpu, kernel_width)

            w = torch.from_numpy(np.asarray(coef)).float().cuda()
            w = (w / w.sum()) * n_batch

            # weighted classification (wc) step
            net.train()
            out_train_wc = net(image)
            l_tr_wc = F.cross_entropy(out_train_wc, labels.squeeze(), reduction='none')
            l_tr_wc_weighted = torch.mean(l_tr_wc * w)

            out_val2 = net(val2_image)
            l_val2 = F.cross_entropy(out_val2, val2_labels.squeeze())

            l_total = alpha * l_tr_wc_weighted + (1 - alpha) * l_val2

            opt.zero_grad()
            l_total.backward()
            opt.step()

            # train acc
            train_correct = 0
            train_total = 0
            _, predicted = torch.max(out_train_wc.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.squeeze()).sum().item()
            train_accuracy = train_correct / train_total
            train_acc_tmp.append(train_accuracy)

        train_accuracy_mean = np.mean(train_acc_tmp)
        print("train accuracy mean is", train_accuracy_mean)

        net.eval()
        # test acc
        for itr, (test_img, test_label, __) in enumerate(test_loader):
            test_img, test_label = to_cuda(test_img), to_cuda(test_label)
            test_correct = 0
            test_total = 0
            out_test = net(test_img)
            _, predicted = torch.max(out_test.data, 1)
            test_total += test_label.size(0)
            test_correct += (predicted == test_label.squeeze()).sum().item()
            test_accuracy = test_correct / test_total

            test_acc_tmp.append(test_accuracy)

        test_accuracy_mean = np.mean(test_acc_tmp)
        print("test accuracy mean is", test_accuracy_mean)
        test_acc.append(test_accuracy_mean)
        test_acc_arr = np.array(test_acc)

        scheduler.step()

    # save the output
    np.savetxt('./output/test_acc.txt', test_acc_arr, fmt='%s')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_acc)
    fig.savefig('./output/test_acc.png')


if __name__ == '__main__':
    main()
