import torchvision as tv
import torchvision as tv
import torch as t
from model import *
from config import *
from time import time
from utils import *
import torch.nn.functional as F
import threading
import itertools
import torch
from torch.utils.data import DataLoader
from datasets import ImageDataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

# t.backends.cudnn.benchmark = True
start = time()
opt = Config()


def train(**kwargs):
    device = opt.device

    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.Tensor
    input_A = Tensor(opt.batch_size, 3, opt.image_size, opt.image_size).to(device)
    input_B = Tensor(opt.batch_size, 3, opt.image_size, opt.image_size).to(device)

    # data
    # transforms = tv.transforms.Compose([
    #     # tv.transforms.Resize((opt.image_size, opt.image_size)),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transforms_ = [transforms.Resize(int(opt.image_size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.image_size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(ImageDataset('D:/SUN/archives/datasets/horse2zebra', transforms_=transforms_, unaligned=True),
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)

    # network
    netg1, netd1 = Generator(), Discriminator()
    netg2 = Generator()

    netg1.initialize_weights()
    netd1.initialize_weights()
    netg2.initialize_weights()

    netg1.to(device)
    netg2.to(device)
    netd1.to(device)

    optimizer_g1 = t.optim.Adam(netg1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_g2 = t.optim.Adam(netg2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d1 = t.optim.Adam(netd1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    lr_scheduler_G1 = torch.optim.lr_scheduler.LambdaLR(optimizer_g1, lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                         100).step)
    lr_scheduler_G2 = torch.optim.lr_scheduler.LambdaLR(optimizer_g2, lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                         100).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_d1, lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                        100).step)

    criterion_identity = t.nn.L1Loss().to(device)
    criterion_GAN = t.nn.MSELoss().to(device)

    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)

    print('used {}s to init.'.format(time() - start))

    global_i = 0
    visual = Visualize()
    epoch_start = 0
    time_init = time()

    # load model
    if not opt.debug and len(os.listdir(opt.checkpoint_path)) > 0:
        path = find_new_file(opt.checkpoint_path)
        checkpoint = t.load(path)
        netg1.load_state_dict(checkpoint['g1_state_dict'])
        netg2.load_state_dict(checkpoint['g2_state_dict'])
        netd1.load_state_dict(checkpoint['d1_state_dict'])
        optimizer_g1.load_state_dict(checkpoint['optimizer_g1_state_dict'])
        optimizer_g2.load_state_dict(checkpoint['optimizer_g2_state_dict'])
        optimizer_d1.load_state_dict(checkpoint['optimizer_d1_state_dict'])
        epoch_start = checkpoint['epoch']

    for epoch in range(epoch_start, opt.max_epoch):
        for ii, batch in enumerate(dataloader):
            # Set model input
            A = Variable(input_A.copy_(batch['A']))
            B = Variable(input_B.copy_(batch['B']))

            if (global_i % opt.g_every == 0):
                # train generator2
                optimizer_g2.zero_grad()

                z2, same_A = netg2(A)

                loss_identity2 = criterion_identity(A, same_A)

                loss_identity2.backward()
                optimizer_g2.step()

                # train generator1
                optimizer_g1.zero_grad()

                z1, fake_B = netg1(A)
                z3, same_B = netg1(fake_B)
                pred_fake = netd1(fake_B)

                # loss_identity = criterion_identity(fake_B, same_B)
                loss_identity_a_b = criterion_identity(A, fake_B)
                loss_G1 = 0.5 * torch.mean((pred_fake - true_labels)**2)
                # loss_G1 = criterion_GAN(pred_fake, true_labels)
                loss_Z = criterion_identity(z1, z2.detach())

                V_g = loss_G1 + loss_Z + loss_identity_a_b

                V_g.backward()
                optimizer_g1.step()

            if (global_i % opt.d_every == 0):
                # train discriminator
                optimizer_d1.zero_grad()

                # Real loss
                pred_real = netd1(B)
                loss_d_real = 0.5 * torch.mean((pred_real - true_labels)**2)
                # loss_d_real = criterion_GAN(pred_real, true_labels)
                loss_d_real.backward()

                # Fake loss
                _, fake_B = netg1(A)
                pred_fake = netd1(fake_B.detach())
                loss_d_fake = 0.5 * torch.mean((pred_fake - fake_labels)**2)
                # loss_d_fake = criterion_GAN(pred_fake, fake_labels)
                loss_d_fake.backward()

                V_d = loss_d_real + loss_d_fake
                # V_d.backward()

                optimizer_d1.step()

            global_i += 1
            print("===> Epoch[{}]({}/{}): V_d: {:.4f} V_g1: {:.4f} V_g2: {:.4f}".format(
                epoch, ii, len(dataloader), V_d, V_g, loss_identity2
            ))

            if global_i % opt.plot_every == 0:
                time_consumed = str(round((time() - time_init) / 3600, 2)) + 'h'
                visual.img(epoch, time_consumed, fake_B.detach(), A.detach(), None)
                visual.loss(epoch, {'V_g1': V_g.detach(), 'V_g2': loss_identity2, 'V_d': V_d.detach(),
                                    'loss_z': loss_Z.detach()})

        # Update learning rates
        lr_scheduler_G1.step()
        lr_scheduler_G2.step()
        lr_scheduler_D.step()

        if (epoch + 1) % opt.save_every == 0:
            # save model
            path = opt.checkpoint_path + 'epoch' + str(epoch) + '.tar'
            t.save({
                'epoch': epoch,
                'd1_state_dict': netd1.state_dict(),
                'g1_state_dict': netg1.state_dict(),
                'g2_state_dict': netg2.state_dict(),
                'optimizer_g1_state_dict': optimizer_g1.state_dict(),
                'optimizer_g2_state_dict': optimizer_g2.state_dict(),
                'optimizer_d1_state_dict': optimizer_d1.state_dict()
            }, path)


if __name__ == '__main__':
    train()
