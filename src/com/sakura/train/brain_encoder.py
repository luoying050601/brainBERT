import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch.autograd import Variable
import numpy as np
import pickle
import os
import time

start = time.perf_counter()

subjects = ['derivatives']
participants = ['18']
file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}
# 超参数定义
Epoch = 0
Batch_size = 16
LR = 0.005
Downloads_MNIST = False
train_loss = []
Epoch_line = list(range(Epoch))


# N_Test_img = 5


# train_data = torchvision.datasets.MNIST(
#     root=r'./mnist_data/',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=Downloads_MNIST
# )

class AutoEncoder(nn.Module):
    def __init__(self, length, brain_x, brain_y, brain_z):
        super(AutoEncoder, self).__init__()

        # data_size = length * brain_x * brain_y * brain_z
        # define encoder
        brain_size = brain_x * brain_y * brain_z
        self.encoder = nn.Sequential(
            nn.Linear(brain_size, 714),
            nn.Tanh(),
            # nn.Linear(68 * 95 * 79, 372),
            nn.Linear(714, 357),
            nn.Tanh(),
            # nn.Linear(372, 128),
            nn.Linear(357, 178),
            nn.Tanh(),
            # nn.Linear(128, 64),
            nn.Linear(178, 89),
            nn.Tanh(),
            # nn.Linear(64, 32)
            nn.Linear(89, 45)
            # turn into 32 features
        )
        # define decoder
        self.decoder = nn.Sequential(
            # nn.Linear(32, 64),
            nn.Linear(45, 89),
            nn.Tanh(),
            # nn.Linear(64, 128),
            nn.Linear(89, 178),
            nn.Tanh(),
            # nn.Linear(128, 372),
            nn.Linear(178, 357),
            nn.Tanh(),
            # nn.Linear(372, 79 * 68 * 95),
            nn.Linear(357, 714),
            nn.Tanh(),
            nn.Linear(714, brain_size),
            # nn.Linear(372, 79 * 68 * 95),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))


if __name__ == '__main__':
    for user in participants:
        for sub in subjects:
            print(user, sub)
            img = sitk.ReadImage(DATA_DIR + '/original_data/fMRI/sub-' + user + '/'
                                 + sub + '/sub-' + user + '_' + file_name[sub] + '.nii.gz')
            img = sitk.GetArrayFromImage(img)
            print(img.shape)
            time_point = img.shape[0]
            brain_x = img.shape[1]
            brain_y = img.shape[2]
            brain_z = img.shape[3]
            # (372, 68, 95, 79)

            # print(img.shape())

    train_loader = Data.DataLoader(dataset=img, batch_size=Batch_size, shuffle=True)

    auto_encoder = AutoEncoder(length=time_point, brain_x=brain_x, brain_y=brain_y, brain_z=brain_z)

    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    # init the first image
    # f, a = plt.subplots(2, N_Test_img, figsize=(5, 2))
    # plt.ion()
    # input_img = img[0]
    # view_data = Variable(torch.from_numpy(img) / 255.)

    # for i in range(N_Test_img):
    #     a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (79 * 95 * 68, -1)), cmap='gray')
    #     a[0][i].set_xticks(())
    #     a[0][i].set_yticks(())

    for epoch in range(Epoch):
        for step, input in enumerate(train_loader):
            # print(step)
            if step < (time_point // Batch_size):
                b_x = input.view(Batch_size, -1)
                b_y = input.view(Batch_size, -1)
            else:
                b_x = input.view(time_point % Batch_size, -1)
                b_y = input.view(time_point % Batch_size, -1)

            encoder_out, decoder_out = auto_encoder(b_x.float())
            encoder_out = encoder_out.float()
            # print(encoder_out.shape)
            decoder_out = decoder_out.float()

            loss = loss_func(decoder_out, b_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch:{}, Train_loss:{:.4f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())
                # print("before view_data_Shape:", view_data.shape)

                view_data = view_data.view(view_data.shape[0], -1)
                # print("after view_data_Shape:", view_data.shape)

                _, decoder_data = auto_encoder(view_data.float())
                # for i in range(N_Test_img):
                #     a[1][i].clear()
                #     a[1][i].imshow(np.reshape(decoder_data.data.numpy()[i], (79 * 95 * 68, -1)), cmap='gray')
                #     a[1][i].set_xticks(())
                #     a[1][i].set_yticks(())
                # plt.draw()
                # plt.pause(0.08)
    # plt.ioff()
    # plt.show()
    # torch.from_numpy(img)
    # view_data = img[:200].view(-1, 79 * 95).type(torch.FloatTensor) / 255.
    # encoder_data, _ = auto_encoder(view_data.float())
    # fig = plt.figure(2)
    # ax = Axes3D(fig)  # 3D graph
    # X = encoder_data.data[:, 0].numpy()
    # Y = encoder_data.data[:, 1].numpy()
    # Z = encoder_data.data[:, 2].numpy()
    # values = img.train_labels[:200].numpy()
    # for x, y, z, s in zip(X, Y, Z, values):
    #     c = cm.rainbow(int(255 * s / 9))
    #     ax.text(x, y, z, s, backgroundcolor=c)
    # ax.set_xlim(X.min(), X.max())
    # ax.set_xlim(Y.min(), Y.max())
    # ax.set_xlim(Z.min(), Z.max())
    # plt.show()
    # long running
    # do something other
end = time.perf_counter()
print((end - start) / 60)
plt.plot(Epoch_line, train_loss, '.-', label='Loss change')
plt.xticks(Epoch_line)
plt.xlabel('epoch')
plt.legend()
plt.savefig(DATA_DIR + '/output/Loss_brain/result_' + str(int(time.time())) + '.jpg')
plt.show()
