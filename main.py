import os

from torch import optim
from tqdm import tqdm
from func.utils import *
from torch.utils.data import DataLoader
from func.utils import ImpedanceDataset1D
import matplotlib.pyplot as plt

# 1. load data

data_name = 'Overthrust' # Marmousi Overthrust SEAM
data_dic = np.load(('./data/' + data_name + '_20Hz.npy'), allow_pickle=True).item()
seismic = data_dic['seismic']
impedance = data_dic['impedance'] / 1000

random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

## 2. build train set (synthetic data)
## 2.1 parameter

[h, w] = impedance.shape
batch_size_seismic = 16
batch_size_inversion = 6
epoch_seismic = 30
epoch_inversion = 1000
lr_seismic = 0.001
lr_inversion = 0.01

## 2.2 build train set

mask = np.zeros([1,w])
if data_name == 'SEAM':
    mask[:,80::170] = 1 # 10 well logs
elif data_name == 'Marmousi':
    mask[:, 170::110] = 1  # 14 well logs
else:
    mask[:,90::100]=1 # 10 well logs

impedance_log = impedance * mask
seismic, impedance_log, log_mean, log_std = normal(seismic, impedance_log)

seismic_dataset1D = SeismicDataset1D(seismic)
seismic_dataloader1D = DataLoader(seismic_dataset1D,batch_size=batch_size_seismic, shuffle=True)

impedance_dataset1D = ImpedanceDataset1D(seismic, impedance_log, mask)
impedance_dataloader1D = DataLoader(impedance_dataset1D,batch_size=batch_size_inversion, shuffle=True)

## 3. train model

from func.model import Model
model = Model(input_channels=1, L=seismic.shape[0], mode='2D').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
e_num, i_num, r_num, d_num = model_summary(model)
print('total parameters:',e_num + i_num + r_num + d_num)
print('Encoder parameters: %d, Inverter parameters: %d, Reconstructor parameters: %d, Dimension reducer parameters: %d' % (e_num, i_num, r_num, d_num))

model.train()
loss_func = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr_seismic, weight_decay=1e-4)
loss_1 = []
loss_2 = []
loss_3 = []

mode = 'encoder'
with tqdm(total=epoch_seismic) as t:
    for i in range(epoch_seismic):
        for x, n in seismic_dataloader1D:

            re, n_out = model(x, mode)

            loss1 = loss_func(x, re)
            loss2 = loss_func(n, n_out)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_1.append(loss1.item())
        loss_2.append(loss2.item())

        t.set_postfix(loss=loss.item(), reconstruct_loss=loss1.item(), domain_predict_loss=loss2.item())
        t.update(1)

mode = 'inversion'
optimizer = optim.Adam(model.parameters(),lr_inversion, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epoch_inversion / 4), gamma=0.5)
model.partly_freeze()
with tqdm(total=epoch_inversion) as t:
    for i in range(epoch_inversion):
        for x, y in impedance_dataloader1D:

            out = model(x)
            loss = loss_func(y, out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss_3.append(loss.item())

        t.set_postfix(loss=loss.item())
        t.update(1)

## 4. validation

mode = 'validation'
output = seismic
re = seismic
n = []
print('infering')

import time
start = time.time()
model.eval()
for i in range(w):
    input = torch.tensor(seismic[:, i], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x, re_out, n_out = model(input, mode)
    re[:,i] = re_out.cpu().detach().squeeze(0).squeeze(0).numpy()
    n.append(n_out.cpu().detach().squeeze(0).squeeze(0).numpy() * w)
    output[:, i] = x.cpu().detach().squeeze(0).squeeze(0).numpy()

output = output * log_std + log_mean

snr, ssim, r2, mae, mse = metric(output, impedance)
end = time.time()
print('infering time consumption:',end-start)
print('SNR: %.4f , SSIM: %.4f , R2: %.4f, MAE: %.4f, MSE: %.4f ' % (snr, ssim, r2, mae, mse))

# 5. visualization

if data_name == 'Marmousi':
    min = 2
    max = 12
elif data_name == 'Overthrust':
    min = 2
    max = 18
else:
    min = 2
    max = 12

plt.subplot(1,3,1)
plt.imshow(impedance*(1-mask)+mask*10,'jet', vmin=min, vmax=max,aspect='auto')
plt.title('ground truth and wells')
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(output,'jet', vmin=min, vmax=max,aspect='auto')
plt.title('prediction')
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.abs(impedance-output),'gist_yarg', vmin=0, vmax=2,aspect='auto')
plt.title('residuals of impedance')
plt.colorbar()

plt.show()



