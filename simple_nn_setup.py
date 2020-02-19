import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler
from torch.utils import data
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class arc_dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.data[index][0]
        y = self.data[index][1]
        if self.transform:
            sample = self.transform(sample)
        return sample, y
    
    def __len__(self):
        return len(self.data)
        
train_dataset = arc_dataset(raw_train)
val_dataset = arc_dataset(raw_eval)
batch_size = 1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

in_size = train_dataset.__getitem__(0)[0].shape[0]
hidden_size = in_size * 2
out_size = train_dataset.__getitem__(0)[1].shape[0]

model = arc_nn(in_size, hidden_size, out_size)

loss = nn.MSELoss()

#model.type(torch.FloatTensor)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=2e-3)

class arc_nn(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        
        self.action = nn.Sequential(nn.Linear(in_size, hidden_size*2),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(3),
                                    nn.Linear(hidden_size*2, hidden_size),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(3),
                                    nn.Linear(hidden_size, out_size),
                                    nn.Softmax(dim=1))
    def forward(self, x):
        return self.action(x)
        
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    loss_accum = 0
    for i_step, (x, y) in enumerate(train_loader):
        pred = model(x.type(torch.FloatTensor))
        loss_value = loss(pred, y.type(torch.FloatTensor))
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        loss_accum += loss_value
    ave_loss = loss_accum / i_step
    print(f'epoch_{epoch} loss {ave_loss}')
