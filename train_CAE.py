import torch
import torch.nn.functional as F
import torch.utils.data.dataset as dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from pytorch_msssim import ssim

import torch_directml

from lib.maze_generation import generate_collection_of_mazes
from lib.models.convolutional_autoencoder import CAE


device = torch_directml.device()
shape = (15,15)

maze_set = generate_collection_of_mazes(shape,400,["r-prim","prim&kill","dfs"])

for i in range(len(maze_set)):
    maze_set[i] = maze_set[i].to(device)

train_set, test_set = dataset.random_split(maze_set,[0.8,0.2])

train_loader = DataLoader(train_set,4,shuffle=True)
test_loader = DataLoader(test_set,4,shuffle=True)

model = CAE(3,32).to(device)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=5e-3)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=15,eta_min=1e-5)

alpha = 0.65

for epoch in range(30):

    train_loss = 0.0
    for batch in train_loader:

        output = model(batch.float())

        loss = alpha * criterion(output,batch.float()) + (1-alpha)*(1-ssim(output,batch.float(),data_range=1))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
    
    train_loss = train_loss / len(train_loader)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} total loss {train_loss} | LR {current_lr}")

torch.save(model,f"weights/CAE{shape}.pth")

sim_avg = 0
for test in test_loader:
    output = model(test.float())

    test_flat = test.cpu().view(1,-1).detach()
    out_flat= output.round().cpu().view(1,-1).detach()
    

    cos_sim = F.cosine_similarity(test_flat, out_flat, dim=1)

    sim_avg += cos_sim

print(f"average cosine similarity {sim_avg / len(test_loader)}")

torch.save(model.encoder,f"weights/FeatureExtractor_{shape}.pth")