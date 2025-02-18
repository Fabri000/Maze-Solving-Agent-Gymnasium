import torch
import torch.nn.functional as F
import torch.utils.data.dataset as dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_directml

from lib.maze_generation import generate_collection_of_mazes
from lib.models.convolutional_autoencoder import CAE


device = torch_directml.device()

shape = (15,15)

maze_set = generate_collection_of_mazes(shape,400)

for i in range(len(maze_set)):
    maze_set[i] = maze_set[i].to(device)

train_set, test_set = dataset.random_split(maze_set,[0.8,0.2])

train_loader = DataLoader(train_set,1,shuffle=True)
test_loader = DataLoader(test_set,1,shuffle=True)

model = CAE(3,16).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-2)

for epoch in range(10):

    train_loss = 0.0
    for batch in train_loader:
        
        optimizer.zero_grad()

        output = model(batch.float())

        loss = criterion(output,batch.float())
        loss.backward()

        optimizer.step()
    
        train_loss += loss
    
    train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch} total loss {train_loss}")

torch.save(model,f"weights/CAE_{shape}.pth")

loaded_model = torch.load(f"weights/CAE_{shape}.pth").to(device)

sim_avg = 0
for test in test_loader:
    output = loaded_model(test.float())

    test_flat = test.cpu().view(1,-1).detach()
    out_flat= output.round().cpu().view(1,-1).detach()

    cos_sim = F.cosine_similarity(test_flat, out_flat, dim=1)

    sim_avg += cos_sim

print(f"average cosine similarity {sim_avg / len(test_loader)}")