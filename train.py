import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from model import VAE
from tqdm import tqdm
import json

latent_dim = 20
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

def VAE_Loss(original, reconstructed, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_loss

vae = VAE(latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def train_one_epoch():
    vae.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch, _ in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        reconstructed, mu, log_var = vae(batch)
        loss = VAE_Loss(batch, reconstructed, mu, log_var)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    vae.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, _ in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)
            reconstructed, mu, log_var = vae(batch)
            loss = VAE_Loss(batch, reconstructed, mu, log_var)
            total_loss += loss.item()
    return total_loss / len(val_loader)

all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')

for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(vae.state_dict(), 'best_vae_weights.pth')
        print(f'Best model saved at epoch {epoch+1}')

torch.save(vae.state_dict(), 'final_vae_weights.pth')
with open('losses.json', 'w') as f:
    json.dump({'train': all_train_losses, 'val': all_val_losses}, f)
print('Done.')
