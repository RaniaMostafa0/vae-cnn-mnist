import torch
import matplotlib.pyplot as plt
from model import VAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

latent_dim = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(latent_dim).to(device)
vae.load_state_dict(torch.load('best_vae_weights.pth'))
vae.eval()

# load test set
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

images, _ = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    reconstructed, _, _ = vae(images)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title('Original', loc='left', fontsize=12)
axes[1, 0].set_title('Reconstructed', loc='left', fontsize=12)
plt.tight_layout()
plt.savefig('results/reconstructions.png')
plt.close()

with torch.no_grad():
    z = torch.randn(8, latent_dim).to(device)
    generated = vae.decoder(z)

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    axes[i].imshow(generated[i].cpu().squeeze(), cmap='gray')
    axes[i].axis('off')
axes[0].set_title('Generated', loc='left', fontsize=12)
plt.tight_layout()
plt.savefig('results/generated.png')
plt.close()

with open('losses.json', 'r') as f:
    losses = json.load(f)

plt.figure(figsize=(8, 4))
plt.plot(losses['train'], label='Train Loss')
plt.plot(losses['val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('VAE Training and Validation Loss on MNIST')
plt.legend()
plt.savefig('results/training_loss.png')
plt.close()
print('Done.')