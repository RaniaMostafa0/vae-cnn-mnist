import torch
import torch.nn as nn

    
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)   # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1) # 7x7 -> 7x7
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)          # flatten
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 7x7 -> 14x14
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 14x14 -> 28x28
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)                        # 28x28 -> 28x28
        
    def forward(self, z):
        x = torch.relu(self.fc(z))                    # expand latent vector
        x = x.view(x.size(0), 128, 7, 7)             # unflatten to spatial
        x = torch.relu(self.conv1(x))                 # upsample to 14x14
        x = torch.relu(self.conv2(x))                 # upsample to 28x28
        x = torch.sigmoid(self.conv3(x))              # output [0,1]
        return x
    
    

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mu)
        z = mu + std * epsilon
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var
