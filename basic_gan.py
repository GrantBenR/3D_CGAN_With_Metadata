from torch import manual_seed, zeros, sin, rand, randn, ones, cat
from torch import Tensor, nn
from math import pi 
from torch.utils.data import DataLoader
from torch.optim import Adam

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

manual_seed(111)

train_data_length = 1024
train_data = zeros((train_data_length, 2))
train_data[:, 0] = 2 * pi * rand(train_data_length)
train_data[:, 1] = sin(train_data[:, 0])
train_labels = zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

batch_size = 32
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

discriminator = Discriminator()
generator = Generator()

lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()


optimizer_discriminator = Adam(discriminator.parameters(), lr=lr)
optimizer_generator = Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = ones((batch_size, 1))
        latent_space_samples = randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = zeros((batch_size, 1))
        all_samples = cat((real_samples, generated_samples))
        all_samples_labels = cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = randn((batch_size, 2))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")