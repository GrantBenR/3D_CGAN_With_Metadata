import torch
from torch import manual_seed, zeros, randn, ones, cat
from torch import Tensor, nn
from torch.utils.data import DataLoader

from classes.discriminator import Discriminator
from classes.generator import Generator
from classes.optimizer import CreateOptimizer
from classes.dataloader import LoadDefaultData, LoadTrainingData

class ConditionalGAN:
    def __init__(self):
        manual_seed(111)

    def TrainCGAN(
            self,
            num_epochs: int,
            data_loader: DataLoader,
            discriminator: Discriminator,
            generator: Generator,
            loss_function,
            train_data_length: int,
            train_metadata: Tensor
        ) -> Tensor:
        disc_optim = CreateOptimizer(discriminator)
        gen_optim = CreateOptimizer(generator)

        label_dim = 1

        for epoch in range(num_epochs):
            for _, (real_samples, real_metadata) in enumerate(data_loader):

                current_batch_size = real_samples.size(0) 
                
                real_samples_labels = ones((current_batch_size, label_dim))
                generated_samples_labels = zeros((current_batch_size, label_dim))
                
                latent_space_samples = randn((current_batch_size, 2))
                
                generated_samples = generator.forward(
                    z=latent_space_samples,
                    metadata=real_metadata
                )
                
                all_samples = cat((real_samples, generated_samples))
                all_metadata = cat((real_metadata, real_metadata)) 
                all_samples_labels = cat((real_samples_labels, generated_samples_labels))

                # Train Discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator.forward(
                    vectors=all_samples, 
                    metadata=all_metadata
                )
                loss_discriminator = loss_function(
                    output_discriminator,
                    all_samples_labels
                )
                loss_discriminator.backward()
                disc_optim.step()

                # Train Generator
                latent_space_samples = randn((current_batch_size, 2))
                
                random_batch_indices = torch.randint(0, train_data_length, (current_batch_size,))
                generator_target_metadata = train_metadata[random_batch_indices]

                generator.zero_grad()
                generated_samples = generator.forward(
                    z=latent_space_samples, 
                    metadata=generator_target_metadata
                )
                output_discriminator_generated = discriminator.forward(
                    vectors=generated_samples, 
                    metadata=generator_target_metadata
                )
                
                generator_targets = ones((current_batch_size, 1))
                loss_generator = loss_function(
                    output_discriminator_generated, 
                    generator_targets
                )
                loss_generator.backward()
                gen_optim.step()

            # Fixed the metric printing trigger so it doesn't try to look for a rigid step index
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Final Batch Size: {current_batch_size} | Loss D: {loss_discriminator:.4f} | Loss G: {loss_generator:.4f}")

        return generated_samples
    
    def RunGan(
            self, 
            batch_size=32, 
            loss_function=nn.BCELoss(),
            num_epochs=300
        ) -> Tensor:

        train_data, train_data_length, train_metadata, text_dim = LoadDefaultData()

        train_set = [
            (train_data[i], train_metadata[i]) for i in range(train_data_length)
        ]
        # drop_last=False allows the full dataset to be used, even if the final batch is small
        data_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False
        )
        metadata_dim = text_dim * 2
        discriminator = Discriminator(metadata_dim=metadata_dim)
        generator = Generator(metadata_dim=metadata_dim)

        generated_samples = self.TrainCGAN(
            num_epochs=num_epochs,
            data_loader=data_loader,
            discriminator=discriminator,
            generator=generator,
            loss_function=loss_function,
            train_data_length=train_data_length,
            train_metadata=train_metadata
        )
        return generated_samples.detach()

def main():
    cgan = ConditionalGAN()
    generated_samples = cgan.RunGan()
    return 0

if __name__ == "__main__":
    main()