import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def imshow(imgs, titles=None):
    fig, axs = plt.subplots(1, len(imgs), figsize=(12, 4))
    if len(imgs) == 1:
        imgs = [imgs]
    for i, img in enumerate(imgs):
        npimg = img.numpy()
        axs[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axs[i].axis('off')
        if titles:
            axs[i].set_title(titles[i])
    plt.show()

class AutoEncoder(nn.Module):
        def __init__(self) :
            super().__init__()
            # 3 Cin because we work with RGB imgs

            # Encoder
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
            self.max_pooling = nn.MaxPool2d(kernel_size=2, padding=0)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
            self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, padding=0)
            self.conv_latent = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
            
            # Decoder
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)
            self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv_output = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)



        def forward(self, img):
            # Encoder
            img = F.relu(self.conv1(img)) 
            img = self.max_pooling(img)
            img = F.relu(self.conv2(img))
            img = self.max_pooling_2(img) 
            img = F.relu(self.conv_latent(img))

            # Decoder
            img = self.upsample(img)
            img = F.relu(self.conv3(img))
            img = self.upsample1(img)
            img = F.relu(self.conv_output(img)) 

            return img


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor()])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_size = int(0.8 * len(trainset))

    val_size = len(trainset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
                                            
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    cnn = AutoEncoder()
    mse_error = nn.MSELoss()  
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  

    epochs = 1
    
    for epoch in range(epochs): 
        cnn.train()  # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = cnn(inputs)

            # Compute loss
            loss = mse_error(outputs, inputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] training loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Validation phase
        cnn.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for val_data in valloader:
                val_inputs, _ = val_data
                val_outputs = cnn(val_inputs)
                loss = mse_error(val_outputs, val_inputs)
                val_loss += loss.item()

        # Calculate and print average validation loss
        avg_val_loss = val_loss / len(valloader)
        print(f'[Epoch {epoch + 1}] validation loss: {avg_val_loss:.4f}')

    # Test phase
    cnn.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    i = 0
    with torch.no_grad():  # No gradient computation
        for test_data in testloader:
            test_inputs, j = test_data
            test_outputs = cnn(test_inputs)
            loss = mse_error(test_outputs, test_inputs)  
            test_loss += loss.item()

            if i == 0:  # Display for the first batch only
                imshow([torchvision.utils.make_grid(test_inputs), torchvision.utils.make_grid(test_outputs)], 
                    titles=["Test Input", "Test Output"])
            break

    # Calculate and print average test loss
    avg_test_loss = test_loss 
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    print('Training complete')