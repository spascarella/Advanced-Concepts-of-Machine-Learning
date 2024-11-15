import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
    
    def train_model(self, trainloader, valloader, optimizer, criterion, num_epochs):
        training_losses = []  # To store average training loss per epoch
        validation_losses = []  # To store average validation loss per epoch

        for epoch in range(num_epochs):
            # Training phase
            self.train()  # Set the model to training mode
            running_loss = 0.0
            batch_count = 0

            for i, data in enumerate(trainloader, 0):
                inputs, _ = data
                optimizer.zero_grad()
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                # Accumulate loss
                running_loss += loss.item()
                batch_count += 1

            # Calculate average training loss over all batches in the epoch
            avg_train_loss = running_loss / batch_count
            training_losses.append(avg_train_loss)  # Store epoch training loss for plotting
            print(f'[Epoch {epoch + 1}] Average Training Loss: {avg_train_loss:.4f}')

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation for validation
                for val_data in valloader:
                    val_inputs, _ = val_data
                    val_outputs = self(val_inputs)
                    loss = criterion(val_outputs, val_inputs)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(valloader)
            validation_losses.append(avg_val_loss)  # Store epoch validation loss for plotting
            print(f'[Epoch {epoch + 1}] Validation Loss: {avg_val_loss:.4f}')

        # Plot the evolution of the training and validation losses across epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), training_losses, marker='o', color='blue', linestyle='-', label="Training Loss")
        plt.plot(range(1, num_epochs + 1), validation_losses, marker='o', color='orange', linestyle='-', label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

        return training_losses, validation_losses


    def test_model(self, testloader, criterion):
        self.eval()
        test_loss = 0.0
        batch_losses = []  # To store loss for each batch
        i = 0
        
        with torch.no_grad():  # No gradient computation
            for batch_idx, test_data in enumerate(testloader):
                test_inputs, _ = test_data
                test_outputs = self(test_inputs)
                loss = criterion(test_outputs, test_inputs)
                
                test_loss += loss.item()
                batch_losses.append(loss.item())  # Save the loss of each batch
                
                if batch_idx == 0:  # Display for the first batch only
                    imshow([torchvision.utils.make_grid(test_inputs), torchvision.utils.make_grid(test_outputs)], 
                        titles=["Test Input", "Test Output"])
                    
        avg_test_loss = test_loss / len(testloader)
        print(f'Average Test Loss: {avg_test_loss:.4f}')

        return avg_test_loss

# Helper function for displaying images
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

#function to extract the dimensions of each layer
def extract_conv_params(model):
    conv_params = []
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            conv_params.append({
                'type': 'conv',
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size[0],  # Assuming square kernels
                'stride': layer.stride[0],
                'padding': layer.padding[0]
            })
        elif isinstance(layer, nn.MaxPool2d):
            conv_params.append({
                'type': 'pool',
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'padding': layer.padding
            })
        elif isinstance(layer, nn.Upsample):
            # For upsampling, only scale_factor is typically relevant
            conv_params.append({
                'type': 'upsample',
                'scale_factor': layer.scale_factor
            })

    return conv_params

#Calculate the size of the output of each layer in a convolutional neural network and returns the latent space size
def calculate_latent_space_size(input_size, conv_params, target_layer=5):
    width, height, channels = input_size
    for idx, layer in enumerate(conv_params):
        if layer['type'] == 'conv':
            # Convolution layer
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            padding = layer['padding']
            
            width = (width - kernel_size + 2 * padding) // stride + 1
            height = (height - kernel_size + 2 * padding) // stride + 1
            channels = layer['out_channels']  # Update number of channels to out_channels
            
        elif layer['type'] == 'pool':
            # Pooling layer (assuming square kernel for simplicity)
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            padding = layer['padding']
            
            width = (width - kernel_size + 2 * padding) // stride + 1
            height = (height - kernel_size + 2 * padding) // stride + 1

        # Stop once we've reached the target layer (layer 5 in this case)
        if idx + 1 == target_layer:
            return (width, height, channels)

    return (width, height, channels)



if __name__ == '__main__': 
    #normalization
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
    epochs = 5
    input_size = (32, 32, 3)  # Input image size (width, height, channels)
    # Extract convolutional and pooling layer configurations
    conv_params = extract_conv_params(cnn)
    latent_space_size = calculate_latent_space_size(input_size, conv_params)
    print("Size of the latent space representation:", latent_space_size)
    num_epochs = 5
    training_losses, validation_losses = cnn.train_model(trainloader, valloader, optimizer, mse_error, num_epochs)
    cnn.test_model(testloader, mse_error)