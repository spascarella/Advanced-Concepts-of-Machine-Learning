#Authors: Demetrio Loddo and Salvatore Pascarella
#Title: Convolutional AutoEncoder for image colorization 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imshow 
import tqdm

class AutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		# Encoder
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.max_pooling = nn.MaxPool2d(kernel_size=2, padding=0)  # Increased kernel size for efficiency
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
		self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, padding=0)  # Increased kernel size for efficiency
		self.conv_latent = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)  # Latent space with 32 channels

		# Decoder
		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # Upsample to match pooling
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)
		self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")  # Upsample to match pooling
		self.conv_output = nn.Conv2d(in_channels=24, out_channels=2, kernel_size=3, padding=1)

	def forward(self, img):
		# Encoder
		img = F.relu(self.conv1(img)) 
		img = self.max_pooling(img)
		img = F.relu(self.conv2(img))
		img = self.max_pooling_2(img) 
		img = F.relu(self.conv_latent(img))  # Latent representation (32 channels)

		# Decoder
		img = self.upsample(img)
		img = F.relu(self.conv3(img))
		img = self.upsample1(img)
		img = F.tanh(self.conv_output(img))  # Final output (32x32x3)

		return img
	#Trains the model and validates after each epoch
	def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
			
			for epoch in range(num_epochs):
				self.train()  # Set the model to training mode
				train_loss = 0.0
				for L, AB in train_loader:
					optimizer.zero_grad()
					# Forward pass
					outputs = self(L)
					loss = criterion(outputs, AB)
					# Backward pass and optimization
					loss.backward()
					optimizer.step()
					train_loss += loss.item()
				train_loss /= len(train_loader)
				print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
				# Validate the model
				self.validate_model(val_loader, criterion)
	
	#Validates the model and prints one reconstruction output.
	def validate_model(self, val_loader, criterion):

		self.eval()
		print("Evaluating the model...")

		val_loss = 0.0
		example_shown = False  # Flag to ensure only one image is plotted

		with torch.no_grad():
			for L, AB in val_loader:
				# Move data to the same device as the model
				L, AB = L.to(next(self.parameters()).device), AB.to(next(self.parameters()).device)

				# Forward pass
				outputs = self(L)
				loss = criterion(outputs, AB)
				val_loss += loss.item()

				# Show one example reconstruction
				if not example_shown:
					# Convert to RGB
					reconstructed_rgb = lab_to_rgb(L, outputs)  # Convert L and reconstructed AB to RGB
					original_rgb = lab_to_rgb(L, AB)           # Convert L and original AB to RGB

					# Convert the L channel to a NumPy array for plotting
					grayscale_input = L[0, 0].cpu().numpy()  # Select the first image, remove the channel dimension

					# Plot grayscale input, reconstructed RGB, and original RGB
					fig, axs = plt.subplots(1, 3, figsize=(15, 5))
					axs[0].imshow(grayscale_input, cmap="gray")
					axs[0].set_title("Input Grayscale (L Channel)")
					axs[0].axis("off")

					axs[1].imshow(reconstructed_rgb)
					axs[1].set_title("Reconstructed RGB")
					axs[1].axis("off")

					axs[2].imshow(original_rgb)
					axs[2].set_title("Original RGB")
					axs[2].axis("off")

					plt.show()

					example_shown = True  # Ensure only one example is shown

		val_loss /= len(val_loader)
		print(f"Validation Loss: {val_loss:.4f}")
		return val_loss

	#Tests the model and prints one reconstruction output from the test set.
	def test_model(self, test_loader, criterion):

		test_loss = 0.0
		example_shown = False  # Flag to ensure only one example is plotted

		with torch.no_grad():
			for L, AB in test_loader:
				# Move data to the same device as the model
				L, AB = L.to(next(self.parameters()).device), AB.to(next(self.parameters()).device)

				# Forward pass
				outputs = self(L)
				loss = criterion(outputs, AB)
				test_loss += loss.item()

				# Show one example reconstruction
				if not example_shown:
					# Convert to RGB
					reconstructed_rgb = lab_to_rgb(L, outputs)  # Convert L and reconstructed AB to RGB
					original_rgb = lab_to_rgb(L, AB)           # Convert L and original AB to RGB

					# Convert the L channel to a NumPy array for plotting
					grayscale_input = L[0, 0].cpu().numpy()  # Select the first image, remove the channel dimension

					# Plot grayscale input, reconstructed RGB, and original RGB
					fig, axs = plt.subplots(1, 3, figsize=(15, 5))
					axs[0].imshow(grayscale_input, cmap="gray")
					axs[0].set_title("Input Grayscale (L Channel)")
					axs[0].axis("off")

					axs[1].imshow(reconstructed_rgb)
					axs[1].set_title("Reconstructed RGB")
					axs[1].axis("off")

					axs[2].imshow(original_rgb)
					axs[2].set_title("Original RGB")
					axs[2].axis("off")

					plt.show()

					example_shown = True  # Ensure only one example is shown

		test_loss /= len(test_loader)
		print(f"Test Loss: {test_loss:.4f}")
		return test_loss

#Converts LAB channels (L, AB) to an RGB image in numpy format of shape (H, W, 3).  
def lab_to_rgb(L, AB):

    L = L.squeeze(1).cpu().numpy()  # Remove channel dimension and convert to numpy
    AB = AB.permute(0, 2, 3, 1).cpu().numpy()  # Convert to numpy and reshape to (batch_size, H, W, 2)

    rgb_images = []
    for l_channel, ab_channel in zip(L, AB):
        # Combine L and AB channels into a single LAB image
        lab_image = np.zeros((32, 32, 3), dtype=np.float32)
        lab_image[:, :, 0] = l_channel * 100  # Denormalize L to range [0, 100]
        lab_image[:, :, 1:] = ab_channel * 128  # Denormalize AB to range [-128, 128]

        # Convert LAB to RGB
        rgb_image = lab2rgb(lab_image)
        rgb_images.append(rgb_image)

    return rgb_images[0]  # Return the first image for visualization

#Converts a given dataset to LAB space and returns L and AB tensors.
def convert_to_lab(dataset):
	L_values = []
	AB_values = []

	for image, _ in tqdm.tqdm(dataset, desc="Converting dataset to LAB"):
		# Convert tensor to numpy format (H, W, C)
		image_np = image.permute(1, 2, 0).numpy()

		# Ensure the image is in the range [0, 1]
		image_np = np.clip(image_np, 0, 1)

		# Convert RGB to LAB
		lab_image = rgb2lab(image_np)

		# Append L and AB channels to the lists
		L_values.append(lab_image[:, :, 0] / 100)  # Normalize L channel to [0, 1]
		AB_values.append(lab_image[:, :, 1:] / 128)  # Normalize AB channels to [-1, 1]

	# Convert lists to numpy arrays
	L_values = np.array(L_values).reshape(-1, 32, 32, 1)  # Add channel dimension
	AB_values = np.array(AB_values)

	# Convert NumPy arrays to PyTorch tensors
	L_tensor = torch.from_numpy(L_values).permute(0, 3, 1, 2).float()  # (N, 1, H, W)
	AB_tensor = torch.from_numpy(AB_values).permute(0, 3, 1, 2).float()  # (N, 2, H, W)

	return L_tensor, AB_tensor

if __name__ == '__main__': 
	# Normalization
	transform = transforms.Compose(
		[transforms.ToTensor()])

	batch_size = 32

	# Load CIFAR10 dataset
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	train_size = int(0.8 * len(trainset))
	val_size = len(trainset) - train_size
	train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	# Convert train, validation, and test datasets
	L_train, AB_train = convert_to_lab(train_subset)
	L_val, AB_val = convert_to_lab(val_subset)
	L_test, AB_test = convert_to_lab(testset)

	# Create DataLoaders for training and validation
	trainloader = torch.utils.data.DataLoader(list(zip(L_train, AB_train)), batch_size=batch_size, shuffle=True)
	valloader = torch.utils.data.DataLoader(list(zip(L_val, AB_val)), batch_size=batch_size, shuffle=False)
	testloader = torch.utils.data.DataLoader(list(zip(L_test, AB_test)), batch_size=batch_size, shuffle=False)

	# Print data shape for verification
	print(f"L_train shape: {L_train.shape}, AB_train shape: {AB_train.shape}")
	print(f"L_val shape: {L_val.shape}, AB_val shape: {AB_val.shape}")
	print(f"L_test shape: {L_test.shape}, AB_test shape: {AB_test.shape}")

	#model parameters
	cae = AutoEncoder()
	criterion = nn.MSELoss()  
	optimizer = torch.optim.Adam(cae.parameters(), lr=0.001)
	
	# Train and validate the model: return the training and valdiation loss for each epoch together with one example reconstruction
	print("Training the model...")
	cae.train_model(trainloader, valloader, criterion, optimizer, num_epochs=10)
	
	# Test the model and display one example reconstruction
	print("Testing the model on the test set...")
	test_loss = cae.test_model(testloader, criterion)