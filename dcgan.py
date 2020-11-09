import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tensor of images, number of images,
    size per image.
    Plots and prints the images in a  uniform grid
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.show()


class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vecor, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        # Build the Neural Network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim*4),
            self.make_gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True)
            )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """
        Function to retrun a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.

        Args:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the utput feature representation should have
            kernel_size: the size of the  each convolutional filter (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, True if it is the finl layer, and False otherwise.
                (affects activation and batchnorm)
        """

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )
    
    def unsqueeze_noise(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width annd height = 1 and channels=z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        unsqueezed = noise.view(len(noise), self.z_dim, 1, 1)
        return unsqueezed

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters: 
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """

        x = self.unsqueeze_noise(noise)
        return self.gen(x)


def get_noise(n_samples, z_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vectors, a scalar
        device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


class Discriminator(nn.Module):
    """
    Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim*2),
            self.make_disc_block(hidden_dim*2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN,
        corresponding to a convolution, a btachnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter. (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, True if it is the final layer, False otherwise
                (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        return a 1-dimentsion tensor representing fake/real.
        Parameters:
            image: a flattened iamge tensor with dimension (im_dim)
        """
        disc_pred =self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
