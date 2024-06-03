import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize a double convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        This class defines a block that consists of two sequential convolutional layers
        followed by batch normalization and ReLU activation. It is typically used as a
        building block within the UNet architecture.

        Forward Pass:
            - Input shape: (batch_size, in_channels, H, W)
            - Output shape: (batch_size, out_channels, H, W)

        Example:
            If you create a DoubleConv block with in_channels=64 and out_channels=128,
            the forward pass will take an input tensor of shape (batch_size, 64, H, W),
            apply two convolutional layers, and produce an output tensor of shape
            (batch_size, 128, H, W).
        """
        
        super(DoubleConv, self).__init__()
        
        ### START YOUR CODE HERE
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, device = 'cuda'),
            nn.BatchNorm2d(out_channels, device = 'cuda'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, device = 'cuda'),
            nn.BatchNorm2d(out_channels, device = 'cuda'),
            nn.ReLU(inplace=True,),
        )
    
        ### END YOUR CODE HERE

        
    def forward(self, x):
        """
        Forward pass through the double convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W).

        """
        return self.conv(x)

### Unet
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        """
        Initialize a U-Net architecture.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            features (list): List of integers representing the number of features at each U-Net level.

        This class defines a U-Net architecture for image segmentation tasks. It consists of an
        encoder (downsampling path), a bottleneck, and a decoder (upsampling path).


        Example:
        If you create a UNET with in_channels=3, out_channels=1, and features=[64, 128, 256, 512],
        the forward pass will take an input tensor of shape (batch_size, 3, H, W), process it
        through the U-Net architecture, and produce an output tensor of shape (batch_size, 1, H, W)
        for binary segmentation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).
            return_skipconnections (bool): Whether to return skip connections (default is False).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W).
            list: List of skip connections if `return_skipconnections` is True.
        """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.features = [64, 128, 256, 512]
        

        ### START YOUR CODE HERE
        ### the components
        ''' 
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        ##Encoder
        self.downs.append(DoubleConv(in_channels=in_channels, out_channels=features[0]))
        self.downs.append(DoubleConv(in_channels=features[0], out_channels=features[1]))
        self.downs.append(DoubleConv(in_channels=features[1], out_channels=features[2]))
        self.downs.append(DoubleConv(in_channels=features[2], out_channels=features[3]))
        
        ##Bottleneck
        self.bottleneck= DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)

        ##Decoder

        self.ups.append(nn.ConvTranspose2d(in_channels = features[-1]*2, out_channels = features[3], kernel_size=3))
        self.ups.append(DoubleConv(in_channels=features[3]*2, out_channels=features[3]))
        self.ups.append(nn.ConvTranspose2d(in_channels = features[3]*2, out_channels = features[2], kernel_size=3))
        self.ups.append(DoubleConv(in_channels=features[2]*2, out_channels=features[2]))
        self.ups.append(nn.ConvTranspose2d(in_channels = features[2]*2, out_channels= features[1], kernel_size=3))
        self.ups.append(DoubleConv(in_channels=features[1]*2, out_channels=features[1]))
        self.ups.append(nn.ConvTranspose2d(in_channels = features[1]*2, out_channels = features[0], kernel_size=3))
        self.ups.append(DoubleConv(in_channels=features[0]*2, out_channels=features[0]))
      
        ##Final Layer
        self.final_conv= nn.Conv2d(in_channels = features[0], out_channels = out_channels, kernel_size=1)
        '''
        
        skip_connections = []
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Encoder
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
          

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2, device = 'cuda'
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, device = 'cuda')
        

        ### END YOUR CODE HERE

    def forward(self, x, return_skipconnections=False):
        skip_connections=[]
        
        ### START YOUR CODE HERE
        ''' self.downs(x) #encoder
        self.bottleneck(x) #bottleneck
        self.ups(x) #decoder
        self.final_conv(x) #final_conv layer '''

        # Encoder
        for down in self.downs:
            x = x.to(device='cuda')
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]


        for idx in range(0, len(self.ups), 2):
          x = self.ups[idx](x)
          skip_connection = skip_connections[idx//2]

          if x.shape != skip_connection.shape:
            x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
          

        # Final
        x = self.final_conv(x)

        
        ### END YOUR CODE HERE
        if return_skipconnections:
            return x, reversed(skip_connections)
        return x
        
