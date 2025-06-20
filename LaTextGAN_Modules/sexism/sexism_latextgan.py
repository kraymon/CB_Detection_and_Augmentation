# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Model, layers

class ResidualBlock(Model):
    def __init__(self, units=100):
        """Initialize a Residual Block with ReLU activation.
        
        Args:
            units (int): Number of units in the dense layers
        """
        super(ResidualBlock, self).__init__()
        self.dense1 = layers.Dense(units, activation="relu")
        self.dense2 = layers.Dense(units)
        
    def call(self, x):
        """Forward pass of the residual block.
        
        Args:
            x (tensor): Input tensor
            
        Returns:
            tensor: Output after residual connection and ReLU
        """
        residual = x
        x = self.dense1(x)
        x = self.dense2(x)
        x += residual
        return tf.nn.relu(x)

class Generator(Model):
    def __init__(self, latent_dim=100, output_dim=600, num_blocks=40):
        """Initialize the Generator model.
        
        Args:
            latent_dim (int): Dimension of the input noise
            output_dim (int): Dimension of the output
            num_blocks (int): Number of residual blocks
        """
        super(Generator, self).__init__()
        self.blocks = [ResidualBlock() for _ in range(num_blocks)]
        self.output_layer = layers.Dense(output_dim)
        
    def call(self, x):
        """Forward pass of the generator.
        
        Args:
            x (tensor): Input noise tensor
            
        Returns:
            tensor: Generated output
        """
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class Discriminator(Model):
    def __init__(self, num_blocks=40):
        """Initialize the Discriminator model.
        
        Args:
            num_blocks (int): Number of residual blocks
        """
        super(Discriminator, self).__init__()
        self.initial_layer = layers.Dense(100)
        self.blocks = [ResidualBlock() for _ in range(num_blocks)]
        self.output_layer = layers.Dense(1)
        
    def call(self, x):
        """Forward pass of the discriminator.
        
        Args:
            x (tensor): Input tensor (real or fake)
            
        Returns:
            tensor: Discriminator output (logits)
        """
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)