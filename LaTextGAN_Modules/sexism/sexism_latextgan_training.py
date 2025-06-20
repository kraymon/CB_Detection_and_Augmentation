# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import numpy as np

plt.style.use('ggplot')

class GANTrainer:
    def __init__(self, generator, discriminator, autoencoder, word2vec_model):
        """Initialize the GAN trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            autoencoder: Autoencoder model
            word2vec_model: Pretrained glove model converted to word2vec format
        """
        self.generator = generator
        self.discriminator = discriminator
        self.autoencoder = autoencoder
        self.word2vec_model = word2vec_model
        
        # Fixed inputs for consistent visualization
        self.fixed_inputs = [tf.random.normal([1, 100]) for _ in range(2)]
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
    def gradient_penalty(self, real_samples, fake_samples):
        """Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real samples from dataset
            fake_samples: Generated samples from generator
            
        Returns:
            Gradient penalty term
        """
        batch_size = real_samples.shape[0]
        
        # Random interpolation between real and fake samples
        alpha = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            disc_interpolates = self.discriminator(interpolates)
            
        gradients = tape.gradient(disc_interpolates, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        penalty = tf.reduce_mean((slopes - 1.) ** 2)
        return penalty
    
    @tf.function
    def train_step(self, real_samples, train_generator=True):
        """Single training step for both generator and discriminator.
        
        Args:
            real_samples: Batch of real samples
            train_generator: Whether to update generator this step
            
        Returns:
            Generator loss and discriminator loss
        """
        batch_size = real_samples.shape[0]
        noise = tf.random.normal([batch_size, 100])
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_samples = self.generator(noise)
            
            real_output = self.discriminator(real_samples)
            fake_output = self.discriminator(fake_samples)
            
            # WGAN losses
            g_loss = -tf.reduce_mean(fake_output)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            
            # Add gradient penalty
            gp = self.gradient_penalty(real_samples, fake_samples)
            d_loss += 10 * gp
            
        # Update discriminator
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Update generator (if this is a generator step)
        if train_generator:
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
            
        return g_loss, d_loss
    
    def visualize_progress(self, epoch):
        """Visualize training progress with sample generations and loss curves."""
        clear_output()
        fixed_input = [self.generator(inp) for inp in self.fixed_inputs]
        random_input = [self.generator(tf.random.normal([1, 100])) for _ in range(2)]
        print()
        print(f"From Fixed Vector: {' '.join([self.word2vec_model.index_to_key[i.numpy()[0] -1] for i in self.autoencoder.Decoder.inference_mode(states=fixed_input[0], training=False) if i.numpy()[0] != 0])}")
        print(f"From Fixed Vector: {' '.join([self.word2vec_model.index_to_key[i.numpy()[0] -1] for i in self.autoencoder.Decoder.inference_mode(states=fixed_input[1], training=False) if i.numpy()[0] != 0])}")
        print()
        print(f"From Random Vector: {' '.join([self.word2vec_model.index_to_key[i.numpy()[0] -1] for i in self.autoencoder.Decoder.inference_mode(states=random_input[0], training=False) if i.numpy()[0] != 0])}")
        print(f"From Random Vector: {' '.join([self.word2vec_model.index_to_key[i.numpy()[0] -1] for i in self.autoencoder.Decoder.inference_mode(states=random_input[1], training=False) if i.numpy()[0] != 0])}")
                
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.g_losses, label='Generator')
        plt.plot(self.d_losses, label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()

        if len(self.g_losses) > 5:
            min_loss = -40
            max_loss = 40
            plt.ylim(min_loss - 1, max_loss + 1)  # Marges haut/bas pour visibilité
        plt.show()
    
    def train(self, dataset, epochs=150, lr=1e-4, d_steps=10):
        """Main training loop.
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            lr: Learning rate
            d_steps: Number of discriminator steps per generator step
        """
        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_g_loss = 0
            epoch_d_loss = 0
            step_count = 0
            
            for batch in tqdm(dataset, desc=f"Epoch {epoch + 1}"):
                # Train discriminator more frequently than generator
                train_g = (step_count % d_steps == 0)
                g_loss, d_loss = self.train_step(batch, train_g)
                
                # Accumulate losses
                epoch_g_loss += g_loss if train_g else 0
                epoch_d_loss += d_loss
                step_count += 1
            
            # Calculate average losses
            avg_g_loss = epoch_g_loss / (step_count / d_steps)
            avg_d_loss = epoch_d_loss / step_count
            
            self.g_losses.append(avg_g_loss.numpy())
            self.d_losses.append(avg_d_loss.numpy())
            
            # Visualize progress
            self.visualize_progress(epoch)
            print(f"Time: {time.time() - start_time:.2f}s")