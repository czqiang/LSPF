import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xarray as xr
import numpy as np

# dimension
latent_dim     = 100


# read piomas
ds_piomas      = xr.open_dataset('piomas.nc')
piomas_data    = ds_piomas['piomas'].values

# read glor
ds_glor        = xr.open_dataset('glor.nc')
glor_data      = ds_glor['glor'].values  

# read oras
ds_oras        = xr.open_dataset('oras.nc')
oras_data      = ds_oras['oras'].values  

# read cglo
ds_cglo        = xr.open_dataset('cglo.nc')
cglo_data      = ds_cglo['cglo'].values  

# merge to one big data set
sea_ice_data   = np.concatenate((piomas_data, glor_data, oras_data, cglo_data), axis=0)


# Encoder
encoder_inputs = keras.Input(shape=(432, 432, 2))
x              = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x              = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x              = layers.Flatten()(x)
x              = layers.Dense(160, activation="relu")(x)
z_mean         = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var      = layers.Dense(latent_dim, name="z_log_var")(x)
encoder        = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")


# Sampler
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size     = tf.shape(z_mean)[1]
        epsilon    = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Decoder
latent_inputs   = keras.Input(shape=(latent_dim,))
x               = layers.Dense(108 * 108 * 64, activation="relu")(latent_inputs)
x               = layers.Reshape((108, 108, 64))(x)
x               = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x               = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2D(2, 3, activation="linear", padding="same")(x)
decoder         = keras.Model(latent_inputs, decoder_outputs, name="decoder")



# VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder                     = encoder
        self.decoder                     = decoder
        self.sampler                     = Sampler()
        self.total_loss_tracker          = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker             = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z                 = self.sampler(z_mean, z_log_var)
            reconstruction    = self.decoder(z)
            
            # Huber loss
            huber_lose_fn       = keras.losses.Huber(delta=1.0, reduction='none')
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    huber_lose_fn(data, reconstruction), axis=(1,2))
            )
            
            # KL loss
            kl_loss    = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            
            # total
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Compile model 
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))  


# Define callback
callbacks = [
    keras.callbacks.EarlyStopping(monitor='total_loss', patience=10, restore_best_weights=True, mode='min'),
    keras.callbacks.ModelCheckpoint(filepath='vae.weights.h5', save_best_only=True, save_weights_only=True, monitor='total_loss', mode='min')
]

# Train model
vae.fit(sea_ice_data, epochs=100, batch_size=16, callbacks=callbacks)