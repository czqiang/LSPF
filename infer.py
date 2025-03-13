import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import xarray as xr

# dim
latent_dim = 100

# sampler
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * (epsilon * 1.) # epsilon is set to vary sampler
    
# encoder
encoder_inputs = keras.Input(shape=(432, 432, 2))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(160, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

# decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(108 * 108 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((108, 108, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2D(2, 3, activation="linear", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# vae
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampler(z_mean, z_log_var)
        return self.decoder(z)
    
# ini vae
sampler = Sampler()
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# load weights
vae.load_weights('vae.weights.h5')

# read obs.nc
obs_ds = xr.open_dataset('obs.nc')
obs_ice = obs_ds['obs'].values  
obs_z_mean, obs_z_log_var = encoder(obs_ice)
obs_z = sampler(obs_z_mean, obs_z_log_var).numpy()

# read sis2.nc
sis2_ds = xr.open_dataset('sis2.nc')
sis2_ice = sis2_ds['sis2'].values  

#  filter
sis2_z_list = []
weights = []


for i in range(100):
    sis2_z_mean, sis2_z_log_var = encoder(sis2_ice)
    sis2_z = sampler(sis2_z_mean, sis2_z_log_var).numpy()
    sis2_z_list.append(sis2_z)

    # testing sigma can be set to fix value 
    # sigma = 1.
    # sigma = np.exp(sis2_z_log_var.numpy())
    
    # compute weights
    weight = np.exp(-np.sum((obs_z - sis2_z)**2) / (2 * sigma))
    weights.append(weight)

# normalize weights
weights = np.array(weights)
weights /= weights.sum()

# weighted mean
final_z = np.average(sis2_z_list, axis=0, weights=weights)

# decode from latent space
final_ice = decoder(final_z).numpy()

# no time axis
final_ice = np.squeeze(final_ice, axis=0)

# save
ds_out = xr.Dataset(
    {
        "fused_ice": (["y", "x", "variable"], final_ice)
    },
    coords={
        "y": obs_ds["ncl1"].values,
        "x": obs_ds["ncl2"].values,
        "variable": ["concentration", "thickness"], 
    }
)

ds_out.to_netcdf('vae.nc')