import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Data generation for two labels
def generate_data(num_samples_label1, num_samples_label2, label1, label2):
    mat1 = np.eye(10)
    mat2 = np.array([[0, 0.1, 0.02, 0.01, 0, 0, 0, 0, 0, 0],
                     [0.01, 0, 0.04, 0.03, 0.01, 0, 0, 0, 0, 0],
                     [0.02, 0.04, 0, 0.05, 0.03, 0.01, 0, 0, 0, 0],
                     [0.01, 0.03, 0.05, 0, 0.04, 0.02, 0.01, 0, 0, 0],
                     [0, 0.01, 0.03, 0.04, 0, 0.1, 0.03, 0.01, 0, 0],
                     [0, 0, -0.01, 0.02, -0.05, 0, -0.04, 0.02, -0.01, 0],
                     [0, 0, 0, -0.01, -0.03, -0.4, 0, -0.05, -0.03, -0.01],
                     [0, 0, 0, 0, -0.01, -0.02, -0.05, 0, -0.04, -0.02],
                     [0, 0, 0, 0, 0, 0.01, -0.03, -0.04, 0, -0.05],
                     [0, 0, 0, 0, 0, 0, 0.01, -0.02, -0.05, 0]]) * 2 
    mat3 = np.transpose(mat2)
    mat4 = np.eye(10)

    top_row = np.concatenate((mat1, mat2), axis=1)
    bottom_row = np.concatenate((mat3, mat4), axis=1)
    cov_matrix_label1 = np.concatenate((top_row, bottom_row), axis=0)
    cov_matrix_label2 = -cov_matrix_label1   # or np.abs(cov_matrix_label1) if needed
    for i in range(cov_matrix_label2.shape[0]):
        cov_matrix_label2[i, i] = 1

    mean = np.zeros(20)
    data_label1 = np.random.multivariate_normal(mean, cov_matrix_label1, size=num_samples_label1)
    data_label2 = np.random.multivariate_normal(mean, cov_matrix_label2, size=num_samples_label2)

    data = np.concatenate((data_label1, data_label2), axis=0)
    labels = np.concatenate((np.full(num_samples_label1, label1), np.full(num_samples_label2, label2)))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]



# Define the sample_z function
def sample_z(args):
    mu, log_sigma = args
    sigma = tf.exp(0.5 * log_sigma)
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mu))
    return mu + sigma * epsilon

input_dim = 20
latent_dim = 50  # Latent dimension

# Define the encoder
input_data = Input(shape=(input_dim,))
x = Dense(64, activation='relu')(input_data)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
mu = Dense(latent_dim, name='mu')(x)
log_sigma = Dense(latent_dim, name='log_sigma')(x)
z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, log_sigma])
encoder = Model(input_data, [mu, log_sigma, z], name='encoder')

# Define the decoder
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(32, activation="relu")(decoder_inputs)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
decoder_outputs = Dense(input_dim, activation='linear')(x)
decoder = Model(decoder_inputs, decoder_outputs, name="decoder")


class CustomVAE(Model):
    def __init__(self, encoder, decoder):
        super(CustomVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def vae_loss(self, y, reconstructed, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(tf.square(y - reconstructed))
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstructed = self.decoder(z)
            loss = self.vae_loss(y, reconstructed, z_mean, z_log_var)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed



# Example usage
num_samples_label1 = 7000
num_samples_label2 = 7000
label1 = 0
label2 = 1
data, labels = generate_data(num_samples_label1, num_samples_label2, label1, label2)
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)

# Instantiate and compile the custom VAE
vae = CustomVAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer)

# Train the model
batch_size = 128
dataset_size = data.shape[0]
adjusted_size = (dataset_size // batch_size) * batch_size  # Truncate dataset to fit batch size
vae.fit(data[:adjusted_size], data[:adjusted_size], epochs=50, batch_size=batch_size)

# Visualization
cases = len(data)
sample = vae.predict(data[:cases])
data_sample = data[:cases, :]


label_classes = labels

dim = input_dim // 2
n = dim * 2

def compute_and_plot_correlation(data, sample, label_class):
    corr_sample = np.zeros((n, n), dtype=np.float64)
    corr_data = np.zeros((n, n), dtype=np.float64)

    data_class = data[label_classes == label_class]
    sample_class = sample[label_classes == label_class]

    fig, axs = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                axs[i, j].plot(data_class[:, i], data_class[:, j], 'b.', markersize=1)
                axs[i, j].plot(sample_class[:, i], sample_class[:, j], 'r.', markersize=1)
                corr_sample[i, j] = stats.spearmanr(sample_class[:, i], sample_class[:, j])[0]
                corr_data[i, j] = stats.spearmanr(data_class[:, i], data_class[:, j])[0]
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()

    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Correlation Data
    fig, axs = plt.subplots(1, 1)
    plt.imshow(corr_data, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()

    # Correlation Sample
    fig, axs = plt.subplots(1, 1)
    plt.imshow(corr_sample, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()

# Call the function for each label class
for label_class in np.unique(label_classes):
    compute_and_plot_correlation(data, sample, label_class)