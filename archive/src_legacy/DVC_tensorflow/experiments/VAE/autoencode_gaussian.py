import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append("/Users/safaai/Library/CloudStorage/OneDrive-CompTech/Houman_Work/NPC")
from classes.objects import *
from vine_tree.tree_op import *
from pre_proc.preparation import prep_cop
import cProfile

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


def shuffle_dimensions(data, labels):
    unique_labels = np.unique(labels)
    shuffled_data = []

    for label in unique_labels:
        # Extract data for the current label
        data_label = data[labels == label]

        # Shuffle each dimension of the data for this label
        shuffled_label_data = np.copy(data_label)
        for i in range(data_label.shape[1]):
            np.random.shuffle(shuffled_label_data[:, i])

        # Append the shuffled data for this label
        shuffled_data.append(shuffled_label_data)

    # Concatenate the shuffled data for all labels
    return np.concatenate(shuffled_data, axis=0)

def calculate_cdf(data, labels):
    unique_labels = np.unique(labels)
    cdf_data = np.zeros_like(data)

    for label in unique_labels:
        # Extract data for the current label
        data_label = data[labels == label]

        # Compute CDF for each dimension of the data for this label
        cdf_label_data = np.argsort(np.argsort(data_label, axis=0), axis=0) / float(data_label.shape[0] - 1)
        
        # Assign the computed CDFs back to the corresponding positions in the overall CDF array
        cdf_data[labels == label] = cdf_label_data

    return cdf_data

# Define the sample_z function
def sample_z(args):
    mu, log_sigma, epsilon = args
    sigma = tf.exp(0.5 * log_sigma)
    return mu + sigma * epsilon

input_dim = 20
latent_dim = 50  # Latent dimension

# Define the encoder
input_x_shuffled = Input(shape=(input_dim,))
input_u = Input(shape=(input_dim,))
input_label = Input(shape=(2,))
x1 = Concatenate()([input_x_shuffled, input_label])
x1 = Dense(64, activation='relu')(x1)
x1 = Dense(32, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
mu = Dense(latent_dim, name='mu')(x1)
log_sigma = Dense(latent_dim, name='log_sigma')(x1)
x2 = Concatenate()([input_u, input_label])
x2 = Dense(128, activation='relu')(x2)
x2 = Dense(64, activation='relu')(x2)
x2 = Dense(32, activation='relu')(x2)
x2 = Dropout(0.2)(x2)
epsilon = Dense(latent_dim, activation='sigmoid', name='epsilon')(x2)
z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, log_sigma, epsilon])
encoder = Model([input_x_shuffled, input_u, input_label], [mu, log_sigma, z], name='encoder')

# Define the decoder
decoder_inputs = Input(shape=(latent_dim,))
decoder_input = Concatenate()([decoder_inputs, input_label])
x = Dense(32, activation="relu")(decoder_input)
x = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
decoder_outputs = Dense(input_dim, activation='linear')(x)
decoder = Model([decoder_inputs, input_label], decoder_outputs, name="decoder")


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
        x_shuffled, u, labels = x
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x_shuffled, u, labels])
            reconstructed = self.decoder([z, labels])
            loss = self.vae_loss(y, reconstructed, z_mean, z_log_var)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}
    
    def call(self, inputs):
        input_x_shuffled, input_u, input_label = inputs
        z_mean, z_log_var, z = self.encoder([input_x_shuffled, input_u, input_label])
        reconstructed = self.decoder([z, input_label])
        return reconstructed

# Example usage
num_samples_label1 = 7000
num_samples_label2 = 7000
label1 = 0
label2 = 1
data, labels = generate_data(num_samples_label1, num_samples_label2, label1, label2)
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)
x_train_shuffled = shuffle_dimensions(data, labels)
u_train = calculate_cdf(data,labels)

# Instantiate and compile the custom VAE
vae = CustomVAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer)


# Adjust batch size or dataset size to be compatible
batch_size = 128
dataset_size = data.shape[0]
adjusted_size = (dataset_size // batch_size) * batch_size  # Truncate dataset to fit batch size

# Use the adjusted dataset for training
vae.fit([x_train_shuffled[:adjusted_size], u_train[:adjusted_size], labels_one_hot[:adjusted_size]], 
        data[:adjusted_size], epochs=50, batch_size=batch_size)

# Visualization
cases = len(data)
sample = vae.predict([x_train_shuffled[:cases], u_train[:cases], labels_one_hot[:cases]])
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

##################################################################
##################################################################
##################################################################
########################## VINE PART #############################
##################################################################
##################################################################

dat = np.array(data[:1000,:], np.float64)

vine_type = "d-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50
vine_total_dim = dat.shape[1]
margin_vine = []
for i in range(0,vine_total_dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p) 

r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, vine_total_dim)
vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots, method, r_matrix)

sort_n = 'rand' 
e = prep_cop(dat, vine, sort_n)

param = False
binning = False
n_bins = 3
parallel = False        

vine_depth_fit = vine_total_dim + 1

gen_dict = {'parallel':parallel, 'binning':binning, 'param':param, 'vine_depth':vine_depth_fit, 'fitted':False}  
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':5}
bin_dict = {'n_bin':n_bins}

cProfile.run('vine.fit(dat,gen_dict,npc_dict,par_dict,bin_dict)')

from pred.prediction import*

exp_dim = 100
dim = 0
points = create_points(dat,dim,exp_dim)
min_dim = tf.math.reduce_min(dat[:,dim])
max_dim = tf.math.reduce_max(dat[:,dim])
y_vec = tf.linspace(min_dim-2e-16+1e-5,max_dim+2e-16,exp_dim)

p, p_cop, logp = vine.evaluation(points)
p1 = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)

p1 = reshaped_tensor = tf.reshape(p1, (dat.shape[0], exp_dim))
y_ML, y_EM = predict_response(p1, y_vec)
p, y_ml, y_em = predict_vine(dat,vine,dim,exp_dim)

from scipy import stats
#corr = stats.pearsonr(dat[:,dim], y_ml)
corre = stats.pearsonr(dat[:,dim], y_em)
#CORR = stats.pearsonr(dat[:,dim], y_ML)
CORRE = stats.pearsonr(dat[:,dim], y_EM)
print('Correlation: ' +  str(corre[0]) + ' , ' + str(CORRE[0]))
plt.figure()
#plt.plot(y_ml, dat[:,dim], 'g.')
plt.plot(y_em, dat[:,dim], 'r.')
plt.plot(y_EM, dat[:,dim], 'b.')
plt.show()

from sampling.vine_sample import *

sample, u, sample_pdf, sample_pdsmple = vine_copula_sample(vine,10000)

fig, axs = plt.subplots(vine.n_cop, vine.n_cop,figsize=(20,20))
for i in range(0,vine.n_cop,1):
    for j in range(i+1,vine.n_cop,1):
        axs[i,j].plot(data[:,i],data[:,j],'b.', markersize=1)    
        axs[i,j].plot(sample[:,i],sample[:,j],'r.', markersize=1)    
        axs[i,j].set_title(str(i+1)+","+str(j+1))

#for label_class in np.unique(label_classes):
#    compute_and_plot_correlation(data, sample, label_class)

# Call the function for each label class
for label_class in np.unique(label_classes):
    compute_and_plot_correlation(data, sample, label_class)