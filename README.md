# Sketch-to-face-transformation-using-pix2pix
# List all files in the dataset folder 
data_paths = [] 
for dirname, _, filenames in os.walk(dataset_path): 
print(f"Current directory: {dirname}") 
for filename in filenames: 
data_paths.append(os.path.join(dirname, filename)) 
# Print the total number of files found 
print(f"Total number of files: {len(data_paths)}") 
# Optionally, print the first few paths 
for path in data_paths[:10]:  # Limit output to the first 10 files 
print(path) 
# This Python 3 environment comes with many helpful analytics libraries installed 
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python 
# For example, here's several helpful packages to load 
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
# Input data files are available in the read-only "../input/" directory 
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input 
directory 
data_paths = [] 
for dirname, _, filenames in os.walk('/content/dataset/N'): 
print(dirname) 
for filename in filenames: 
data_paths.append(os.path.join(dirname, filename)) 
print("total number of data:",len(data_paths)) 
4fd0b6ce88b7f7"  # Replace with your actual API key 
# Download the dataset 
!kaggle datasets download -d arbazkhan971/cuhk-face-sketch-database-cufs 
from PIL import Image, ImageOps 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import os 
import random 
from natsort import natsorted, os_sorted, ns 
import time 
import datetime 
import random 
# Set the environment variable for the Kaggle API key 
os.environ['KAGGLE_CONFIG_DIR'] = "/content" 
# Create a directory for unzipping the dataset (if not exists) 
os.makedirs('/content/dataset', exist_ok=True) 
# Unzip the dataset into the /content/dataset directory 
!unzip cuhk-face-sketch-database-cufs.zip -d /content/dataset/N 
sketches_path = r"/content/dataset/N/sketches" 
portrait_path = r"/content/dataset/N/photos" 
# Makes a list of all files in each folder. 
sketches_imgname = os.listdir(sketches_path)  # Now listed first 
portrait_imgname = os.listdir(portrait_path)  # Now listed second 
# Make the sketch paired to portrait (order reversed from original) 
sketches_imgname = natsorted(sketches_imgname, alg=ns.IGNORECASE)  # Now first 
portrait_imgname = natsorted(portrait_imgname, alg=ns.IGNORECASE)  # Now second 
# Combines folder paths with filenames to create complete file paths. 
sketches_imgs = [sketches_path+"/"+imgname for imgname in sketches_imgname]  # Now first 
portrait_imgs = [portrait_path+"/"+imgname for imgname in portrait_imgname]  # Now second 
imgnum = 5 # Number of image pairs to show 
imgrow = 2 # Creates 2 rows (sketches top, photos bottom - reversed from original) 
for i in range(imgnum): 
image = Image.open(sketches_imgs[i])  # Now loading sketch first 
image2 = Image.open(portrait_imgs[i])  # Now loading portrait second 
ax = plt.subplot(imgrow,imgnum,i+1) 
plt.imshow(image, cmap="gray")  # Sketch shown in grayscale 
plt.xticks([]) 
plt.yticks([]) 
ax = plt.subplot(imgrow,imgnum,i+imgnum+1) 
plt.imshow(image2)  # Portrait shown in color 
plt.xticks([]) 
plt.yticks([]) 
#Ready-to-use image data for AI training! 
#This is like preparing ingredients before cooking - cleaning, cutting, and arranging them so they're easy to 
use in the next steps.      
 
def load(image_path): 
    image = tf.io.read_file(image_path) 
    image = tf.io.decode_jpeg(image,channels=IN_CHANNEL) 
    image = tf.cast(image, tf.float32) 
    return image 
 
def resize(image, height, width): 
  image = tf.image.resize(image, [height, width], 
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
  return image 
 
def normalize(image): 
  image = (image/ 127.5) - 1 
  return image 
 
#data augmentation 
def flip(image): 
    image = tf.image.flip_left_right(image) 
    return image 
 
adjustment_seed = 1 
random.seed(adjustment_seed) 
def random_adjustment(sketch_image, portrait_image):  # Changed parameter names to reflect sketch 
input 
    rand_bright = random.uniform(1,5) 
    rand_contrast = random.uniform(1,2) 
 
    sketch_image = tf.image.adjust_brightness(sketch_image,delta=rand_bright) 
portrait_image = tf.image.adjust_brightness(portrait_image,delta=rand_bright) 
sketch_image = tf.image.adjust_contrast(sketch_image,contrast_factor=rand_contrast) 
portrait_image = tf.image.adjust_contrast(portrait_image,contrast_factor=rand_contrast) 
sketch_image = tf.cast(sketch_image, tf.float32) 
portrait_image = tf.cast(portrait_image, tf.float32) 
return sketch_image, portrait_image  # Changed return variable names 
IMG_SIZE = 256 
#the in channel and out channel has to be the same for tf batch 
#even though the generated image is greyscale (1channel), it is converted into RGB (3channels) 
IN_CHANNEL = 3 
OUT_CHANNEL = IN_CHANNEL 
TRAIN_BATCH = 4 
#temp_dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7,8,9,10]) 
# temp_dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6],[7,8],[9,10]]) 
# for temp_data in temp_dataset: 
#     
print(temp_data) 
# temp_dataset = temp_dataset.batch(3) 
# for temp_data in temp_dataset: 
#     
print(temp_data) 
#This is like preparing twin pairs of shoes - making sure both shoes in each pair are cleaned, 
#polished, and stored together before shipping!            
tempsize = 4 
imgs = [] 
for i in range(tempsize): 
# Load and process sketch first (now as input) 
sketch_img = load(sketches_imgs[i]) 
sketch_img = resize(sketch_img, IMG_SIZE, IMG_SIZE) 
sketch_img = normalize(sketch_img) 
# Load and process portrait second (now as target) 
portrait_img = load(portrait_imgs[i]) 
portrait_img = resize(portrait_img, IMG_SIZE, IMG_SIZE) 
portrait_img = normalize(portrait_img) 
# Store as [sketch, portrait] pair 
imgs.append(tf.convert_to_tensor([sketch_img, portrait_img])) 
print(tf.convert_to_tensor(imgs).shape) 
#Splitting data into training (95%) and testing (5%) sets 
#Creating modified versions of each image to help the AI learn better 
#Processing Each Image Pair 
#For every sketch-portrait pair:  # Changed comment 
#Loads and resizes both images 
#Creates 3 versions of each pair: Original version, Flipped (mirror image), Randomly adjusted 
(brightness/contrast changes) 
#Normalizes all images (scales pixel values to -1 to 1 range)*/ 
BUFFER_SIZE = len(sketches_imgs) # Total number of images (now using sketches_imgs) 
TRAIN_SIZE = int(BUFFER_SIZE*0.95) # 95% for training 
TEST_SIZE = int(BUFFER_SIZE - TRAIN_SIZE) # 5% for testing 
train_imgs_idx = [] 
test_imgs_idx = [] 
train_imgs = [] 
test_imgs = [] 
 
for i in range(BUFFER_SIZE): 
        # Process sketch first (input) 
        sketch_img = load(sketches_imgs[i])  # Changed order 
        sketch_img = resize(sketch_img, IMG_SIZE, IMG_SIZE) 
 
        # Process portrait second (target) 
        portrait_img = load(portrait_imgs[i])  # Changed order 
        portrait_img = resize(portrait_img, IMG_SIZE, IMG_SIZE) 
 
        # Data augmentation - sketch 
        sketch_img2 = flip(sketch_img)  # Changed order 
        sketch_img2 = normalize(sketch_img2) 
 
        # Data augmentation - portrait 
        portrait_img2 = flip(portrait_img)  # Changed order 
        portrait_img2 = normalize(portrait_img2) 
 
        # Random adjustments (order changed in function call) 
        sketch_img3, portrait_img3 = random_adjustment(sketch_img, portrait_img)  # Changed order 
        sketch_img3 = normalize(sketch_img3) 
        portrait_img3 = normalize(portrait_img3) 
 
        # Normalize originals 
        sketch_img = normalize(sketch_img) 
        portrait_img = normalize(portrait_img) 
 
        if (i < TRAIN_SIZE): 
            # Store pairs as [sketch, portrait]  # Changed order 
            train_imgs.append(tf.convert_to_tensor([sketch_img, portrait_img])) 
            train_imgs.append(tf.convert_to_tensor([sketch_img2, portrait_img2])) 
            train_imgs.append(tf.convert_to_tensor([sketch_img3, portrait_img3])) 
        else: 
            test_imgs.append(tf.convert_to_tensor([sketch_img, portrait_img])) 
            test_imgs.append(tf.convert_to_tensor([sketch_img2, portrait_img2])) 
            test_imgs.append(tf.convert_to_tensor([sketch_img3, portrait_img3])) 
 
train_set = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_imgs)) 
print("images in trainset:", len(train_set)) 
train_set = train_set.shuffle(len(train_set)) 
 
for i, data in enumerate(train_set): 
    if (i >= imgnum): 
        break 
 
train_set = train_set.batch(TRAIN_BATCH) 
print("trainset len:", len(train_set)) 
 
for batch_items in train_set.repeat().take(1): #taking 1 batch to show 
    temp_num_of_img_pair = len(batch_items) 
    idx = 1 
    for image_pair in batch_items: 
        sketch_image, portrait_image = image_pair  # Changed variable names 
 
        ax = plt.subplot(temp_num_of_img_pair, 2, idx) 
        plt.imshow(sketch_image, cmap="gray")  # Added cmap for sketch 
        plt.xticks([]) 
        plt.yticks([]) 
        idx += 1 
 
        ax = plt.subplot(temp_num_of_img_pair, 2, idx) 
        plt.imshow(portrait_image)  # Portrait in color 
        plt.xticks([]) 
        plt.yticks([]) 
        idx += 1 
 
test_set = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_imgs)) 
test_set = test_set.shuffle(len(test_set)) 
#Think of them as Lego blocks - the downsample blocks make sketch features smaller as 
#they go deeper into the network, while upsample blocks rebuild portrait features on the way out.            ⇅            
#the output image size will always be input image size/strides as padding="same" 
def downsample(feature, kernel, apply_batchnorm=False): 
    """Downsamples sketch features while preserving spatial information""" 
    down_block = tf.keras.Sequential()  # Changed variable name to be more descriptive 
 
    if apply_batchnorm: 
        down_block.add( 
            tf.keras.layers.Conv2D(feature, kernel, strides=2, 
                                 padding='same', use_bias=False)) 
        down_block.add(tf.keras.layers.BatchNormalization()) 
    else: 
        down_block.add( 
            tf.keras.layers.Conv2D(feature, kernel, strides=2, 
                                 padding='same')) 
 
    down_block.add(tf.keras.layers.LeakyReLU()) 
 
    return down_block 
 
#the output image size will always be input image size/strides as padding="same" 
def upsample(feature, kernel, apply_batchnorm=False, apply_dropout=False, dropout=0): 
    """Upsamples features to reconstruct portrait from sketch embeddings""" 
    up_block = tf.keras.Sequential()  # Changed variable name to be more descriptive 
 
    if apply_batchnorm: 
        up_block.add( 
            tf.keras.layers.Conv2DTranspose(feature, kernel, strides=2, 
                                          padding='same', use_bias=False)) 
        up_block.add(tf.keras.layers.BatchNormalization()) 
    else: 
        up_block.add( 
            tf.keras.layers.Conv2DTranspose(feature, kernel, strides=2, 
                                          padding='same')) 
 
    if apply_dropout: 
        up_block.add(tf.keras.layers.Dropout(dropout)) 
 
    up_block.add(tf.keras.layers.LeakyReLU()) 
 
    return up_block 
input_shape = (1, IMG_SIZE, IMG_SIZE, IN_CHANNEL) #1 represents batch size \ 1 sketch, 256x256 pixels, 3 
color channels 
x = tf.random.normal(input_shape) # Creates random sketch pattern 
 
inp = data[0]  # Takes first sketch from dataset 
x = inp[tf.newaxis, ...] # Adds batch dimension (now shape [1,256,256,3]) 
y = downsample(12,4)(x) # Extracts sketch features using 12 filters, size-4 kernel 
print(y.shape) # Outputs: (1, 128, 128, 12) 
 
x_restored = upsample(OUT_CHANNEL,4)(y) # Generates portrait from features 
print(x_restored.shape) # Outputs: (1, 256, 256, 3) 
plt.imshow(x_restored[0, ...]) # Displays generated portrait 
plt.colorbar() # Shows value scale 
# Uses tanh activation to produce pixel values between -1 and 1 
# Gives the final generated portrait 
# This architecture (called a U-Net) is great for: 
# Sketch-to-portrait conversion, Style transfer, Photo enhancement 
# It's like teaching an AI to "imagine" what a portrait would look like from a sketch!      →         
def Generator(): 
    sketch_input = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, IN_CHANNEL])  # Changed variable 
name 
 
    # Feature extraction from sketch (downsampling) 
    down_stack = [ 
        downsample(64, 4),  # (batch_size, 128, 128, 64) 
        downsample(128, 4),  # (batch_size, 64, 64, 128) 
        downsample(256, 4),  # (batch_size, 32, 32, 256) 
        downsample(512, 4),  # (batch_size, 16, 16, 512) 
    ] 
 
    # Portrait generation (upsampling) 
    up_stack = [ 
        upsample(256, 4),  # (batch_size, 32, 32, 256) 
        upsample(128, 4),  # (batch_size, 64, 64, 128) 
        upsample(64, 4),  # (batch_size, 128, 128, 64) 
    ] 
 
    # Final portrait output layer 
    portrait_output = tf.keras.layers.Conv2DTranspose(OUT_CHANNEL, 4,  # Changed variable name 
                                           strides=2, 
                                           padding='same', 
                                           activation='tanh')  # (batch_size, 256, 256, channel) 
 
    x = sketch_input  # Changed variable name 
 
    # Sketch feature extraction 
    skips = [] 
    for down in down_stack: 
        x = down(x) 
        skips.append(x) 
    skips = reversed(skips[:-1]) 
 
    # Portrait generation with skip connections 
    for up, skip in zip(up_stack, skips): 
        x = up(x) 
        x = tf.keras.layers.Concatenate()([x, skip]) 
 
    x = portrait_output(x)  # Changed variable name 
 
    return tf.keras.Model(inputs=sketch_input, outputs=x)  # Changed variable name 
generator = Generator()  # Renamed to reflect task 
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, 
to_file='sketch_to_portrait_generator.png')  # Updated filename 
generator.summary()  # Consistent naming 
# Visualization code (commented out) 
# temp_img_num = 3 
# for step, (input_image, target) in train_set.repeat().take(temp_img_num).enumerate(): 
#     ax = plt.subplot(temp_img_num,2,(int(step)*2)+1) 
#     plt.imshow(input_image[0]) 
#     plt.xticks([]) 
#     plt.yticks([]) 
# 
#     gen_output = generator(input_image[0][tf.newaxis, ...], training=False) 
#     ax = plt.subplot(temp_img_num,2,(int(step)*2)+2) 
#     plt.imshow(gen_output[0,...], cmap="gray") 
#     plt.xticks([]) 
#     
plt.yticks([]) 
# Single image test 
inp = data[0]  # Keep original variable name 
gen_output = generator(inp[tf.newaxis, ...], training=False)  # Keep original variable name 
# Visualization 
plt.figure(figsize=(10,5)) 
ax = plt.subplot(1,2,1) 
plt.imshow(inp) 
plt.title('Input Sketch') 
plt.xticks([]) 
plt.yticks([]) 
ax = plt.subplot(1,2,2) 
plt.imshow(gen_output[0, ...]) 
plt.title('Generated Portrait') 
plt.xticks([]) 
plt.yticks([]) 
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) #will measure how convincing the 
images are 

def generator_loss(disc_generated_output, gen_output, target): 
# Part 1: Adversarial loss (how convincing the generated portrait is) 
gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output) 
# Part 2: Accuracy loss (how close to real portrait) 
# Mean absolute error between generated and target portrait 
l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) 
    # Combine both losses (realism + accuracy) 
    total_gen_loss = gan_loss + (LAMBDA * l1_loss) 
 
    return total_gen_loss, gan_loss, l1_loss 
 
# total_gen_loss: Combined loss for portrait generation quality 
# gan_loss: Measures how realistic the generated portrait appears 
# l1_loss: Measures pixel-level accuracy of generated portrait 
def Discriminator(): 
    sketch_input = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, IN_CHANNEL], name='sketch_input') 
    portrait_input = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUT_CHANNEL], 
name='portrait_input') 
 
    x = tf.keras.layers.concatenate([sketch_input, portrait_input]) 
 
    # Feature extraction with LayerNormalization for stability 
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x) 
    x = tf.keras.layers.LayerNormalization()(x) 
    x = tf.keras.layers.LeakyReLU(0.2)(x) 
 
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x) 
    x = tf.keras.layers.LayerNormalization()(x) 
    x = tf.keras.layers.LeakyReLU(0.2)(x) 
 
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x) 
    x = tf.keras.layers.LayerNormalization()(x) 
    x = tf.keras.layers.LeakyReLU(0.2)(x) 
 
    # Final output 
    x = tf.keras.layers.Conv2D(1, 1, strides=1, padding='valid')(x) 
    return tf.keras.Model(inputs=[sketch_input, portrait_input], outputs=x) 
input_shape = (1, 32, 32, 256)  # 1 represents batch size of sketch features 
x = tf.random.normal(input_shape)  # Random tensor simulating sketch features 
print("Sketch feature shape:", x.shape)  # Outputs: (1, 32, 32, 256) 
 
y = tf.keras.layers.Conv2D(1, 1, strides=1, padding='valid')(x)  # Convert to discrimination output 
print("Discriminator output shape:", y.shape)  # Outputs: (1, 32, 32, 1) 
discriminator = Discriminator() 
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file='discriminator.png') 
# This is like having an art critic analyze which portrait regions look unrealistic 
inp = data[0]  # Original variable name kept 
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)  # Original call preserved 
plt.imshow(disc_out[0, ...], cmap="gray")  # Same visualization parameters 
plt.colorbar()  # Kept unchanged 
def discriminator_loss(real_output, fake_output, real_images, fake_images, input_sketches): 
    real_loss = loss_object(tf.ones_like(real_output), real_output) 
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output) 
 
    # Gradient penalty with corrected norm calculation 
    alpha = tf.random.uniform(shape=[tf.shape(real_images)[0], 1, 1, 1]) 
    interpolated = alpha * real_images + (1 - alpha) * fake_images 
 
    with tf.GradientTape() as gp_tape: 
        gp_tape.watch(interpolated) 
        pred = discriminator([input_sketches, interpolated], training=True) 
 
    gradients = gp_tape.gradient(pred, [interpolated])[0] 
 
    # Corrected gradient norm calculation 
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])) 
    gp = tf.reduce_mean((slopes - 1.)**2) 
 
    return real_loss + fake_loss + 10 * gp 
# Change only these values at the top of your code: 
LAMBDA = 10  # Reduced from 100 
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # Same as before 
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)  # Slower (originally 2e-4) 
@tf.function 
def train_step(input_sketches, target_portraits, epoch): 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: 
        gen_output = generator(input_sketches, training=True) 
 
        disc_gen_output = discriminator([input_sketches, gen_output], training=True) 
        disc_real_output = discriminator([input_sketches, target_portraits], training=True) 
 
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss( 
            disc_gen_output, gen_output, target_portraits) 
 
        disc_loss = discriminator_loss( 
            disc_real_output, disc_gen_output, 
            target_portraits, gen_output, 
            input_sketches) 
 
    # Rest of your existing training code remains identical 
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables) 
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables) 
 
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables)) 
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables)) 
 
    return gen_total_loss, disc_loss 
def train(train_dataset, epochs=120): 
    for epoch in range(epochs): 
        start = time.time() 
 
        for batch in train_dataset: 
            input_sketches = [] 
            target_portraits = [] 
 
            for sketch, portrait in batch:  # More explicit unpacking 
                input_sketches.append(sketch) 
                target_portraits.append(portrait) 
 
            input_sketches = tf.convert_to_tensor(input_sketches) 
            target_portraits = tf.convert_to_tensor(target_portraits) 
 
            gen_loss, disc_loss = train_step(input_sketches, target_portraits, epoch) 
 
        if epoch % 2 == 0: 
            print(f"Epoch {epoch} - Generator loss: {gen_loss:.4f}, Discriminator loss: {disc_loss:.4f}") 
            print(f"Time for 2 epochs: {time.time() - start:.2f}s") 
 
# Start training 
train(train_set) 
def denormalize(image): 
    """Converts normalized image tensor (-1 to 1) back to pixel values (0-255) for visualization""" 
    image = (image + 1)  # Shift from [-1,1] to [0,2] 
    image = image * 127.5  # Scale to [0,255] 
    return image 
import os 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
def denormalize(image): 
"""Converts normalized image tensor (-1 to 1) back to pixel values (0-255) for visualization""" 
image = (image + 1) * 127.5 
return image 
def load_and_preprocess_image(path): 
"""Loads and preprocesses image for model input""" 
img = load_img(path, target_size=(256, 256)) 
img = img_to_array(img) 
img = (img / 127.5) - 1  # Normalize to [-1, 1] 
return tf.convert_to_tensor(img, dtype=tf.float32) 
# 1. Load file paths from your sketches folder 
sketch_folder = "/content/dataset/N/sketches" 
sketch_paths = [os.path.join(sketch_folder, fname) for fname in sorted(os.listdir(sketch_folder)) if 
fname.endswith(('.png', '.jpg', '.jpeg'))] 
# 2. Load and preprocess images 
sketch_tensors = [load_and_preprocess_image(path) for path in sketch_paths] 
# 3. Create test dataset 
test_set = tf.data.Dataset.from_tensor_slices(sketch_tensors).batch(1) 
# 4. Visualization 
temp_img_num = min(5, len(sketch_tensors)) 
plt.figure(figsize=(15, 6)) 
for step, input_sketch in enumerate(test_set.take(temp_img_num)): 
# Row 1: Input sketch 
ax = plt.subplot(2, temp_img_num, step + 1) 
    plt.imshow((input_sketch[0].numpy() * 0.5 + 0.5))  # Convert from [-1, 1] to [0, 1] 
    plt.title(f"Sketch {step+1}") 
    plt.xticks([]) 
    plt.yticks([]) 
 
    # Row 2: Generated portrait 
    if 'generator' in globals(): 
        generated = generator(input_sketch, training=False) 
        denorm_output = denormalize(generated[0]) 
        denorm_output = tf.cast(tf.clip_by_value(denorm_output, 0, 255), tf.uint8) 
        ax = plt.subplot(2, temp_img_num, temp_img_num + step + 1) 
        plt.imshow(denorm_output.numpy().astype(np.uint8)) 
        plt.title(f"Generated {step+1}") 
        plt.xticks([]) 
        plt.yticks([]) 
 
plt.tight_layout() 
plt.show() 
 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio as compare_psnr 
from skimage.metrics import structural_similarity as compare_ssim 
 
def denormalize(image): 
    """Converts image from [-1, 1] to [0, 255] uint8""" 
    image = (image + 1.0) * 127.5 
    return tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8) 
 
# Evaluate generator on test set 
psnr_scores = [] 
ssim_scores = [] 
 
# You can also batch this if needed 
for i, (sketch, real_portrait) in enumerate(test_imgs): 
    sketch_input = sketch[tf.newaxis, ...]  # Add batch dimension 
    generated = generator(sketch_input, training=False)[0]  # Remove batch dim 
 
    # Denormalize images 
    gen_img = denormalize(generated).numpy() 
    real_img = denormalize(real_portrait).numpy() 
 
    # Convert to grayscale or RGB (SSIM needs same channel count) 
    if gen_img.shape[-1] == 3: 
        multichannel = True 
    else: 
        multichannel = False 
 
    # Compute PSNR and SSIM 
    psnr = compare_psnr(real_img, gen_img, data_range=255) 
    ssim = compare_ssim(real_img, gen_img, data_range=255, channel_axis=-1) 
 
    psnr_scores.append(psnr) 
    ssim_scores.append(ssim) 
 
    print(f"[{i+1}/{len(test_imgs)}] PSNR: {psnr:.2f}, SSIM: {ssim:.4f}") 
 
# Average scores 
avg_psnr = np.mean(psnr_scores) 
avg_ssim = np.mean(ssim_scores) 
 
print(f"\n   
Average PSNR: {avg_psnr:.2f}") 
print(f"   
Average SSIM: {avg_ssim:.4f}")
