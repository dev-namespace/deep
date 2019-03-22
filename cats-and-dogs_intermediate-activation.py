from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# model loading
model = load_model('./models/cats-vs-dogs.h5')

# preprocessing single image
img_path = './datasets/dogs-vs-cats/small/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255. # As it was in the model
# shape = (1, 150, 150, 3)

# plt.imshow(img_tensor[0])
# plt.show()

# activation model
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]] # first 8
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # The model will return the layer outpts

activations = activation_model.predict(img_tensor)
first = activations[0]
# plt.matshow(first[0, :, :, 6], cmap='viridis') # showing 4th channel of first output
# plt.show()

# Activation grid visualization
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]
    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]
    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
