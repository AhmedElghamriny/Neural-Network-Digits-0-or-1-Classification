import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# DATASET FROM https://www.kaggle.com/datasets/artgor/handwritten-digits

def load_images_and_labels(base_path):
    images = []
    labels = []

    # Iterate through each label folder (0 and 1)
    for label in ['digit_0', 'digit_1']:
        label_folder = os.path.join(base_path, label)
        
        # Iterate through each image in the folder
        for image_name in os.listdir(label_folder):

            # Get the image path
            image_path = os.path.join(label_folder, image_name)
            
            # Load the image
            image = Image.open(image_path)
            # Convert to grayscale (pixels between 0 and 255)
            image = image.convert('L')
            # Resize the images to 20 x 20
            image = image.resize((20, 20))

            # Convert values to np array
            image_array = np.array(image)
            
            # Append the image and label to the lists
            images.append(image_array)
            if label == 'digit_0':
                labels.append(0)
            else:
                labels.append(1)
    
    return np.array(images), np.array(labels)

path = 'C:/Users/agham/Desktop/Docs/Projects/MLprojects/0_1DigitNN'
x, y = load_images_and_labels(path)

print("Original dataset shape:", x.shape)
print("Original labels shape:", y.shape)

# Separate the classes
x_0 = x[y == 0]
x_1 = x[y == 1]
y_0 = y[y == 0]
y_1 = y[y == 1]

print(f"Number of class 0 samples: {len(x_0)}")
print(f"Number of class 1 samples: {len(x_1)}")

# Undersample the majority class
n_samples = min(len(x_0), len(x_1))
x_1_undersampled = x_1[:n_samples]
y_1_undersampled = y_1[:n_samples]

# Combine the balanced dataset
x_balanced = np.vstack((x_0, x_1_undersampled))
y_balanced = np.hstack((y_0, y_1_undersampled))

# Shuffle the balanced dataset
x_balanced, y_balanced = shuffle(x_balanced, y_balanced, random_state=42)

print("Balanced dataset shape:", x_balanced.shape)
print("Balanced labels shape:", y_balanced.shape)

# Split the data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_balanced, y_balanced, test_size=0.2, random_state=42)

 
# Reshape and normalize the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_val = x_val.reshape(x_val.shape[0], -1) / 255.0


print("Training set shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

model = Sequential (
    [
        tf.keras.Input(shape=(400,)),
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ]
)

model.summary()

[layer1, layer2, layer3] = model.layers

W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=150
)

val_loss, val_accuracy = model.evaluate(x_val, y_val)
print(f"Validation accuracy: {val_accuracy:.4f}")


fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(len(x_train))
    
    # Display the image
    ax.imshow(x_train[random_index].reshape(20, 20), cmap='gray')

    # Predict using the Neural Network
    prediction = model.predict(x_train[random_index].reshape(1,400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    
    # Display the label above the image
    ax.set_title(f"{str(y_train[random_index])},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)

plt.show()