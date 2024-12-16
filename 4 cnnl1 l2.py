#Lab Experiment 4
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., None]  # Add channel dimension
x_test = x_test[..., None]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Input layer with image size 28x28 and 1 channel
    layers.Conv2D(32, (3, 3), activation='relu'), 
    layers.MaxPooling2D(),  
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D(),  
    layers.Flatten(),  
    layers.Dense(128, activation='relu'),  
    # Uncomment the below lines for regularizers as per requirements
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(1e-4)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes (Fashion MNIST)
])

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")