#
# Handwritten Character Recognition using EMNIST Dataset
#
# This script builds and trains a Convolutional Neural Network (CNN)
# to recognize handwritten alphabetic characters (A-Z).
#
# The process includes:
# 1. Loading the EMNIST 'letters' dataset using TensorFlow Datasets.
# 2. Preprocessing the image data for the CNN.
# 3. Defining the CNN model architecture.
# 4. Compiling and training the model.
# 5. Evaluating the model's accuracy on the test set.
#

# --- 1. Import Necessary Libraries ---
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def build_and_train_hcr_model():
    """
    Main function to load data, build, train, and evaluate the character recognition model.
    """
    # --- 2. Load and Prepare the EMNIST Dataset ---
    print("Step 1: Loading and Preparing EMNIST 'letters' Dataset...")

    # Load the EMNIST/letters dataset. It contains images of letters a-z.
    # The dataset is split into training and testing sets.
    # as_supervised=True loads the data as a (image, label) tuple.
    try:
        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        print("EMNIST dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and tensorflow_datasets is installed.")
        return

    # --- 3. Preprocessing ---
    print("\nStep 2: Preprocessing the Data...")

    # Define a function to normalize and reshape the images.
    def normalize_img(image, label):
        """Normalizes images to [0,1] and reshapes for CNN."""
        # EMNIST images are 28x28 grayscale.
        # We need to add a channel dimension for the CNN.
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=-1)
        # The labels are 1-based (1-26). We convert them to 0-based (0-25).
        return image, label - 1

    # Apply the preprocessing function to both training and testing datasets.
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    # Create batches and prefetch for performance.
    # Batching groups samples together for training.
    # Caching keeps the data in memory after the first epoch.
    # Prefetching loads the next batch while the current one is being processed.
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128).cache().prefetch(tf.data.AUTOTUNE)
    
    print("Data preprocessing complete.")
    num_classes = ds_info.features['label'].num_classes

    # --- 4. Define the CNN Model ---
    print("\nStep 3: Building the CNN Model Architecture...")

    model = tf.keras.models.Sequential([
        # Input Layer: 28x28 grayscale images
        tf.keras.layers.Input(shape=(28, 28, 1)),
        
        # First Convolutional Block
        # Conv2D learns features from the images.
        # MaxPooling2D reduces the spatial dimensions (downsampling).
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the 2D feature maps into a 1D vector
        tf.keras.layers.Flatten(),
        
        # Add a Dropout layer to prevent overfitting.
        # It randomly sets a fraction of input units to 0 at each update during training.
        tf.keras.layers.Dropout(0.5),
        
        # Dense (Fully Connected) Output Layer
        # The number of neurons equals the number of classes (26 letters).
        # Softmax activation gives a probability distribution over the classes.
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # --- 5. Compile the Model ---
    print("Step 4: Compiling the Model...")
    
    # Compile the model with an optimizer, loss function, and metrics.
    # 'adam' is a popular and effective optimizer.
    # 'sparse_categorical_crossentropy' is used for multi-class classification
    # when the labels are integers (not one-hot encoded).
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # --- 6. Train the Model ---
    print("\nStep 5: Training the Model...")
    
    # Train the model for a set number of epochs.
    # An epoch is one complete pass through the entire training dataset.
    history = model.fit(
        ds_train,
        epochs=5, # Using 5 epochs for a reasonable training time. Increase for higher accuracy.
        validation_data=ds_test
    )

    print("Model training complete.")

    # --- 7. Evaluate the Model ---
    print("\nStep 6: Evaluating Model Performance on the Test Set...")
    
    test_loss, test_acc = model.evaluate(ds_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # --- 8. Visualize Training History ---
    print("\nStep 7: Visualizing Training History...")
    
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    build_and_train_hcr_model()
