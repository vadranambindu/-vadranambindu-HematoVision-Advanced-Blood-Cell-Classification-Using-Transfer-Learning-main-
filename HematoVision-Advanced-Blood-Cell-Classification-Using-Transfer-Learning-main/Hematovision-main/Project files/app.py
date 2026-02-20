import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Data preprocessing for training
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

if not os.path.exists('./dataset/TRAIN'):
    print(f"Error: Dataset not found at {os.path.abspath('./dataset/TRAIN')}")
    
    # Debugging: Print what is actually found
    if os.path.exists('./dataset'):
        print(f"Found 'dataset' folder. It contains: {os.listdir('./dataset')}")
    else:
        print(f"'dataset' folder is MISSING in {os.getcwd()}")
        print(f"Folders found here: {os.listdir('.')}")
        
    print("Please create a 'dataset' folder and place the 'TRAIN' and 'TEST' folders inside it.")
    exit()

if not os.path.exists('./dataset/TEST'):
    print(f"Error: Test dataset not found at {os.path.abspath('./dataset/TEST')}")
    print("Please ensure the 'TEST' folder is inside the 'dataset' folder.")
    exit()

train_generator = train_datagen.flow_from_directory('./dataset/TRAIN',  # Directory with training images
    target_size=(224, 224),  # MobileNetV2 input size
    batch_size=32,
    class_mode='categorical'
)

# Data preprocessing for testing
test_datagen = image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './dataset/TEST',  # Directory with testing images
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for evaluation
)

# Load MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False
# Create the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Use the number of classes dynamically
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model for a maximum of 5 epochs
history = model.fit(train_generator, epochs=5)
# Evaluate the model
test_generator.reset()  # Reset the generator for evaluation
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Accuracy: {accuracy:.2f}')
# Generate classification report
report = classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys())
print(report)

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Save the model at the last stage
model.save('blood_cell_classifier_mobilenetv2.h5')