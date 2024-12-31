import os
from pathlib import Path
import jiwer
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras import utils
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Define constants
IMG_HEIGHT = 50   # Height of the captcha image
IMG_WIDTH = 200   # Width of the captcha image
NUM_CHANNELS = 1  # Grayscale images
NUM_CLASSES = 36  # 26 letters + 10 digits
CAPTCHA_LENGTH = 5  # Fixed length of the captcha

# Define the CNN model
def create_cnn_model():
    model = models.Sequential()

    # Convolutional Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(CAPTCHA_LENGTH * NUM_CLASSES))
    model.add(layers.Reshape((CAPTCHA_LENGTH, NUM_CLASSES)))  # Reshape for each character
    model.add(layers.Softmax(axis=-1))  # Softmax over NUM_CLASSES for each character

    return model

# Compile the model
model = create_cnn_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Print the model summary
model.summary()

# Function to preprocess data
def preprocess_data(images, labels):
    # Load and process images
    processed_images = []
    for image_path in images:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to required dimensions
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize
        processed_images.append(image)

    # Convert to NumPy array and add channel dimension
    processed_images = np.expand_dims(np.array(processed_images), axis=-1)

    # One-hot encode labels for each character
    char_to_num = {char: idx for idx, char in enumerate(characters)}
    encoded_labels = []
    for label in labels:
        # Encode each character and one-hot encode it
        encoded_label = [
            utils.to_categorical(char_to_num[char], num_classes=NUM_CLASSES)
            for char in label
        ]
        encoded_labels.append(encoded_label)

    # Convert to NumPy array and reshape
    encoded_labels = np.array(encoded_labels, dtype=np.float32)  # Shape: (num_samples, CAPTCHA_LENGTH, NUM_CLASSES)

    return processed_images, encoded_labels


# Define the data directory
Dir_path = os.getcwd()
file_internal_path = "\modules\Captcha_bypass_algorithm\captcha_images"
FULL_path = Dir_path + file_internal_path

data_dir = Path(FULL_path)

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = sorted(set(char for label in labels for char in label))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Split data into training, validation, and test sets
x_train_full, x_test, y_train_full, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1765, random_state=42)

# Preprocess the data
x_train, y_train = preprocess_data(x_train, y_train)
x_val, y_val = preprocess_data(x_val, y_val)
x_test, y_test = preprocess_data(x_test, y_test)

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")



def plot_evaluation_metrics(metrics):
    """
    Plot evaluation metrics as a bar chart with values displayed above each column in percentage.
    
    Parameters:
    - metrics (dict): A dictionary containing evaluation metrics (e.g., WER, CER, etc.)
    """
    # Extract metric names and convert values to percentages
    metric_names = list(metrics.keys())
    metric_values = [v * 100 for v in metrics.values()]  # Convert to percentage
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color='green', edgecolor='black')
    
    # Add percentage values above the bars
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate
            bar.get_height() + 1,              # Y-coordinate (slightly above the bar)
            f'{value:.2f}%',                   # Text (formatted as percentage)
            ha='center',                       # Center-align text
            fontsize=10                        # Font size
        )
    
    # Add labels and title
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Values (%)", fontsize=12)
    plt.title("Evaluation Metrics (%)", fontsize=14)
    plt.ylim(0, max(metric_values) + 10)  # Adjust Y-axis to fit text above bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# Function to calculate evaluation metrics
def evaluate_model(pred_texts, orig_texts):
    """
    Evaluate the model using various metrics:
    - WER (Word Error Rate)
    - CER (Character Error Rate)
    - Character Accuracy
    - Word Accuracy
    - Precision, Recall, F1 Score
    """
    # Initialize counters for word accuracy
    correct_word_count = 0
    total_word_count = len(orig_texts)
    
    # Initialize counters for character accuracy
    correct_char_count = 0
    total_char_count = 0

    # Calculate WER and CER using `jiwer` package
    wer = jiwer.wer(orig_texts, pred_texts)
    cer = jiwer.cer(orig_texts, pred_texts)

    # Calculate word and character-level metrics
    for orig, pred in zip(orig_texts, pred_texts):
        # Word-level accuracy
        if orig == pred:
            correct_word_count += 1
        
        # Character-level accuracy
        for oc, pc in zip(orig, pred):
            if oc == pc:
                correct_char_count += 1
        total_char_count += max(len(orig), len(pred))
    
    # Calculate Word Accuracy
    word_accuracy = correct_word_count / total_word_count
    
    # Calculate Character Accuracy
    char_accuracy = correct_char_count / total_char_count

    # Prepare lists for Precision, Recall, and F1-Score
    y_true, y_pred = [], []
    for orig, pred in zip(orig_texts, pred_texts):
        y_true.extend(list(orig))
        y_pred.extend(list(pred))
    
    # Align lengths of `y_true` and `y_pred` for metrics calculation
    max_length = max(len(y_true), len(y_pred))
    y_true = y_true + [''] * (max_length - len(y_true))
    y_pred = y_pred + [''] * (max_length - len(y_pred))

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # Print the results
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Character Accuracy: {char_accuracy:.4f}")
    print(f"Word Accuracy: {word_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics = {
        "WER": wer,
        "CER": cer,
        "Char_Accuracy": char_accuracy,
        "Word_Accuracy": word_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }
    plot_evaluation_metrics(metrics)
    return metrics

# Decode predictions back to text
def decode_predictions(preds, characters, captcha_length):
    """
    Decode the predictions into text using argmax to get class indices.
    """
    pred_texts = []
    for pred in preds:  # Iterate over batch
        text = "".join([characters[np.argmax(char_vector)] for char_vector in pred])
        pred_texts.append(text)
    return pred_texts


# Predict on the test set
predictions = model.predict(x_test)

# Decode predictions back to text
pred_texts = decode_predictions(predictions, characters, CAPTCHA_LENGTH)

# Ensure ground truth is in text format
# Ensure ground truth is in text format
orig_texts = ["".join([characters[np.argmax(c)] for c in label.reshape(CAPTCHA_LENGTH, NUM_CLASSES)]) for label in y_test]

# Call the evaluate_model function
metrics = evaluate_model(pred_texts, orig_texts)
