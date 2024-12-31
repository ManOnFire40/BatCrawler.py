import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf
import keras
from keras import ops
from keras import layers

## Load the data: [Captcha Images](https://www.kaggle.com/fournierp/captcha-version-2-images)





"""
The dataset contains 1040 captcha files as `png` images. The label for each sample is a string,
the name of the file (minus the file extension).
We will map each character in the string to an integer for training the model. Similary,
we will need to map the predictions of the model back to strings. For this purpose
we will maintain two dictionaries, mapping characters to integers, and integers to characters,
respectively.
"""

Dir_path = os.getcwd()
file_internal_path = "\modules\Captcha_bypass_algorithm\captcha_images"
FULL_path = Dir_path + file_internal_path

# Path to the data directory
# Path("./captcha_images_v2/")
data_dir = Path(FULL_path)

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Batch size for training and validation
batch_size = 16
# Desired image dimensions
img_width = 200
img_height = 50
# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4
# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])


## Preprocessing
# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


"""
def split_data(images, labels, train_size=0.7, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = ops.arange(size)
    if shuffle:
        indices = keras.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid
Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
"""


def split_data(images, labels, train_size=0.7, val_size=0.15, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Create indices and shuffle if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)

    # 3. Calculate the sizes of each split
    train_samples = int(size * train_size)
    val_samples = int(size * val_size)

    # 4. Split data into training, validation, and testing sets
    x_train = images[indices[:train_samples]]
    y_train = labels[indices[:train_samples]]

    x_valid = images[indices[train_samples:train_samples + val_samples]]
    y_valid = labels[indices[train_samples:train_samples + val_samples]]

    x_test = images[indices[train_samples + val_samples:]]
    y_test = labels[indices[train_samples + val_samples:]]

    return x_train, x_valid, x_test, y_train, y_valid, y_test

# Splitting the data
x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = ops.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = ops.transpose(img, axes=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


## Create `Dataset` objects
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Testing Dataset
testing_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
testing_dataset = (
    testing_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

combined_dataset = validation_dataset.concatenate(testing_dataset)
# Optional: Shuffle the combined dataset
combined_dataset = combined_dataset.shuffle(buffer_size=1000)  # Adjust buffer size as needed


## Visualize the data



_, ax = plt.subplots(4, 4, figsize=(10, 5))

# Iterate over the training dataset batches
for batch in train_dataset:
    images = batch["image"]
    labels = batch["label"]
    
    # Get the actual number of images in this batch
    batch_size = len(images)
    
    # Iterate over the minimum of 16 or the batch size
    for i in range(min(batch_size, 16)):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
        
plt.show()


## Model


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"),
        vals_sparse,
        ops.cast(label_shape, dtype="int64"),
    )


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
model.summary()

## Training



# TODO restore epoch count.
epochs = 100
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)





def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)


# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
prediction_model.summary()

pred_texts_global=[]
orig_texts_global = []

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some validation samples
for batch in validation_dataset:
    batch_images = batch["image"]
    batch_labels = batch["label"]
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    #pred_texts_global = decode_batch_predictions(preds)  
    pred_texts_global.extend(pred_texts)       
    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)      
    orig_texts_global.extend(orig_texts)  # Accumulate original texts            
    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")       
plt.show()
##################################################### modifications
def load_model():
    """Build and return the OCR model."""
    model = build_model()
    return model

def load_prediction_model():
    """Build and return the prediction model (for inference)."""
    model = load_model()
    prediction_model = keras.models.Model(
        model.input[0], model.get_layer(name="dense2").output
    )
    return prediction_model

    
# Load the prediction model
prediction_model = load_prediction_model()






# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import jiwer
import numpy as np


"""
# Function to plot histograms of evaluation metrics
def plot_histograms(metrics):
    labels = metrics.keys()
    values = metrics.values()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Evaluation Metrics Histogram")
    plt.ylim(0, 1.0)  # Assuming metric values range from 0 to 1
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
"""


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



pred_test_texts_global=[]
orig_test_texts_global = []
# Example usage of the evaluation function
for batch in testing_dataset:
    batch_images = batch["image"]
    batch_labels = batch["label"]

    # Get predictions
    preds = prediction_model.predict(batch_images)
    pred_test_texts = decode_batch_predictions(preds)
    pred_test_texts_global.extend(pred_texts)

    # Get original texts
    orig_test_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_test_texts.append(label)
    orig_test_texts_global.extend(orig_texts)  # Accumulate original texts



    # Evaluate model
#    metrics = evaluate_model(pred_texts_global, orig_texts_global)

print("pred_texts_global  ##############################################################################")

print(pred_texts_global)
print("orig_texts_global  ##############################################################################")

print(orig_texts_global)

metrics = evaluate_model(pred_test_texts_global, orig_test_texts_global)




print("DAM  ##############################################################################")


weights_sys_path = Dir_path+ "\modules\Captcha_bypass_algorithm\ocr_model_weights.h5"

weights_path ="./ocr_model_weights.h5"



import pickle
with open('OCR_model.pkl', 'wb') as f:
 pickle.dump(model, f)
print("Model saved using Pickle!")

# Just like with Pickle, you can load the Joblib-saved model and make predictions.




# this line is to increase performance by saving the teained model.  
#model.save_weights(weights_path)