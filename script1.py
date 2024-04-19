import cv2
import tensorflow as tf
import numpy as np
from typing import List
import os

# Define your vocabulary for character-to-integer mapping
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Define StringLookup layers for character-to-integer and integer-to-character mapping
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=vocab,
    oov_token="",
    invert=False,
    output_mode="int"
)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True,
)

# Load your LipNet model
model = tf.keras.models.load_model('Lip_reader_40/Models/my_lip_model_40.keras')

# Define the preprocessing function
def preprocess_frame(frame):
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Define the center of the frame
    center_x, center_y = frame_gray.shape[1] // 2, frame_gray.shape[0] // 2
    # Define the size of the cropped area
    crop_size = 100  # Adjust as needed
    # Crop the frame around the center
    cropped_frame = frame_gray[center_y - crop_size // 2: center_y + crop_size // 2,
                               center_x - crop_size // 2: center_x + crop_size // 2]
    # Resize the cropped frame to the desired dimensions (140x46)
    resized_frame = cv2.resize(cropped_frame, (140, 46))
    # Convert the resized frame to a float32 tensor
    processed_frame = tf.convert_to_tensor(resized_frame, dtype=tf.float32)
    # Normalize the frame
    mean = tf.reduce_mean(processed_frame)
    std = tf.math.reduce_std(processed_frame)
    normalized_frame = (processed_frame - mean) / std
    return normalized_frame

# Define function to load video
def load_video(path:str) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        frames.append(frame)
    cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# Define function to load data
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    # File name splitting for both Unix and Windows
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

# Define function to load alignments
def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Initialize the video capture from the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Initialize variables to accumulate frames
frames_sequence = []

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break
    
    processed_frame = preprocess_frame(frame)
    
    # Draw a rectangle around the lips area
    cv2.rectangle(frame, (frame.shape[1] // 2 - 50, frame.shape[0] // 2 - 50),
                  (frame.shape[1] // 2 + 50, frame.shape[0] // 2 + 50), (0, 255, 0), 2)
    
    # Accumulate frames until we have enough to form a sequence of length 75
    if len(frames_sequence) < 75:
        frames_sequence.append(processed_frame)
    else:
        # Remove the oldest frame and add the new frame
        frames_sequence.pop(0)
        frames_sequence.append(processed_frame)
        
        # Convert frames sequence to a tensor of shape (1, 75, 46, 140, 1)
        frames_tensor = tf.expand_dims(tf.stack(frames_sequence), axis=0)
        
        # Perform inference on the frames sequence using the model
        yhat = model.predict(frames_tensor)
        
        # Decode the predictions (if needed) and display the results
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        print('~'*100, 'PREDICTIONS')
        print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded])
        
        # Reset the frames sequence
        frames_sequence = []
    
    # Display the frame
    cv2.imshow('Processed Frame', frame)
    
    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
