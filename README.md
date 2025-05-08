# epoch_core_sub
Multimodal Emotion Recognition from Audio and Transcript

Objective
Design and train a deep learning model that classifies human emotions using both:
Audio data (processed visually as spectrograms for CNN input)
Generated text transcripts (via speech-to-text for NLP modeling)

This challenge aims to apply Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and optionally Transformers, while exploring multimodal learning from real-world emotional speech data.

Dataset
RAVDESS Emotional Speech Audio
Kaggle Link 
Audio-only version of RAVDESS
1440 speech clips (male and female actors)
8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised

to import the files into colab:
from google.colab import files
files.upload()  # Upload your kaggle.json here

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
!kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
!unzip -q ravdess-emotional-speech-audio.zip




# Phases of the Task
# Phase 1 – Unimodal Pipelines
# Audio CNN:
Convert audio to spectrograms or MFCCs
Train a CNN to classify emotions from these 2D visual representations

A 3-layer CNN processes input of shape [3, 128, 256] through Conv1 (3→16), Conv2 (16→32), Conv3 (32→64) with MaxPooling after each, followed by flattening and dense layers FC1 (128 units, ReLU, dropout) and FC2 (8 output classes via CrossEntropyLoss).

# Text RNN:
Generate transcripts from the audio (e.g., using Whisper or Google Speech-to-Text)
Train an RNN (e.g., LSTM or GRU) on these transcripts to classify emotions

Embedding layer (128-dim), bidirectional GRU for temporal context, fully connected output layer for emotion classification, trained with CrossEntropyLoss and Adam optimizer (lr=1e-3).

![image](https://github.com/user-attachments/assets/e1749a8b-9a02-4e3f-bc64-33be0b541fe8)
![image](https://github.com/user-attachments/assets/2c9ad3df-0f5a-4038-8b79-55d16b5ea7a7)

I have not used this for multimodal fusion due to low accuary as the transcrpits do not have much significant differneces for the RNN to learn. 

# Phase 2 – Multimodal Fusion
Merge the visual features from the CNN and text features from the RNN
Explore different fusion strategies:
Early fusion: concatenate embeddings before classification
Late fusion: average/logit voting

I have combined mel-spectrograms (128×160) and audio features (pitch, RMS, ZCR, etc.) using a CNN with early fusion.


![image](https://github.com/user-attachments/assets/84e7628a-a33f-461f-a34c-e8e3a31cddfc)

