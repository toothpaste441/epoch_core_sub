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

Audio files are preprocessed by converting them into 128-band mel spectrograms of fixed width (256 frames), normalized, and padded or cropped as needed to ensure uniform input shape. The CNN model comprises five convolutional blocks with batch normalization and ReLU activation, followed by max pooling to progressively reduce spatial dimensions. After flattening, the features are passed through two fully connected layers with dropout for regularization, ending in an output layer that predicts one of the eight emotion classes. The model is trained using the Adam optimizer with a learning rate of 0.003 and CrossEntropyLoss as the criterion. Data is split into 80% training and 20% validation sets, and training runs for 50 epochs. While training accuracy improves steadily, reaching around 80%, validation accuracy fluctuates between 30% and 67%, likely due to small validation size, class imbalance, or overfitting. The absence of techniques like early stopping and learning rate scheduling may also contribute to the instability. Despite these issues, the model shows clear learning progress. Improvements such as stratified sampling, data augmentation, class-weighted loss, early stopping, and learning rate scheduling are recommended to enhance generalization and stabilize validation performance.

training and validation accuracy:

![image](https://github.com/user-attachments/assets/707d830e-1093-464c-a9bd-3c1b038b7ddb)


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

