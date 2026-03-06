import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from features.extract_logmel import extract_log_mel
from model.cnn_model import create_model

# Labels
labels = ["yes","no","stop","hello","how","are","you","he","is","good","right","down","left","up"]
label_map = {label:i for i,label in enumerate(labels)}

X = []
y = []

# Load dataset
for label in labels:

    folder = f"dataset/{label}"

    for file in os.listdir(folder):

        path = os.path.join(folder,file)

        features = extract_log_mel(path)

        X.append(features)
        y.append(label_map[label])

# Convert to numpy
X = np.array(X)   # ❌ DO NOT add newaxis
y = np.array(y)

print("Dataset shape:", X.shape)

# One hot encoding
y = to_categorical(y,num_classes=len(labels))

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = create_model(X_train.shape[1:],len(labels))

# Train
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test,y_test)
)

# Save model
model.save("speech_cnn_model.h5")

print("✅ Model saved successfully")