from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_shape, num_classes):

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(256,(3,3),activation='relu',padding='same'),
        BatchNormalization(),

        GlobalAveragePooling2D(),

        Dense(256,activation='relu'),
        Dropout(0.5),

        Dense(num_classes,activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model