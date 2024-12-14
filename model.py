import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, ReLU, MaxPooling2D, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Verificar se a GPU está disponível
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Treinando no dispositivo: {device_name}")

with tf.device(device_name):
    # Configurações
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 128
    EPOCHS = 150
    NUM_CLASSES = 2  # Alterado para 2 classes (benign e malignant)
    LEARNING_RATE = 0.0001
    TRAIN_DIR = "train"  # Caminho para a pasta de treino
    TEST_DIR = "test"  # Caminho para a pasta de teste

    # Modelo baseado na arquitetura da imagem
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Primeiras 130 camadas da ResNet50 (transfer learning)
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
    for layer in base_model.layers[:130]:
        layer.trainable = False

    # Camadas personalizadas
    x = base_model.output
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    feature_extractor_model = Model(inputs=input_layer, outputs=x)  # Modelo para extrair features

    # Finalizar modelo com camadas densas
    x = Dense(1024, activation='relu')(x)
    x = Dense(NUM_CLASSES)(x)
    predictions = Softmax()(x)

    # Construir o modelo final
    model = Model(inputs=input_layer, outputs=predictions)

    # Compilar o modelo
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Geradores de dados para treinamento e validação
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Não embaralhar para alinhar previsões e rótulos
    )

    # Treinar o modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Salvar o modelo treinado
    model.save("reshist_model.h5")
    print("Treinamento concluído e modelo salvo.")

    # Extração de features para KNN
    print("\nExtraindo features para KNN...")
    train_features = feature_extractor_model.predict(train_generator)
    val_features = feature_extractor_model.predict(val_generator)

    y_train = train_generator.classes
    y_val = val_generator.classes

    # Treinar o KNN com as features extraídas
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, y_train)

    # Avaliar o KNN
    print("\nAvaliando o KNN...")
    knn_predictions = knn.predict(val_features)

    knn_accuracy = accuracy_score(y_val, knn_predictions)
    knn_precision = precision_score(y_val, knn_predictions, average='weighted')
    knn_recall = recall_score(y_val, knn_predictions, average='weighted')
    knn_f1 = f1_score(y_val, knn_predictions, average='weighted')

    print("\nMétricas de Avaliação do KNN:")
    print(f"Acurácia: {knn_accuracy:.4f}")
    print(f"Precisão: {knn_precision:.4f}")
    print(f"Recall: {knn_recall:.4f}")
    print(f"F1-Score: {knn_f1:.4f}")

    print("\nRelatório de Classificação do KNN:")
    print(classification_report(y_val, knn_predictions, target_names=list(val_generator.class_indices.keys())))

    # Avaliação do modelo CNN
    print("\nAvaliando o modelo CNN...")
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    print("\nMétricas de Avaliação do Modelo CNN:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nRelatório de Classificação do Modelo CNN:")
    print(classification_report(y_val, y_pred, target_names=list(val_generator.class_indices.keys())))