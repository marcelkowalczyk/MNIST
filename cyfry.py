# Importowanie niezbędnych bibliotek

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# 1. Zaimportowanie zbioru danych MNIST
# Zestaw danych MNIST jest dostępny w bibliotekach TensorFlow i Keras, co umożliwia łatwe załadowanie
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. Przygotowanie danych: Normalizacja i przygotowanie danych wejściowych
# Normalizacja wartości pikseli (z zakresu [0, 255] do [0, 1])
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Rozszerzenie wymiarów danych wejściowych do formatu (28, 28, 1) dla sieci konwolucyjnych
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# One-hot encoding etykiet
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# 3. Definiowanie architektury sieci neuronowej
# Tworzymy prostą sieć konwolucyjną z warstwami Conv2D, MaxPooling2D oraz Dense

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. Kompilacja modelu
# Używamy 'adam' jako optymalizatora i 'categorical_crossentropy' jako funkcji straty
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()

# 5. Trenowanie modelu
# Trenujemy model na danych treningowych, walidując na danych testowych
history = model.fit(train_images, train_labels_cat, epochs=10, batch_size=128, validation_data=(test_images, test_labels_cat))

# 6. Ocena modelu
# Obliczamy dokładność klasyfikacji na danych testowych
test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')

# Przewidywanie etykiet dla danych testowych
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Macierz konfuzji
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Wyświetlenie szczegółowego raportu klasyfikacji
report = classification_report(test_labels, predicted_labels)
print("Classification Report:\n", report)

# 7. Optymalizacja modelu
# Można eksperymentować z różnymi architekturami i technikami optymalizacji (np. Dropout, Batch Normalization)
# oraz innymi hiperparametrami. Przykładem jest dodanie dodatkowych warstw lub zmiana współczynnika uczenia.

# 8. Wizualizacja wyników
# Wizualizacja dokładności i straty podczas trenowania
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 9. Obliczenie krzywej ROC dla każdej klasy
fpr = {}
tpr = {}
roc_auc = {}

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(test_labels_cat[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Wyświetlenie krzywych ROC
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Podsumowanie
# Model osiąga wysoką dokładność klasyfikacji na zbiorze testowym.
# Eksperymenty z różnymi architekturami i hiperparametrami mogą dodatkowo poprawić wyniki.
