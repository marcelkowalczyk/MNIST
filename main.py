# Importowanie niezbędnych bibliotek
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# 1. Wczytanie i przygotowanie danych MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
train_images = np.expand_dims(train_images, axis=-1)  # Dodanie wymiaru kanału
test_images = np.expand_dims(test_images, axis=-1)
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# 2. Augmentacja danych
data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # losowe obroty
    zoom_range=0.1,     # losowe skalowanie
    width_shift_range=0.1,  # losowe przesunięcia w poziomie
    height_shift_range=0.1  # losowe przesunięcia w pionie
)

# 3. Definiowanie architektury sieci konwolucyjnej
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Pierwsza warstwa Dense o 128 neuronach
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),   # Druga warstwa Dense o 64 neuronach
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax') # Warstwa wyjściowa dla klasyfikacji cyfr
])

# 4. Kompilacja modelu i trening z augmentacją danych
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(data_augmentation.flow(train_images, train_labels_cat, batch_size=128),
                    epochs=20, validation_data=(test_images, test_labels_cat))

# 5. Ewaluacja modelu na zbiorze testowym
test_loss, test_accuracy = model.evaluate(test_images, test_labels_cat, verbose=0)
print(f'\nDokładność na zbiorze testowym: {test_accuracy:.4f}')

# Przewidywanie etykiet na zbiorze testowym
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# 6. Macierz konfuzji i raport klasyfikacji
confusion_mtx = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Macierz konfuzji')
plt.xlabel('Przewidywane etykiety')
plt.ylabel('Prawdziwe etykiety')
plt.show()

report = classification_report(test_labels, predicted_labels)
print("Raport klasyfikacji:\n", report)

# 7. Wizualizacja dokładności i straty podczas trenowania
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Liczba epok')
plt.ylabel('Dokładność')
plt.legend()
plt.title('Dokładność modelu')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Liczba epok')
plt.ylabel('Strata')
plt.legend()
plt.title('Strata modelu')
plt.show()

# 8. Obliczenie i wyświetlenie krzywej ROC dla każdej klasy
fpr = {}
tpr = {}
roc_auc = {}

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(test_labels_cat[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f'Klasa {i} (pole = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Wskaźnik fałszywie pozytywnych')
plt.ylabel('Wskaźnik prawdziwie pozytywnych')
plt.title('Krzywe ROC dla poszczególnych klas')
plt.legend(loc='lower right')
plt.show()

# 9. Funkcja rozpoznawania liczby z dokładnym przetwarzaniem obrazu
def recognize_number(image_path):
    # Wczytanie obrazu i konwersja do skali szarości
    image = Image.open(image_path).convert('L').resize((28, 28))

    # Odwrócenie kolorów na białą cyfrę na czarnym tle (tak jak w MNIST)
    image_array = np.array(image)
    if image_array.mean() > 127:  # Jeśli tło jest jasne, a cyfry ciemne
        image_array = 255 - image_array  # Odwrócenie kolorów

    # Normalizacja do zakresu [0, 1] (przygotowanie do wprowadzenia do modelu)
    processed_image_array = image_array.astype("float32") / 255
    processed_image_array = np.expand_dims(np.expand_dims(processed_image_array, axis=-1), axis=0)

    # Wyświetlenie przetworzonego obrazu dla weryfikacji w matplotlib
    plt.imshow(processed_image_array[0, :, :, 0], cmap='gray')
    plt.title("Przetworzony obraz - biała cyfra na czarnym tle")
    plt.show()

    # Przewidywanie etykiety
    predictions = model.predict(processed_image_array)
    predicted_number = np.argmax(predictions)

    # Przygotowanie obrazu do wyświetlenia w tkinter (0-255)
    display_image_array = np.where(image_array > 128, 255, 0).astype("uint8")
    display_image = Image.fromarray(display_image_array)
    img_tk = ImageTk.PhotoImage(display_image.resize((150, 150)))

    # Aktualizacja GUI
    result_label.config(text=f'Rozpoznana liczba: {predicted_number}')
    image_label.config(image=img_tk)
    image_label.image = img_tk


# 10. Interfejs graficzny z tkinter
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_number(file_path)

root = tk.Tk()
root.title("Rozpoznawanie Liczb")
root.geometry("300x400")

btn_select = tk.Button(root, text="Wybierz obraz", command=select_file)
btn_select.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="Rozpoznana liczba: ", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
