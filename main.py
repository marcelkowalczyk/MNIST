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
# Dane MNIST są wczytywane i normalizowane do zakresu [0, 1]
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
train_images = np.expand_dims(train_images, axis=-1)  # Dodanie wymiaru kanału
test_images = np.expand_dims(test_images, axis=-1)
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# 2. Definiowanie architektury sieci konwolucyjnej z regularyzacją
# Sieć składa się z warstw konwolucyjnych, Batch Normalization, MaxPooling oraz Dense
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout dla zapobiegania przeuczeniu
    layers.Dense(10, activation='softmax')  # Warstwa wyjściowa dla klasyfikacji cyfr
])

# 3. Kompilacja modelu i trening na danych MNIST
# Model trenowany jest przez 10 epok z użyciem Adam jako optymalizatora
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels_cat, epochs=10, batch_size=128,
                    validation_data=(test_images, test_labels_cat))

# 4. Ewaluacja modelu na zbiorze testowym
# Ocena wydajności na danych testowych i wyświetlenie dokładności
test_loss, test_accuracy = model.evaluate(test_images, test_labels_cat, verbose=0)
print(f'\nDokładność na zbiorze testowym: {test_accuracy:.4f}')

# Przewidywanie etykiet na zbiorze testowym, co pozwoli obliczyć dodatkowe metryki
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# 5. Macierz konfuzji i raport klasyfikacji
# Tworzenie macierzy konfuzji dla oceny, jak model radzi sobie z poszczególnymi klasami
confusion_mtx = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Macierz konfuzji')
plt.xlabel('Przewidywane etykiety')
plt.ylabel('Prawdziwe etykiety')
plt.show()

# Wyświetlenie szczegółowego raportu klasyfikacji: precyzja, czułość i F1-score dla każdej klasy
report = classification_report(test_labels, predicted_labels)
print("Raport klasyfikacji:\n", report)

# 6. Wizualizacja dokładności i straty podczas trenowania
# Wykresy pozwalają zobaczyć, jak model uczy się podczas każdej epoki
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

# 7. Obliczenie i wyświetlenie krzywej ROC dla każdej klasy
# Krzywa ROC pozwala na szczegółową analizę wydajności modelu dla każdej cyfry
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


# 8. Funkcja do rozpoznawania liczby z wprowadzonego obrazu
# Funkcja przetwarza obraz, przewiduje cyfrę i aktualizuje GUI
def recognize_number(image_path):
    image = Image.open(image_path).convert('L').resize((28, 28))
    image_array = 255 - np.array(image)  # Konwersja na format MNIST
    image_array = image_array.astype("float32") / 255
    image_array = np.expand_dims(np.expand_dims(image_array, axis=-1), axis=0)

    predictions = model.predict(image_array)
    predicted_number = np.argmax(predictions)

    result_label.config(text=f'Rozpoznana liczba: {predicted_number}')
    img_tk = ImageTk.PhotoImage(image.resize((150, 150)))
    image_label.config(image=img_tk)
    image_label.image = img_tk


# 9. Interfejs graficzny z tkinter
# Funkcja wybierająca obraz i wywołująca klasyfikację
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_number(file_path)


# Konfiguracja GUI
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
