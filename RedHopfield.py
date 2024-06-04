import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class RedHopfield:
    def __init__(self, num_neuronas):
        self.num_neuronas = num_neuronas
        self.pesos = np.zeros((num_neuronas, num_neuronas))
    
    def entrenar(self, patrones):
        for p in patrones:
            p = np.reshape(p, (self.num_neuronas, 1))
            self.pesos += np.dot(p, p.T)
        np.fill_diagonal(self.pesos, 0)
    
    def predecir(self, patron, pasos=5):
        patron = patron.copy()
        for _ in range(pasos):
            for i in range(self.num_neuronas):
                entrada_neta = np.dot(self.pesos[i], patron)
                patron[i] = 1 if entrada_neta > 0 else -1
        return patron

def imagen_a_patron(ruta_imagen, tamaño):
    imagen = Image.open(ruta_imagen).convert('L')
    imagen = imagen.resize(tamaño, Image.ANTIALIAS)
    imagen = np.asarray(imagen)
    imagen = np.where(imagen > 128, 1, -1)
    return imagen.flatten()

def patron_a_imagen(patron, tamaño):
    patron = np.reshape(patron, tamaño)
    imagen = np.where(patron == 1, 255, 0).astype(np.uint8)
    return Image.fromarray(imagen)

# Cargar y convertir imágenes a patrones
tamaño_imagen = (10, 10)  # Redimensionar imágenes a 10x10 píxeles

patron_1 = imagen_a_patron('imagen1.png', tamaño_imagen)
patron_2 = imagen_a_patron('imagen2.png', tamaño_imagen)

# Crear y entrenar la red de Hopfield
num_neuronas = tamaño_imagen[0] * tamaño_imagen[1]
red = RedHopfield(num_neuronas)
red.entrenar([patron_1, patron_2])

# Cargar y convertir una imagen ruidosa
patron_ruidoso = imagen_a_patron('imagen_ruidosa.png', tamaño_imagen)

# Recuperar el patrón
patron_recuperado = red.predecir(patron_ruidoso)

# Convertir el patrón recuperado a imagen y mostrarla
imagen_recuperada = patron_a_imagen(patron_recuperado, tamaño_imagen)
imagen_recuperada.show()

# Mostrar todas las imágenes para comparación
fig, ejes = plt.subplots(1, 4, figsize=(12, 3))
ejes[0].imshow(patron_a_imagen(patron_1, tamaño_imagen), cmap='gray')
ejes[0].set_title('Patrón 1')
ejes[1].imshow(patron_a_imagen(patron_2, tamaño_imagen), cmap='gray')
ejes[1].set_title('Patrón 2')
ejes[2].imshow(patron_a_imagen(patron_ruidoso, tamaño_imagen), cmap='gray')
ejes[2].set_title('Patrón Ruidoso')
ejes[3].imshow(imagen_recuperada, cmap='gray')
ejes[3].set_title('Patrón Recuperado')
plt.show()