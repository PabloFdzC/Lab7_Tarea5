import numpy as np
import os
import cv2
from  random import randint

def cargar_imagenes(carpeta, cantidad):
  imagenes = []
  nombres = []
  carpetas = []

  for archivo in os.listdir(carpeta):
    # si se llega a la cantidad determinada por el usuario antes
    # de terminar entonces hay que parar abrir las imagenes
    path = os.path.join(carpeta,archivo)
    # si se encuentra una carpeta entonces se mete a la carpeta
    # y obtiene las imágenes que hayan ahí
    if os.path.isdir(path):
      imagenes2, nombres2, carpetas2 = cargar_imagenes(path, cantidad)
      imagenes = imagenes+imagenes2
      nombres = nombres+nombres2
      carpetas = carpetas+carpetas2
    else:
      if cantidad == 0:
        break
      imagen = cv2.imread(path)
      if imagen is not None:
        nombres.append(archivo)
        imagenes.append(imagen)
        carpetas.append(carpeta)
      cantidad -= 1
      
  return imagenes, nombres, carpetas

def rotar(imagen, angulo):
  alto, ancho = imagen.shape[:2]
  M = cv2.getRotationMatrix2D((ancho//2,alto//2),angulo,1)
  return cv2.warpAffine(imagen,M,(ancho,alto))

def trasladar(imagen, alto2, ancho2):
  alto, ancho = imagen.shape[:2]
  T = np.float32([[1, 0, ancho2], [0, 1, alto2]])
  return cv2.warpAffine(imagen, T, (ancho, alto))

def escalar(imagen, alto2, ancho2):
  alto, ancho = imagen.shape[:2]
  nuevoAlto, nuevoAncho = alto+alto2, ancho +ancho2
  return cv2.resize(imagen, (nuevoAncho, nuevoAlto), interpolation= cv2.INTER_LINEAR)

# En esta función se calculan los mentos de Hu y se 
# despliegan en forma de tabla en la consola
def calcular_momentos_hu(imagen):
  _,imagen = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY)
  momentos = cv2.moments(imagen) 
  momentosHu = cv2.HuMoments(momentos)
  momentosHu = np.asarray(momentosHu).reshape(-1)
  fila = carpetas[i][len(carpetas[i])-1] + "\t"
  for h in momentosHu:
    fila += str(h) + "\t"
  print(fila)

# Se puede usar el programa usando
# python lab7.py

# La carpeta de especimenes debería tener la forma:
# Especimenes/0/especimen.jpg


carpetaImagenes = "Especimenes"
cantidadImagenes = 1
imagenes, nombres, carpetas = cargar_imagenes(carpetaImagenes, cantidadImagenes)
angulos = []
trasladosX = []
trasladosY = []
escaladosX = []
escaladosY = []
print("Momentos Hu iniciales")
for i in range(len(imagenes)):
  imagen = cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY)
  calcular_momentos_hu(imagen)
  angulos.append(randint(90, 270))
  trasladosX.append(randint(-20, 20))
  trasladosY.append(randint(-20, 20))
  escaladosX.append(randint(100, 300))
  escaladosY.append(randint(100, 300))
  
print("\nMomentos Hu imagenes rotadas")
for i in range(len(imagenes)):
  # Rotación
  imagen = cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY)
  imagenRotada = rotar(imagen, angulos[i])
  calcular_momentos_hu(imagenRotada)

print("\nMomentos Hu imagenes trasladadas")
for i in range(len(imagenes)):
  # Traslación
  imagen = cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY)
  imagenTrasladada = trasladar(imagen, trasladosX[i], trasladosY[i])
  calcular_momentos_hu(imagenTrasladada)

print("\nMomentos Hu imagenes escaladas")
for i in range(len(imagenes)):
  imagen = cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY)
  # Escalar
  imagenEscalada = escalar(imagen, escaladosX[i], escaladosY[i])
  calcular_momentos_hu(imagenEscalada)

print("\nMomentos Hu imagenes con las tres transformaciones")
for i in range(len(imagenes)):
  imagen = cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY)
  imagen2 = rotar(imagen, angulos[i])
  imagen2 = trasladar(imagen2, trasladosX[i], trasladosY[i])
  imagen2 = escalar(imagen2, escaladosX[i], escaladosY[i])
  calcular_momentos_hu(imagen2)