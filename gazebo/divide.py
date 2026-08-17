import os
from PIL import Image

# ¡Crucial! Desactivamos el límite de seguridad de píxeles de Pillow.
# Tu imagen tiene más de 500 millones de píxeles, lo que Pillow detectaría como un ataque de memoria.
Image.MAX_IMAGE_PIXELS = None

def dividir_imagen(ruta_imagen, tamaño_cuadricula=8192):
    print(f"Abriendo la imagen gigante: {ruta_imagen}...")
    try:
        img = Image.open(ruta_imagen)
    except FileNotFoundError:
        print("Error: No se encontró la imagen. Asegúrate de que 'terrain.png' esté en esta carpeta.")
        return

    ancho, alto = img.size
    print(f"Resolución original: {ancho}x{alto} px")
    print(f"Cortando en mosaicos de {tamaño_cuadricula}x{tamaño_cuadricula} px...")

    # Crear una carpeta para guardar los pedazos limpios
    carpeta_salida = "tiles_terreno"
    os.makedirs(carpeta_salida, exist_ok=True)

    # Calcular cuántas filas y columnas necesitamos
    columnas = (ancho + tamaño_cuadricula - 1) // tamaño_cuadricula
    filas = (alto + tamaño_cuadricula - 1) // tamaño_cuadricula

    contador = 1
    
    # Recorrer la imagen y recortar
    for fila in range(filas):
        for col in range(columnas):
            # Coordenadas de la caja de recorte (izquierda, superior, derecha, inferior)
            izq = col * tamaño_cuadricula
            sup = fila * tamaño_cuadricula
            der = min(izq + tamaño_cuadricula, ancho) # 'min' evita salirnos de los bordes
            inf = min(sup + tamaño_cuadricula, alto)
            
            print(f"Procesando Cuadrante {contador}: X({izq} a {der}), Y({sup} a {inf})")
            
            # Recortar y guardar
            pedazo = img.crop((izq, sup, der, inf))
            nombre_archivo = f"{carpeta_salida}/terreno_fila{fila}_col{col}.png"
            pedazo.save(nombre_archivo)
            
            contador += 1

    print(f"\n¡Proceso completado! Se han guardado {contador - 1} imágenes en la carpeta '{carpeta_salida}'.")

# Ejecutar la función apuntando a tu imagen
dividir_imagen("terrain.png")
