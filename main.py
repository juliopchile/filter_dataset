import os
import glob
import shutil
import zipfile
import random
import xml.etree.ElementTree as ET
import cv2
import numba
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
from collections.abc import Callable
from typing import List, Dict, Tuple, Optional, Literal
from timeit import default_timer as timer
#np.set_printoptions(suppress=True)

SALMON = 0
MEDIUM = 1
SMALL = 2
TINY = 3
FUZZY = 4
CLASSES = ["salmon", "medium", "small", "tiny", "fuzzy"]  # index 0 -> 'salmon'
#CLASSES = ["salmon"]

RATIOS = {'train': 1, 'valid': 0, 'test': 0}
SEED = 201721022

# ?
# ? Métricas de las máscaras
# ?
def calc_area(polygon: NDArray[np.int32]) -> float:
    """Calcula el área de un polígono usando el método de contorno de OpenCV.
    
    :param NDArray[np.int32] polygon: Arreglo de forma (n, 1, 2) con coordenadas (x, y) en píxeles.
    :return float: Área del polígono en píxeles cuadrados.
    """
    return cv2.contourArea(polygon)

def calc_perimetro(polygon: NDArray[np.int32]) -> float:
    """Calcula el perímetro del polígono usando la longitud de arco de OpenCV.

    :param NDArray[np.int32] polygon: Arreglo de forma (n, 1, 2) con coordenadas (x, y) en píxeles.
    :return float: Perímetro del polígono.
    """
    return cv2.arcLength(polygon, closed=True)

def calc_aspect_ratio(polygon: NDArray[np.int32], dimensions: Optional[Tuple[float, float]] = None) -> float:
    """Calcula la relación de aspecto del rectángulo mínimo que contiene a un polígono.

    :param NDArray[np.int32] polygon: Arreglo de forma (n, 1, 2) con coordenadas (x, y) en píxeles.
    :param Optional[Tuple[float, float]] dimensions: Ancho y alto del rectángulo mínimo. Opcional, se calcula si no se proporciona.
    :return float: Relación de aspecto.
    """
    (center_r, (dimensions), angle_r) = cv2.minAreaRect(polygon) if dimensions is None else (0, (dimensions), 0)
    ratio = dimensions[0] / dimensions[1] if dimensions[1] != 0 else 0.0
    return ratio if ratio <= 1 else 1 / ratio

def calc_rectangularidad(polygon: NDArray[np.int32], area: Optional[float] = None, dimensions: Optional[Tuple[float, float]] = None) -> float:
    """Calcula la rectangularidad como la razón entre el área del polígono y su rectángulo mínimo.

    :param NDArray[np.int32] polygon: Arreglo de forma (n, 1, 2) con coordenadas (x, y) en píxeles.
    :param Optional[float] area: Área del polígono en píxeles cuadrados. Opcional, se calcula si no se proporciona.
    :param Optional[Tuple[float, float]] dimensions: Ancho y alto del rectángulo mínimo. Opcional, se calcula si no se proporciona.
    :return: Rectangularidad del polígono (adimensional).
    :rtype: float
    """
    area = calc_area(polygon) if area is None else area
    (center_r, (dimensions), angle_r) = cv2.minAreaRect(polygon) if dimensions is None else (0, (dimensions), 0)
    rect_area = dimensions[0] * dimensions[1]
    return area / rect_area if rect_area != 0 else 0.0

def calc_excentricidad(polygon: NDArray[np.int32]) -> float:
    """Calcula la excentricidad ajustando una elipse al polígono.

    :param NDArray[np.int32] polygon: Arreglo de forma (n, 1, 2) con coordenadas (x, y) en píxeles.
    :return float: Excentricidad de la elipse ajustada.
    """
    eccentricity = -1.0  # Valor por defecto para error
    if len(polygon) >= 6:
        try:
            ellipse = cv2.fitEllipse(polygon)
            (center_e, (ma, MA), angle_e) = ellipse
            if MA > 0 and ma <= MA:
                eccentricity = np.sqrt(1 - (ma / MA) ** 2)
        except Exception as e:
            print(f"Error en cv2.fitEllipse para polígono con {len(polygon)} puntos: {str(e)}")
    return eccentricity

def calc_elongation(polygon: NDArray[np.int32]) -> float:
    """Calcula la elongación del polígono usando sus momentos.

    La elongación mide qué tan estirada es la forma del polígono, con valores cercanos a 0 para formas circulares
    y cercanos a 1 para formas alargadas. Usa los valores propios de la matriz de covarianza.

    :param NDArray[np.int32] polygon: polygon: Arreglo de forma (n, 1, 2) con coordenadas (x, y) en píxeles.
    :return float: Elongación del polígono (adimensional, entre 0 y 1). Devuelve 1.0 si el área es cero o la forma es degenerada.
    """
    moments = cv2.moments(polygon)
    return compute_elongation_from_moments(moments['mu20'], moments['mu02'], moments['mu11'], moments['m00'])

@numba.jit(nopython=True)
def compute_elongation_from_moments(mu20: float, mu02: float, mu11: float, m00: float) -> float:
    """Calcula la elongación de un polígono a partir de sus momentos centrales normalizados.

    :param float mu20: Segundo momento central respecto al eje x.
    :param float mu02: Segundo momento central respecto al eje y.
    :param float mu11: Momento central cruzado (covarianza entre x e y).
    :param float m00: Momento de orden cero (área del polígono).
    :return float: Elongación del polígono (adimensional, entre 0 y 1). Devuelve 1.0 si el área es cero o la forma es degenerada.
    """
    if m00 == 0:
        return 1.0
    mu20 /= m00
    mu02 /= m00
    mu11 /= m00
    term1 = mu20 + mu02
    term2 = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    lambda1 = term1 + term2
    lambda2 = term1 - term2
    e = lambda1 / lambda2 if lambda2 != 0 else float('inf')
    return (e - 1) / (e + 1)

@numba.jit(nopython=True)
def calc_circularidad(area: float, perimetro: float) -> float:
    """Calcula la circularidad de un polígono como 4π veces la compacidad.

    La circularidad mide qué tan parecida es la forma del polígono a un círculo, con un valor de 1.0 para un círculo
    perfecto y valores menores para formas irregulares o alargadas.

    :param float area: Área del polígono.
    :param float perimetro: Perímetro del polígono
    :return float: Circularidad del polígono.
    """
    return 4 * np.pi * calc_compacidad(area, perimetro)

@numba.jit(nopython=True)
def calc_compacidad(area: float, perimetro: float) -> float:
    """Calcula la compacidad de un polígono como el área dividida por el cuadrado del perímetro.

    La compacidad mide la eficiencia de la forma del polígono, con valores más altos para formas que maximizan
    el área para un perímetro dado. Un círculo tiene la máxima compacidad (1/(4π) ≈ 0.0796), mientras que
    formas irregulares o alargadas tienen valores menores.

    :param float area: Área del polígono.
    :param float perimetro: Perímetro del polígono
    :return float: Circularidad del polígono.
    """
    return area / (perimetro ** 2) if perimetro != 0 else 0.0

def calc_nitidez(polygon: NDArray[np.int32], gradientes: Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]] = None,
                ksize: float = 5, peso_borde: float = 0.5) -> float:
    """Calcula la nitidez de un una máscara de segmentación en una imagen, combinando borde e interior.

    - Nitidez de borde (edge sharpness):  
      Extrae una banda alrededor del contorno (dilatación con kernel de tamaño ksize) y promedia la magnitud de gradiente Sobel allí.
    - Nitidez de interior (interior sharpness):  
      Llena el polígono y calcula la varianza del Laplaciano dentro.

    La puntuación final = peso_borde * nitidez_borde + (1-peso_borde) * nitidez_interior.

    :param NDArray[np.int32] polygon: Polígono de forma (n, 1, 2) con las coordenadas del contorno.
    :param Tuple[NDArray[np.float32], NDArray[np.float32]] gradientes: Gradientes Sobel y Laplaciano (escalados) de la imagen.
    :param float peso_borde: En [0,1], peso de la nitidez de borde.
    :param int ksize: Tamaño del kernel elíptico para dilatación de borde.
    :return float: Nitidez ponderada del polígono.
    """
    if gradientes is None:
        return 0.0
    (grad_mag, laplaciano) = gradientes

    # 1. Nitidez del borde
    mascara_contorno = np.zeros_like(grad_mag, dtype=np.uint8)
    cv2.drawContours(mascara_contorno, [polygon], -1, 255, thickness=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    banda_contorno = cv2.dilate(mascara_contorno, kernel, iterations=1)
    pixeles_banda = grad_mag[banda_contorno == 255]
    nitidez_borde = np.mean(pixeles_banda)

    # 2. Nitidez interna
    mascara_interior = np.zeros_like(laplaciano, dtype=np.uint8)
    cv2.fillPoly(mascara_interior, [polygon], 255)
    pixeles_interior = laplaciano[mascara_interior == 255]
    nitidez_interior = np.std(pixeles_interior)

    # 3. Combinar ambas métricas
    return np.log1p(np.power(nitidez_borde, peso_borde) * np.power(nitidez_interior, (1 - peso_borde)))
    #gradientes = (nitidez_borde, nitidez_interior)
    #return gradientes


#*
#* Funciones usadas para calcular la escala de kernels para normalizarlos (no usado casi)
#*
@numba.jit(nopython=True)
def calcular_escala_kernel(kernel: NDArray) -> float:
    """Calcula el factor de escala para normalizar un kernel.

    - Aplana el kernel y suma por separado valores positivos y negativos.
    - Determina el mayor valor absoluto entre ambos sumatorios.
    - Devuelve 1 / max_abs para escalar el kernel a rango [-1, 1].

    :param NDArray[np.float32] kernel: matriz de coeficientes del filtro.
    :return float: factor de escala inverso al máximo absoluto de coeficientes.
    """
    coeficientes = kernel.ravel()
    suma_positivos = 0.0
    suma_negativos = 0.0
    for coef in coeficientes:
        if coef > 0:
            suma_positivos += coef
        elif coef < 0:
            suma_negativos += coef
    valor_maximo = suma_positivos
    valor_minimo = suma_negativos
    max_abs = max(abs(valor_maximo), abs(valor_minimo))
    return 1 / max_abs

@numba.jit(nopython=True)
def create_impulse_matrix(klength: int) -> Tuple[NDArray[np.float32], int]:
    """Genera una matriz cuadrada de tamaño (klength+2) con centro igual a 1.0.

    El tamaño de la matriz asegura poder convolucionarlo con un kernel luego.

    :param int klength: longitud del kernel deseado.
    :return Tuple[NDArray[np.float32], int]: imagen impulso y coordenada central.
    """
    size = klength + 2
    impulse = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    impulse[center, center] = 1.0
    return impulse, center

@numba.jit(nopython=True)
def extract_kernel(image: MatLike, klength: int, center: int) -> NDArray[np.float32]:
    """Extrae un sub-bloque cuadrado de tamaño klength centrado en 'center'.

    :param MatLike image: imagen de entrada.
    :param int klength: tamaño del bloque a extraer.
    :param int center: índice central de la imagen.
    :return NDArray[np.float32]: bloque de forma (klength, klength).
    """
    half_size = klength // 2
    start = center - half_size
    end = start + klength
    return image[start:end, start:end]

def get_filter_kernel(filter_fn: Callable[[MatLike, int, Optional[int]], MatLike], ksize: int, default_ksize: int = 3):
    """Obtiene el kernel de un filtro aplicando la respuesta a una imagen impulso.

    :param Callable filter_fn: función que aplica el filtro (image, ksize) -> image.
    :param int ksize: tamaño del kernel o -1/None para usar default_ksize.
    :param int default_ksize: tamaño por defecto si ksize es -1 o None.
    :return NDArray[np.float32]: kernel resultante extraído de la respuesta.
    """
    klength = default_ksize if ksize in (-1, None) else ksize
    impulse, center = create_impulse_matrix(klength)
    filtered_image = filter_fn(impulse, ksize)
    return extract_kernel(filtered_image, klength, center)

def get_sobel_kernel(ksize: int = -1) -> NDArray[np.float32]:
    """Devuelve el kernel Sobel horizontal.

    :param int ksize: Tamaño del Sobel (pasar -1 para FILTER_SCHARR).
    :return NDArray[np.float32]: Kernel Sobel de derivada en x.
    """
    def sobel_fn(image: MatLike, ksize: int):
        return cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    return get_filter_kernel(sobel_fn, ksize)

def get_laplacian_kernel(ksize: int = None) -> NDArray[np.float32]:
    """Devuelve el kernel Laplaciano.

    :param int or None ksize: Tamaño del filtro; None usa el valor por defecto de OpenCV.
    :return NDArray[np.float32]: Kernel del operador Laplaciano.
    """
    def laplacian_fn(image: MatLike, ksize: int):
        return cv2.Laplacian(image, cv2.CV_32F) if ksize is None else cv2.Laplacian(image, cv2.CV_32F, ksize=ksize)
    return get_filter_kernel(laplacian_fn, ksize)


# ?
# ? Funciones para el manejo del filtrado de un dataset
# ?
def load_yolo_segmentation(label_path: str) -> List[Dict[str, int | NDArray[np.float32]]]:
    """Lee un archivo de etiquetas YOLO de segmentación y retorna las etiquetas en una lista de diccionarios.

    Cada línea tiene el formato:
      class_id x1 y1 x2 y2 ... xN yN
    donde xi, yi están normalizados en (0 : 1).

    :param str label_path: Dirección del archivo con las etiquetas YOLO a cargar.
    :return List[Dict[str, int | NDArray[np.float32]]]: Retorna una lista de diccionarios:
    [{'class_id': int, 'points': NDArray[np.float32] de forma (N, 2)}, ...]
    """
    segmentations = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            # Extraer coordenadas como lista de floats
            coords = [float(x) for x in parts[1:]]
            # Convertir a array de NumPy con forma (N, 2)
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            segmentations.append({'class_id': class_id, 'points': points})
    return segmentations

def save_yolo_segmentation(segmentations: List[Dict[str, int | NDArray[np.float32]]], label_path: str) -> None:
    """
    Guarda una lista de segmentaciones en un archivo de texto con formato YOLO.
    Cada entrada en 'segmentations' debe ser un dict con:
      - 'class_id': int
      - 'points': NDArray[np.float32] de forma (N, 2) con coordenadas (x, y)

    El archivo resultante tendrá líneas:
      class_id x1 y1 x2 y2 ... xN yN
    donde xi, yi son floats normalizados (0.0 a 1.0) con 6 decimales.
    """
    with open(label_path, 'w') as file:
        for seg in segmentations:
            class_id = seg['class_id']
            points: NDArray[np.float32] = seg['points']  # Array de forma (N, 2)
            # Convertir el array a una lista plana de coordenadas [x1, y1, x2, y2, ...]
            coords = [f"{coord}" for coord in points.flatten()]
            # Construir la línea: clase seguida de coordenadas
            line = " ".join([str(class_id)] + coords)
            file.write(line + "\n")

@numba.jit(nopython=True)
def scale_polygon(points: NDArray[np.float32], img_width: int, img_height: int) -> NDArray[np.int32]:
    """ Escala un polígono con coordenadas normalizadas a píxeles.

    :param NDArray[np.float32] points: Coordenadas de un polígono de forma (N, 2) con coordenadas normalizadas (x, y)
    :param int img_width: Ancho de la imagen en píxeles.
    :param int img_height: Alto de la imagen en píxeles.
    :raises ValueError: Si el arreglo 'points' no tiene la forma adecuada
    :return NDArray[np.int32]: Arreglo de forma (N, 1, 2) con coordenadas escaladas (x, y) en píxeles.
    """
    # Validar la entrada (Numba no usa excepciones complejas, validamos antes)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("El array 'points' debe tener forma (N, 2)")

    # Crear array de salida con tipo int32 y forma (N, 2)
    n_points = points.shape[0]
    scaled_points = np.empty((n_points, 2), dtype=np.int32)

    # Escalar coordenadas directamente
    for i in range(n_points):
        scaled_points[i, 0] = int(points[i, 0] * img_width)  # Escalar x
        scaled_points[i, 1] = int(points[i, 1] * img_height) # Escalar y

    # Redimensionar a (N, 1, 2) para compatibilidad con OpenCV
    return scaled_points.reshape(n_points, 1, 2)

def calc_metrics_single_polygon(polygon: NDArray[np.int32], metrics_list: List[str], peso_borde: float = 0.5, ksize_band: int = None,
                                gradientes: Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]] = None) -> Dict[str, float]:
    """Esta función realiza el cálculo de métricas para un único polígono.

    :param NDArray[np.int32] polygon: Polígono de forma (N, 1, 2) con coordenadas escaladas (x, y).
    :param List[str] metrics_list: Lista con los nombres de las métricas a calcular.
    :param Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]] gradientes: Tupla de gradientes de Sobel y Laplaciano, defaults to None
    :raises ValueError: Si el polígono no tiene la forma adecuada.
    :raises ValueError: Si hay una métrica inválida.
    :return Dict[str, float]: Diccionario con las métricas calculadas.
    """
    # Validar forma del polígono
    if polygon.ndim != 3 or polygon.shape[1] != 1 or polygon.shape[2] != 2:
        raise ValueError("El polígono debe tener forma (n, 1, 2)")

    # Definir mapeo de métricas a funciones
    metric_functions = {
        "area": lambda: calc_area(polygon),
        "perimeter": lambda: calc_perimetro(polygon),
        "aspect_ratio": lambda: calc_aspect_ratio(polygon),
        "rectangularity": lambda: calc_rectangularidad(polygon),
        "eccentricity": lambda: calc_excentricidad(polygon),
        "elongation": lambda: calc_elongation(polygon),
        "circularity": lambda: calc_circularidad(
            results.get("area", calc_area(polygon)),
            results.get("perimeter", calc_perimetro(polygon))
        ),
        "nitidez": lambda: calc_nitidez(polygon, gradientes, ksize_band, peso_borde)
    }

    # Validar métricas solicitadas
    invalid_metrics = set(metrics_list) - set(metric_functions.keys())
    if invalid_metrics:
        raise ValueError(f"Métricas no válidas: {invalid_metrics}")

    # Calcular solo las métricas solicitadas
    results: Dict[str, float] = {}
    for metric in metrics_list:
        results[metric] = metric_functions[metric]()

    return results

def desicion_making(metricas: Dict[str, float]) -> Literal[3, 1, 2, 0]:
    """Esta función recibe las métricas calculadas para un polígono y decide a que clase pertenecerá.
    
    Modificar a gusto.

    :param Dict[str, float] metricas: Diccionario de métricas calculadas.
    :return int: La clase a la que pertenecerá la etiqueta
    """
    # Calcular métricas
    area = metricas["area"]
    perimeter = metricas["perimeter"]
    #aspect_ratio = metricas["aspect_ratio"]
    #nitidez = metricas["nitidez"]

    #if nitidez < 1.2:
    #    return FUZZY
    # Área minuscula
    if area <= 1700:
        #if nitidez >= 1.5 and perimeter >= 255:
        if perimeter >= 255:
            return SMALL
        else:
            return TINY
    # Área pequeña
    elif 1700 < area <= 4200:
        #if nitidez >= 1.65 and perimeter >= 400:
        if perimeter >= 425:
            return MEDIUM
        else:
            return SMALL
    # Área mediana
    elif 4200 < area <= 7500:   
        return MEDIUM
    # Ariana grande
    else:
        return SALMON

def procesar_poligonos(segmentations: List[Dict[str, int | NDArray[np.float32]]], img_height: int, img_width: int, peso_borde: float = 0.5,
                       image: MatLike = None, ksizes: Tuple[int, int, int] = (-1, None, 5), include: bool = False) -> List[Dict[str, int | NDArray[np.float32]]]:
    """Esta función procesa las etiquetas de segmentación de instancias de una imagen dada y determina si filtrarlas o no del dataset.

    :param List[Dict[str, int  |  NDArray[np.float32]]] segmentations: Lista de diccionarios con las etiquetas.
    Tiene formato [{'class_id': int, 'points': NDArray[np.float32] de forma (N, 2)}, ...].
    :param int img_height: Altura de la imagen.
    :param int img_width: Ancho de la imagen.
    :param MatLike image: Imagen, se utiliza para obtener la nitidez, defaults to None.
    :param bool include: Parámetro para decidir si excluir imagenes o agregarlas con otra clase, defaults to False.
    :return List[Dict[str, int | NDArray[np.float32]]]: Lista de diccionarios con las etiquetas filtradas.
    Tiene formato [{'class_id': int, 'points': NDArray[np.float32] de forma (N, 2)}, ...].
    """
    nuevos_poligonos = []
    #metric_name_list = ["area", "perimeter", "aspect_ratio", "nitidez"]
    metric_name_list = ["area", "perimeter"]

    # En caso de querer calcularse la nitidez, es necesario pasar la imagen original.
    if image is not None:
        # Calcular coeficiente de escalado para el Sobel
        ksize_sobel, ksize_lap, ksize_band = ksizes
        sob_kernel = get_sobel_kernel(ksize_sobel)
        sob_factor = calcular_escala_kernel(sob_kernel)
        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calcular gradiente y laplaciano de la imagen
        gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=ksize_sobel, scale=sob_factor)
        gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=ksize_sobel, scale=sob_factor)
        grad_mag = np.sqrt(gx**2 + gy**2)
        laplaciano = cv2.Laplacian(gris, cv2.CV_32F) if ksize_lap is None else cv2.Laplacian(gris, cv2.CV_32F, ksize_lap)
        laplaciano_mag = laplaciano # np.log1p(np.abs(laplaciano))
        gradientes = grad_mag, laplaciano_mag
    else:
        gradientes = None
        ksize_band = None

    # Iterar por cada etiqueta de este archivo de etiquetas.
    for seg in segmentations:
        points = seg['points']
        polygon = scale_polygon(points, img_width, img_height)
        metricas = calc_metrics_single_polygon(polygon, metric_name_list, peso_borde, ksize_band, gradientes)
        score = desicion_making(metricas)

        #if score in (MEDIUM, SMALL) and seg['class_id'] == FUZZY:
        #    score = FUZZY

        # Decidir si añadir casos como otra clase o simplemente filtrarlos
        if include:
            nuevos_poligonos.append({'class_id': score, 'points': points})
        else:
            if score == SALMON or score == MEDIUM:
                nuevos_poligonos.append(seg)

    return nuevos_poligonos     

def filtrar_dataset(labels_path: str, filtered_path: str, images_path: str, use_nitidez=True, include=False) -> None:
    """Esta función filtra un dataset de segmentación de instancias en formato YOLO.
    
    Se utilizan métricas y componentes de las distintas etiquetas que se encuentran en el dataset (segmentaciones)
    y con esto se decide de que forma asignar clases o filtrar etiquetas.

    :param str labels_path: Directorio del dataset YOLO. Este debería tener subdirectorios ["test", "train", "valid"].
    :param str filtered_path: Directorio donde guardar las etiquetas filtradas o modificadas.
    :param str images_path: Directorio donde se encuentran las imagenes del dataset. Necesario para calcular nitidez.
    :param bool use_nitidez: Define si se calculará o no la nitidez de las segmentaciones, defaults to True.
    :param bool include: Define si se incluiran las etiquetas filtradas con otras clases, defaults to False.
    """
    subfolders = ["test", "train", "valid"]
    # Se usan siempre estas dimensiones para calcular las métricas, ya que se quieren tresholds "normalizados" a 640x640
    img_height, img_width = 640, 640
    ksizes = (-1, None, 5)
    peso_borde = 0.15

    for sub in subfolders:
        label_subdir = os.path.join(labels_path, sub)
        image_subdir = os.path.join(images_path, sub)
        filtered_subdir = os.path.join(filtered_path, sub)

        # Crear la carpeta de destino si no existe
        os.makedirs(filtered_subdir, exist_ok=True)

        # Obtener lista de labels
        label_files = [f for f in os.listdir(label_subdir) if f.endswith(".txt")]
        for label_file in label_files:
            label_path = os.path.join(label_subdir, os.path.splitext(label_file)[0] + ".txt")
            image_path = os.path.join(image_subdir, os.path.splitext(label_file)[0] + ".jpg")

            # Saltarse imagenes sin etiquetas
            if not os.path.exists(label_path):
                continue

            # Leer las etiquetas del archivo
            segmentations = load_yolo_segmentation(label_path)
            if segmentations is None:
                print(f"Error: No se pudieron leer las etiquetas del archivo {label_path}")
                continue

            # Cargar la imagen si es necesaria.
            if use_nitidez:
                image = cv2.imread(image_path)
                # img_height, img_width = image.shape[:2]
            else:
                image = None

            # Filtrar las etiquetas
            filtered_labels = procesar_poligonos(segmentations, img_height, img_width, peso_borde, image, ksizes, include)

            # Guardar las nuevas etiquetas filtradas
            if len(filtered_labels) > 0:
                filtered_label_path = os.path.join(filtered_subdir, os.path.splitext(label_file)[0] + ".txt")
                save_yolo_segmentation(filtered_labels, filtered_label_path)


# ?
# ? Crear el archivo ZIP que puede importarse a CVAT
# ?
def folder_to_zip(folder_path: str) -> None:
    """Comprime una carpeta en un archivo ZIP, lo mueve al directorio padre y elimina la carpeta original.

    El archivo ZIP tendrá el mismo nombre que la carpeta, con extensión `.zip`, y contendrá todos los archivos
    y subcarpetas de la carpeta original, preservando su estructura interna.

    :param str folder_path: Ruta absoluta o relativa de la carpeta a comprimir.
    :raises FileNotFoundError: Si la carpeta especificada no existe.
    :raises PermissionError: Si no se tienen permisos para leer la carpeta, escribir el ZIP o eliminar la carpeta.
    :return None: No retorna ningún valor.
    """
    # Obtener el nombre de la carpeta y el directorio padre
    folder_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    
    # Nombre del archivo ZIP (mismo nombre que la carpeta)
    zip_path = os.path.join(parent_dir, f"{folder_name}.zip")
    
    # Crear el archivo ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Recorrer todos los archivos y subcarpetas dentro de la carpeta
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calcular la ruta relativa para mantener la estructura dentro del ZIP
                relative_path = os.path.relpath(file_path, folder_path)
                # Agregar el archivo al ZIP
                zipf.write(file_path, relative_path)
    
    # Mover el archivo ZIP al directorio padre (si no está ya ahí)
    final_zip_path = os.path.join(parent_dir, f"{folder_name}.zip")
    if zip_path != final_zip_path:
        shutil.move(zip_path, final_zip_path)
    
    # Eliminar la carpeta original
    shutil.rmtree(folder_path)
    
    print(f"Carpeta '{folder_path}' comprimida en '{final_zip_path}' y eliminada.")

def crear_import(labels_path: str, images_path: str, import_folder: str, subfolders_pairs=None) -> None:
    """
    Crea la estructura para importar un dataset YOLO a CVAT,
    copiando etiquetas y asociando imágenes de cualquier extensión.

    :param labels_path: Carpeta con subcarpetas 'test', 'train', 'valid' de .txt
    :param images_path: Carpeta con subcarpetas 'test', 'train', 'valid' de imágenes
    :param import_folder: Carpeta destino donde se crea la estructura y ZIP
    :param subfolders_pairs: Lista de tuples ("etiqueta", "NombreTask")
    """
    if subfolders_pairs is None:
        subfolders_pairs = [("test", "Test"), ("train", "Train"), ("valid", "Validation")]

    # Validación de existencia
    for sub, _ in subfolders_pairs:
        lp = os.path.join(labels_path, sub)
        ip = os.path.join(images_path, sub)
        if not os.path.isdir(lp):
            raise FileNotFoundError(f"No existe labels: {lp}")
        if not os.path.isdir(ip):
            raise FileNotFoundError(f"No existe imágenes: {ip}")

    # Procesar cada partición
    for sub, folder in subfolders_pairs:
        label_subdir = os.path.join(labels_path, sub)
        image_subdir = os.path.join(images_path, sub)
        import_subdir = os.path.join(import_folder, sub)

        # Crear carpetas destino
        labels_dest = os.path.join(import_subdir, "labels", folder)
        images_dest = os.path.join(import_subdir, "images", folder)
        os.makedirs(labels_dest, exist_ok=True)
        os.makedirs(images_dest, exist_ok=True)

        # Para cada archivo .txt buscar su imagen asociada
        task_list = []
        for label_file in os.listdir(label_subdir):
            if not label_file.lower().endswith('.txt'):
                continue
            base = os.path.splitext(label_file)[0]
            # copiar etiqueta
            src_label = os.path.join(label_subdir, label_file)
            dst_label = os.path.join(labels_dest, label_file)
            shutil.copy2(src_label, dst_label)

            # buscar imagen en cualquier extensión
            pattern = os.path.join(image_subdir, base + '.*')
            matches = glob.glob(pattern)
            if not matches:
                print(f"Advertencia: imagen no encontrada para {base} en {image_subdir}")
                continue
            # tomar la primera coincidencia
            img_path = matches[0]
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(images_dest, img_name))
            task_list.append(f"data/images/{folder}/{img_name}")

        # escribir archivo de tarea
        task_file = os.path.join(import_subdir, f"{folder}.txt")
        with open(task_file, 'w') as f:
            for line in task_list:
                f.write(line + '\n')

        # escribir data.yaml
        yaml_file = os.path.join(import_subdir, "data.yaml")
        with open(yaml_file, 'w') as f:
            f.write(f"{folder}: {folder}.txt\n")
            f.write("names:\n")
            for i, cls in enumerate(CLASSES):
                f.write(f"  {i}: {cls}\n")
            f.write("path: .\n")

        # comprimir
        folder_to_zip(import_subdir)

# ?
# ? Convertir dataset CVAT a formato YOLO
# ?
def normalize_polygon(points: List[Tuple[float, float]], img_w: float, img_h: float) -> List[float]:
    """Normaliza las coordenadas de las etiquetas según el tamaño de imagen, en el rango [0, 1].

    :param List[Tuple[float, float]] points: Recibe puntos como lista de tuplas [(x1, y1), ..., (xn, yn)].
    :param float img_w: Ancho de la imagen.
    :param float img_h: Alto de la imagen.
    :return List[float]: Devuelve una lista con valores [x_norm_1, y_norm_1, ..., x_norm_n, y_norm_n].
    """
    normalized_points = []
    for x, y in points:
        normalized_points.append(x / img_w)
        normalized_points.append(y / img_h)
    return normalized_points

def parse_and_convert_segmentations(xml_path: str, output_dir: str):
    """Esta función lee un archivo XML con etiquetas de segmentación y las guarda en formato YOLO de segmentación usando save_yolo_segmentation.

    :param str xml_path: Dirección del archivo XML con el dataset en formato CVAT.
    :param str output_dir: Directorio donde guardar las etiquetas en formato YOLO.
    """
    # Leer archivo XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Crear carpeta donde guardar los archivos en formato YOLO
    os.makedirs(output_dir, exist_ok=True)

    # Recorrer todas las imágenes del dataset
    for img in root.findall('image'):
        # Saltarse imágenes sin etiquetas
        polygons = img.findall('polygon')
        if not polygons:
            continue

        # Nombre base sin extensión
        img_name = img.get('name')
        base, _ = os.path.splitext(img_name)
        label_file = os.path.join(output_dir, f"{base}.txt")
        img_w = float(img.get('width'))
        img_h = float(img.get('height'))

        # Lista para almacenar las segmentaciones de esta imagen
        segmentations: List[Dict[str, int | NDArray[np.float32]]] = []

        # Procesar cada polígono
        for poly in polygons:
            pts_str = poly.get('points')  # "x1,y1;x2,y2;..."
            # Parsear puntos
            pts = []
            for pair in pts_str.split(';'):
                x_str, y_str = pair.split(',')
                pts.append((float(x_str), float(y_str)))
            # Normalizar coordenadas de polígonos entre [0, 1]
            normalized_points = normalize_polygon(pts, img_w, img_h)
            # Convertir a NDArray[np.float32] de forma (N, 2)
            points_array = np.array(normalized_points).reshape(-1, 2).astype(np.float32)
            # Obtener class_id
            class_name = poly.get('label')
            class_id = CLASSES.index(class_name)
            # Crear diccionario para la segmentación
            seg = {'class_id': class_id, 'points': points_array}
            # Agregar a la lista de segmentaciones
            segmentations.append(seg)

        # Guardar las segmentaciones usando save_yolo_segmentation
        save_yolo_segmentation(segmentations, label_file)

def split_files(files: List[str], output_dir: str) -> Dict[str, List[str]]:
    """Divide una lista de archivos en tres listas según RATIOS."""
    random.seed(SEED)
    files_sorted = sorted(files)
    random.shuffle(files_sorted)
    total = len(files_sorted)
    n_train = int(total * RATIOS['train'])
    n_valid = int(total * RATIOS['valid'])
    train = files_sorted[:n_train]
    valid = files_sorted[n_train:n_train + n_valid]
    test = files_sorted[n_train + n_valid:]
    # Crear directorios
    splits = {'train': train, 'valid': valid, 'test': test}
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    return splits

def parse_and_convert_tracking_with_split(xml_path: str, output_dir: str, video_path: str = None, imgs_dir: str = None) -> None:
    """
    Convierte anotaciones CVAT a YOLO y extrae imágenes,
    luego divide ambos (etiquetas e imágenes) en train/valid/test.
    """
    # Directorio plano temporal
    temp_labels = os.path.join(output_dir, 'all_labels')
    os.makedirs(temp_labels, exist_ok=True)
    if video_path and imgs_dir:
        temp_imgs = os.path.join(imgs_dir, 'all_images')
        os.makedirs(temp_imgs, exist_ok=True)
    else:
        temp_imgs = None

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta = root.find('meta')
    original = meta.find('original_size') or meta.find('job/original_size') or  meta.find('task/original_size')
    width = float(original.find('width').text)
    height = float(original.find('height').text)

    # Acumular segmentaciones por frame
    frames_dict: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for track in root.findall('track'):
        class_name = track.get('label')
        if class_name not in CLASSES:
            continue
        class_id = CLASSES.index(class_name)
        for poly in track.findall('polygon'):
            outside = int(poly.get('outside'))
            if outside == 0:
                frame = int(poly.get('frame'))
                pts = [(float(x), float(y)) for x, y in
                       (p.split(',') for p in poly.get('points').split(';'))]
                norm = normalize_polygon(pts, width, height)
                pts_arr = np.array(norm, dtype=np.float32).reshape(-1, 2)
                seg = {'class_id': class_id, 'points': pts_arr}
                frames_dict.setdefault(frame, []).append(seg)

    video_name = os.path.splitext(os.path.basename(video_path))[0] if video_path else 'frame'

    # Guardar todas las etiquetas en carpeta plana
    for frame, segs in frames_dict.items():
        label_file = os.path.join(temp_labels, f"{video_name}_{frame:06d}.txt")
        save_yolo_segmentation(segs, label_file)

    # Extraer y guardar todas las imágenes en carpeta plana
    #if video_path and temp_imgs:
    #    cap = cv2.VideoCapture(video_path)
    #    if not cap.isOpened():
    #        raise IOError(f"No se pudo abrir video: {video_path}")
    #    for frame in sorted(frames_dict.keys()):
    #        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #        ret, img = cap.read()
    #        if not ret:
    #            print(f"No se pudo leer frame {frame}")
    #            continue
    #        img_file = os.path.join(temp_imgs, f"{video_name}_{frame:06d}.png")
    #        cv2.imwrite(img_file, img)
    #    cap.release()

    if video_path and temp_imgs:
        cmd = (
            f'ffmpeg -i "{video_path}" '
            f'-start_number 0 "{os.path.join(temp_imgs, video_name)}_%06d.png" '
            f'-hide_banner -loglevel error'
        )
        os.system(cmd)
        print(f"Convertido video completo a imágenes en: {temp_imgs}")

    # Obtener listas de archivos y dividir
    label_files = [os.path.join(temp_labels, f) for f in os.listdir(temp_labels) if f.endswith('.txt')]
    splits_labels = split_files(label_files, output_dir)

    if temp_imgs:
        img_files = [os.path.join(temp_imgs, f) for f in os.listdir(temp_imgs) if f.endswith('.png')]
        splits_imgs = split_files(img_files, imgs_dir)
    else:
        splits_imgs = {}

    # Mover archivos a sus carpetas finales
    for split, files in splits_labels.items():
        for src in files:
            dst = os.path.join(output_dir, split, os.path.basename(src))
            shutil.move(src, dst)
    for split, files in splits_imgs.items():
        for src in files:
            dst = os.path.join(imgs_dir, split, os.path.basename(src))
            shutil.move(src, dst)

    # Opcional: eliminar carpetas planas si ya no se necesitan
    shutil.rmtree(temp_labels)
    if temp_imgs:
        shutil.rmtree(temp_imgs)

def convert_xml_images(labels_path: str, output_dir: str):
    """Convierte un dataset en formato CVAT1.1 para imagenes al formato YOLO segmentación

    :param str labels_path: Direcotrio de las etiquetas en formato CVAT.
    :param str output_dir: Directorio donde guardar las etiquetas en formato YOLO.
    """
    subfolders = ["test", "train", "valid"]
    for sub in subfolders:
        label_subdir = os.path.join(labels_path, sub)
        out_label_subdir = os.path.join(output_dir, sub)
        xml_file = os.path.join(label_subdir, "annotations.xml")
        if not os.path.isfile(xml_file):
            continue
        parse_and_convert_segmentations(xml_file, out_label_subdir)

def convert_xml_video(labels_path: str, output_dir: str, video_path: str = None, video_store_path: str = None):
    """Convierte un dataset en formato CVAT1.1 para vide tracking o al formato YOLO segmentación

    :param str labels_path: Directorio de las etiquetas en formato CVAT.
    :param str output_dir: Directorio donde guardar las etiquetas en formato YOLO.
    :param str video_path (opcional): Arhivo de video a procesar.
    """
    xml_file = os.path.join(labels_path, "annotations.xml")
    parse_and_convert_tracking_with_split(xml_file, output_dir, video_path, video_store_path)

def simplest_filter(labels_path: str, filter_out_classes: List[int]):
    subfolders = ["test", "train", "valid"]

    for sub in subfolders:
        label_subdir = os.path.join(labels_path, sub)
        # Verificar si el subdirectorio existe
        if not os.path.exists(label_subdir):
            continue
   
        label_files = [f for f in os.listdir(label_subdir) if f.endswith(".txt")]

        for file in label_files:
            file_path = os.path.join(label_subdir, file)
            # Leer el archivo original
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Filtrar líneas y modificar clases
            new_lines = []
            for line in lines:
                # Obtener el class_id (primer número de la línea)
                parts = line.strip().split()
                if not parts:  # Saltar líneas vacías
                    continue
                class_id = int(parts[0])

                # Saltar líneas que correspondan a clases que queremos filtrar
                if class_id in filter_out_classes:
                    continue

                # Cambiar el class_id a 0 y mantener el resto de la línea
                parts[0] = '0'
                new_lines.append(' '.join(parts) + '\n')

            if len(new_lines) > 0:
                # Escribir el archivo modificado
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
            else:
                # Borrar el archivo si no hay líneas válidas
                os.remove(file_path)

def split_yolo_dataset(dataset_dir, seed=None, image_exts=('jpg', 'jpeg', 'png', 'bmp'), label_ext='txt'):
    """
    Split a YOLO-format instance segmentation dataset into train, test, and valid subsets.

    Args:
        dataset_dir (str): Root directory containing 'images/full' and 'labels/full'.
        ratios (tuple of float): Ratios for train, test, valid splits. Must sum to 1.0.
        seed (int, optional): Random seed for reproducibility.
        image_exts (tuple of str): Allowed image file extensions (without dot).
        label_ext (str): Extension for label files (without dot).

    Result:
        Creates 'images/train', 'images/test', 'images/valid' and corresponding
        'labels/train', 'labels/test', 'labels/valid' directories and copies files.
    """
    # Directories
    images_full = os.path.join(dataset_dir, 'images', 'full')
    labels_full = os.path.join(dataset_dir, 'labels', 'full')
    ratios=(0.65, 0.05, 0.30)

    # Validate directories exist
    if not os.path.isdir(images_full) or not os.path.isdir(labels_full):
        raise ValueError(f"Expected dirs 'images/full' and 'labels/full' under {dataset_dir}")

    # Gather all labeled images
    labeled_images = []
    for lbl_name in os.listdir(labels_full):
        if not lbl_name.endswith(f'.{label_ext}'):
            continue
        stem = os.path.splitext(lbl_name)[0]
        lbl_path = os.path.join(labels_full, lbl_name)
        # Find corresponding image
        for ext in image_exts:
            img_name = f"{stem}.{ext}"
            img_path = os.path.join(images_full, img_name)
            if os.path.exists(img_path):
                labeled_images.append((img_path, lbl_path))
                break

    if not labeled_images:
        print("No labeled images found. Nothing to split.")
        return

    # Shuffle
    if seed is not None:
        random.seed(seed)
    random.shuffle(labeled_images)

    # Check ratios
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must be a tuple of three floats summing to 1.0")
    r_train, r_test, r_valid = ratios

    n = len(labeled_images)
    n_train = int(n * r_train)
    n_test = int(n * r_test)
    n_valid = n - n_train - n_test

    splits = {
        'train': labeled_images[:n_train],
        'test': labeled_images[n_train:n_train + n_test],
        'valid': labeled_images[n_train + n_test:],
    }

    # Create output dirs and copy files
    for split, items in splits.items():
        img_out_dir = os.path.join(dataset_dir, 'images', split)
        lbl_out_dir = os.path.join(dataset_dir, 'labels', split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for img_path, lbl_path in items:
            shutil.copy2(img_path, os.path.join(img_out_dir, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(lbl_out_dir, os.path.basename(lbl_path)))

        print(f"Copied {len(items)} items to {split}")

    print("Dataset split complete.")

if __name__ == "__main__":
    videos = ["Seguimiento_1", "Seguimiento_2"]

    for video in videos:
        xml_path = os.path.join("xml_files", f"{video}")
        yolo_path = os.path.join("salmones", f"labels_{video}")
        filtered_path = os.path.join("salmones", f"labels_{video}_filetered")
        import_folder = os.path.join("salmones", f"import_{video}")
        images_path = "salmones/images"
        video_path = os.path.join("videos", f"{video}.mp4")
        video_store_path = os.path.join("salmones", f"{video}")

        subfolder_pairs = [("test", "Test"), ("train", "Train"), ("valid", "Validation")]

        #start = timer()  # Invocar la función para obtener el tiempo inicial
        #convert_xml_images(xml_path, yolo_path)
        #convert_xml_video(xml_path, yolo_path, video_path, video_store_path)
        #filtrar_dataset(yolo_path, filtered_path, images_path, False, False)
        #split_yolo_dataset(video_store_path)
        #crear_import("SalmonesV5_Complete/labels", "SalmonesV5_Complete/images", "SalmonesV5_Complete/import", subfolder_pairs)
        #end = timer()    # Invocar la función para obtener el tiempo final
        #print(f"Tiempo de ejecución: {end - start:.2f} segundos")

        #filter_out_classes = [2, 3, 4]
        #simplest_filter(yolo_path, filter_out_classes)
