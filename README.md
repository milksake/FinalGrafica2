# Learning to Score Olympic Events

Este proyecto utiliza un modelo C3D + LSTM para predecir las calificaciones de clavados (diving) y salto de potro (gymnastic vault) a partir de clips de video.

## Instalación

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```
### 2. Descargar los Datasets
Ejecuta los siguientes comandos en el directorio raíz del proyecto para descargar y extraer los datasets:
```bash
# Descargar datasets
wget http://rtis.oit.unlv.edu/datasets/diving.tar.gz
wget http://rtis.oit.unlv.edu/datasets/gymnastic_vault.tar.gz

# Extraer datasets
tar -xzvf diving.tar.gz
tar -xzvf gymnastic_vault.tar.gz
```
Esto creará los directorios `diving/` y `gymnastic_vault/`.

## Ejecutar

### 1. Entrenar los Modelos
Este script cargará los datos, entrenará los modelos tanto para clavados como para salto de potro, y guardará los pesos finales.
```bash
python train.py
```
Esto creará los archivos `diving_model_weights_2.pth` y `vault_model_weights_2.pth`.

### 2. Evaluar y Graficar Resultados
Este script evaluará el rendimiento del modelo en el conjunto de prueba. Mostrará gráficos de dispersión de las puntuaciones predichas frente a las reales e imprimirá las predicciones para algunos videos de ejemplo.
```bash
python evaluate.py
```
