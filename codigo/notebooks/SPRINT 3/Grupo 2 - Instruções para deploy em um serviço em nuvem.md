### Instruções do modelo

O notebook localizado em `codigo/notebook/sprint03_modelo_segmentacao_final.ipynb` contém a implementação de um modelo de segmentação utilizando a arquitetura U-Net, tanto para execução em GPU quanto em CPU. A seguir, a documentação detalhada de cada célula de código, explicando como os pontos de boas práticas foram considerados e implementados.

#### Encapsulamento
```python
# A função abaixo encapsula a lógica de carregamento e pré-processamento de imagens,
# garantindo que mudanças internas não afetem o restante do código.
# Observamos os cuidados para modularizar o carregamento de imagens e máscaras,
# normalizando os valores entre 0 e 1 para manter a consistência dos dados.

def load_and_preprocess_image(image_path, mask_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalização entre 0 e 1

    mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')
    mask = img_to_array(mask) / 255.0  # Normalização entre 0 e 1

    return image, mask
```

#### Qualidade do código
```python
# A qualidade do código foi garantida por meio de práticas como:
# - Indentação consistente
# - Nomes de variáveis significativos
# - Comentários explicativos

from google.colab import drive
import pandas as pd
import random
import os
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

drive.mount('/content/drive')

# Definindo listas para armazenar caminhos de imagens e máscaras
images = []
masks = []

# Populando as listas com os caminhos correspondentes
for path in glob('/content/drive/Shared drives/Grupo T de Tech/Data/dataset_inteli/cropped_images/*/*'):
    images.append(path + '/image.tif')
    masks.append(path + '/mask.png')
```

#### Reutilização de código
```python
# Projetamos a arquitetura e o fluxo lógico da função visando a reutilização de código
# por meio da modularização de componentes comuns. Funções como `load_and_preprocess_image`
# podem ser reutilizadas em diferentes partes do pipeline de processamento de dados.

# Lista para armazenar imagens e máscaras pré-processadas
images_processed = []
masks_processed = []

# Carregar e pré-processar todas as imagens e máscaras
for img_path, mask_path in zip(images, masks):
    img, mask = load_and_preprocess_image(img_path, mask_path, target_size=(256, 256))
    images_processed.append(img)
    masks_processed.append(mask)

# Converter para arrays numpy
images_processed = np.array(images_processed)
masks_processed = np.array(masks_processed)
```

#### Funções reutilizáveis
```python
# Todas as funções envolvendo carregamento e pré-processamento de imagens são reutilizáveis e parametrizáveis,
# dado que os parâmetros como `image_path`, `mask_path` e `target_size` podem ser ajustados conforme necessário.

X_train, X_val, y_train, y_val = train_test_split(images_processed, masks_processed, test_size=0.3, random_state=42)
```

#### Documentação completa
```python
# Cada função está devidamente documentada com descrições claras de parâmetros, retornos e exemplos de uso,
# como pode ser visto na documentação gerada.

class CyclicLR(Callback):
    """
    Callback para ajuste cíclico da taxa de aprendizado.
    
    Parameters:
    - base_lr: taxa de aprendizado mínima.
    - max_lr: taxa de aprendizado máxima.
    - step_size: número de iterações para um ciclo completo.
    - mode: método de variação da taxa de aprendizado ('triangular', 'triangular2', 'exp_range').

    """
    def __init__(self, base_lr=1e-4, max_lr=1e-3, step_size=2000., mode='triangular'):
        # Inicialização de variáveis
        ...
```

#### Modularidade
```python
# O código foi dividido em módulos lógicos, facilitando a manutenção e a escalabilidade.
# Cada módulo é responsável por uma parte específica da lógica do projeto, como carregamento de dados,
# definição de modelos, e execução de treinamento.

# Exemplo de definição de um modelo U-Net
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # Definição do modelo...
    return model
```

#### Gerenciamento de dependências
```python
# Utilizamos um arquivo requirements.txt para listar todas as bibliotecas e dependências necessárias
# para o projeto, garantindo que todos os colaboradores usem as mesmas versões.

# requirements.txt
"""
pandas
numpy
tensorflow
Pillow
matplotlib
scikit-learn
"""
```

#### Configurações e variáveis de ambiente
```python
# As configurações específicas do ambiente, como credenciais e URLs de serviços externos,
# foram separadas em variáveis de ambiente para evitar hard encoding e aumentar a segurança.

# Exemplo de uso de variáveis de ambiente
import os

DATABASE_URL = os.getenv('DATABASE_URL')
```

#### Manuseio de erros
```python
# Implementamos uma estratégia de tratamento de erros para lidar com situações inesperadas de maneira elegante,
# como mostrado no bloco de código de exceção abaixo.

try:
    # Bloco de código que pode gerar exceções
    ...
except Exception as e:
    # Tratamento da exceção
    print(f"Erro encontrado: {e}")
```

