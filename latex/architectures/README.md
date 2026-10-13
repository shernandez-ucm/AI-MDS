# Semantic Segmentation Architectures

Diagramas en LaTeX de arquitecturas de red neuronal profundas para segmentación semántica.

## Contenido

### 1. **U-Net Architecture** (`unet_architecture.tex`)
- **Descripción**: Arquitectura fundamental para segmentación médica
- **Características**:
  - Encoder-decoder simétrico
  - Skip connections directas desde encoder a decoder
  - Pocos parámetros, entrenamiento eficiente
  - Excelente para datasets pequeños
- **Componentes**:
  - 5 niveles de convoluciones con max pooling (encoder)
  - Bottleneck central
  - 5 niveles de deconvoluciones (decoder)
  - 4 skip connections
- **Entrada/Salida**: `572×572×1` → `388×388×2`

### 2. **SegNet Architecture** (`segnet_architecture.tex`)
- **Descripción**: Arquitectura encoder-decoder basada en VGG-16
- **Características**:
  - Usa índices de max-pooling en lugar de almacenar feature maps
  - Menor consumo de memoria que U-Net
  - Preserva bordes mediante unpooling
  - Excelente para segmentación de escenas
- **Componentes**:
  - Encoder: 5 bloques convolucionales (VGG-like)
  - Decoder simétrico: 5 bloques deconvolucionales
  - Pooling/Unpooling con índices almacenados
- **Entrada/Salida**: `H×W×3` → `H×W×K`

### 3. **DeepLab ASPP Module** (`deeplab_aspp.tex`)
- **Descripción**: Módulo de captura de contexto multi-escala mediante convoluciones atrous
- **Características**:
  - Atrous Spatial Pyramid Pooling (ASPP)
  - Convoluciones dilatadas con tasas r ∈ {1, 6, 12, 18}
  - Rama de image pooling global
  - Mayor receptive field sin pérdida de resolución
- **Componentes**:
  - Backbone ResNet con stride=16
  - 5 ramas paralelas en ASPP:
    - 1×1 convolution (r=1)
    - Atrous 3×3 (r=6, 12, 18)
    - Global average pooling + 1×1 + upsample
  - Decoder: concatenación → 1×1 conv → bilinear upsample 8×
- **Ventaja**: Contexto global + detalles finos

## Compilación

Para compilar cualquier diagrama en PDF:

```bash
cd latex/architectures
pdflatex -interaction=nonstopmode unet_architecture.tex
pdflatex -interaction=nonstopmode segnet_architecture.tex
pdflatex -interaction=nonstopmode deeplab_aspp.tex
```

## Paleta de Colores

- **Azul (Encoder)**: RGB(20, 65, 134) - Convoluciones en fase de compresión
- **Rojo (Decoder)**: RGB(200, 40, 40) - Deconvoluciones en fase de expansión
- **Verde (Especial)**: RGB(0, 130, 100) - Pooling, ASPP, botlenecks
- **Gris**: RGB(240, 240, 240) - Entrada/Salida

## Notas

- Los diagramas usan estilos TikZ consistentes
- Las anotaciones incluyen dimensiones aproximadas de feature maps
- Todos los PDFs incluyen leyendas y etiquetas de componentes
- Compatible con beamer y otros documentos LaTeX

## Referencias

- Ronneberger et al. (2015): U-Net: Convolutional Networks for Biomedical Image Segmentation
- Badrinarayanan et al. (2017): SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
- Chen et al. (2017-2018): DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
