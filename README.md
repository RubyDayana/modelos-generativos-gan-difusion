# Modelos generativos: DCGAN y fine-tuning de Stable Diffusion

**CADI Deep Learning Conceptos** · Universidad de Cundinamarca
**Docente:** Nidia Stella García Roa · **Estudiante:** Ruby Dayana Cárdenas Gómez · Noviembre 2026

Actividad práctica sobre modelos generativos de imágenes:

| Notebook | Contenido | Abrir |
|---|---|---|
| `01-DCGAN-Rostros.ipynb` | DCGAN en PyTorch entrenada desde cero con ~12 700 rostros (64×64, escala de grises). Comparación con una GAN de capas densas, interpolación en el espacio latente y prueba de memorización. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/modelos-generativos-gan-difusion/blob/main/notebooks/01-DCGAN-Rostros.ipynb) |
| `02-StableDiffusion-LoRA-Finetuning.ipynb` | Fine-tuning de Stable Diffusion v1.5 con LoRA usando `diffusers` + `peft` sobre fotografías de mariposas del Smithsonian. Exploración del VAE / espacio latente, difusión hacia adelante, antes vs. después, escala de LoRA y métrica CLIP. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/modelos-generativos-gan-difusion/blob/main/notebooks/02-StableDiffusion-LoRA-Finetuning.ipynb) |

## Estructura

```
modelos-generativos-gan-difusion/
├── notebooks/
│   ├── 01-DCGAN-Rostros.ipynb                 (ejecutado, con salidas)
│   └── 02-StableDiffusion-LoRA-Finetuning.ipynb (ejecutado en Colab con GPU T4)
├── figuras/                                    (figuras generadas por los notebooks)
├── requirements.txt
└── README.md
```

## Cómo ejecutar

1. Abrir el notebook con el botón **Open in Colab**.
2. Para el notebook 2 activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
3. Ejecutar todas las celdas (*Entorno de ejecución → Ejecutar todas*).

Los datos se descargan solos: el notebook 1 clona el repositorio público [muxspace/facial_expressions](https://github.com/muxspace/facial_expressions) (Apache 2.0) y el notebook 2 usa el conjunto [huggan/smithsonian_butterflies_subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) de Hugging Face.

## Resumen de resultados

- **DCGAN:** 2,2 M de parámetros en el generador, 25 épocas en CPU. Ver `figuras/04-rostros-generados-final.png` y el informe.
- **Stable Diffusion + LoRA:** rango 8, 0,19 % de la U-Net entrenable (≈1,6 M parámetros), 400 pasos en una T4. El archivo LoRA resultante pesa pocos MB.

## Referencias principales

- Goodfellow et al. (2014). Generative adversarial nets.
- Radford, Metz y Chintala (2016). Unsupervised representation learning with deep convolutional GANs.
- Ho, Jain y Abbeel (2020). Denoising diffusion probabilistic models.
- Rombach et al. (2022). High-resolution image synthesis with latent diffusion models.
- Hu et al. (2022). LoRA: Low-rank adaptation of large language models.
