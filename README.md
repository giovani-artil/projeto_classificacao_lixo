# Projeto 2: Classificação de Lixo para Reciclagem

## 1. Problema e Objetivo
A **destinação correta de resíduos** é um desafio **global**, especialmente diante do aumento do consumo e dos impactos ambientais associados. Diante desse contexto — alinhado às discussões da COP 30 e às metas de sustentabilidade — **soluções automatizadas se tornam essenciais**.

Este projeto busca desenvolver e avaliar um **classificador de imagens multiclasse** utilizando **Redes Neurais Convolucionais (CNNs)** capaz de identificar automaticamente materiais recicláveis e rejeitos, classificando imagens em seis categorias: **cardboard, glass, metal, paper, plastic e trash**.

Além da construção do modelo, o experimento:
- Avalia o impacto de diferentes níveis de data augmentation.
- Analisa o desempenho usando:
  - acurácia,
  - matriz de confusão,
  - relatório por classe,
  - curvas de treino e validação.
---

## 2. Sobre o Repositório

| Arquivo | Descrição |
| --------------- | --------- |
| **Codigo** | Pasta com o notebook com carregamento do dataset, construção do modelo, treinamento, análises e imagens dos resultados. |
| **dataset** | Pasta com o dataset utilizado para a atividade. |
| **README.md** | Documento técnico com contexto, processo, resultados e conclusões. |

### Fontes usadas como base:

- O dataset TrashNet, disponível no GitHub (https://github.com/garythung/trashnet) e espelhado no Kaggle e Hugging Face (https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification).
- Conteúdo: Consiste em 2.527 imagens, divididas em 6 pastas, redimensionadas
(512x384 pixels) divididas em seis categorias. 

Vale mencionar que:
- O dataset é relativamente **pequeno e um pouco desbalanceado**, visto que a classe trash possui apenas 137 imagens, o que torna o
aumento de dados (**data augmentation**) uma **técnica necessária**.
- O notebook base utiliza python com TensorFlow / Keras.

---

## 3. Processo do Projeto

### 3.1. Carregamento e Preparação dos Dados: 

As imagens são carregadas diretamente do diretório utilizando **ImageDataGenerator**, com rescale=1./255 e divisão entre treino e validação. **Quatro configurações** de data augmentation são exploradas:

  - Baseline (sem augmentation)
  - Leve
  - Moderado
  - Agressivo

Cada configuração aplica transformações distintas para avaliar o impacto na capacidade de generalização do modelo, enquanto a validação permanece sem augmentation.

---

### 3.2 Configurações de Augmentation:

#### 1. Sem Augmentation (baseline)

```python
train_img_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)
```
#### 2. Augmentation Leve

```python
train_img_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split,
    horizontal_flip=True,
    zoom_range=0.05,
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02
)
```

#### 3. Augmentation Moderado

```python
train_img_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    shear_range=0.1,
    fill_mode='nearest'
)
```

#### 4. Augmentation Agressivo

```python
train_img_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.4,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=50,
    shear_range=0.3,
    brightness_range=[0.3, 1.5],
    fill_mode='nearest'
)
```

---

## 4. CNN, Treinamento e Avaliação

### 4.1. Construção da CNN
O modelo é definido com uma arquitetura Sequential composta por:
- Uma arquitetura CNN padrão (blocos de Conv2D -> ReLU -> MaxPooling2D).
- Um classificador (Flatten -> Dense -> Dropout -> Dense).

### 4.2. Treinamento e Avaliação
O modelo foi treinado utilizando os geradores de dados configurados e avaliado por métricas quantitativas e qualitativas, incluindo **acurácia categórica**, **curvas de aprendizado** e **matriz de confusão**, para visualizar quais classe são mais frequentemente confundidas entre si. 
Vale mencionar que:
- O teste foi realizado através do **Google Colab**.
- Utilizou-se a **GPU** como ambiente de execução.
- Configuramos o total de **epochs em 400** com **early stopping** (ele para antes para evitar overfitting).

--- 

## 5. Resultados
Os resultados apresentados correspondem ao cenário 4 — Augmentation Agressivo — por ter sido a configuração que alcançou o melhor desempenho geral no experimento.

### 5.1 Acurácia Geral
- **Treino:** ~70%
- **Validação:** ~70.5%

---

### 5.2 Desempenho por Classe (recall aproximado gerado)

| Classe     | Recall |
|-----------|--------|
| paper     | ~21%   |
| metal     | ~21%   |
| cardboard | ~10%   |
| glass     | ~16%   |
| plastic   | ~19%   |
| trash     | ~10%    |

A classe `trash` foi a mais difícil devido ao baixo número de exemplos e alto nível de variação visual.

---

### 5.3 Curvas de Treino e Validação

![Curva de Acurácia](AccuracyPlot.jpeg)

As curvas indicam:

- Aumento consistente na acurácia ao longo das épocas.
- Menor tendência ao overfitting, porque o modelo foi exposto a variações fortes.
- Validação estabilizada em torno de 70%, mas sem melhora nas métricas por classe.
- Indício de que a rede aprendeu padrões mais gerais, porém pouco específicos para cada categoria.

---

### 5.4 Matriz de Confusão

![Matriz de Confusão](MatrizConfusao.jpeg)

A matriz mostra:

- Melhores desempenhos relativos para metal e paper.
- Confusões frequentes entre plastic, glass e cardboard.
- Classe trash ainda sendo a mais problemática, com dispersão grande e poucos acertos.

Apesar da acurácia global maior, a matriz deixa claro que o modelo não aprendeu bem as classes individualmente.

---

## 6. Impacto do Data Augmentation

Foram analisadas quatro configurações, mas aqui destacamos a comparação mais relevante:

### **1. Sem augmentation**
- Acurácia de teste: ~57%
- Overfitting elevado
- Generalização fraca

### **2. Augmentation leve/moderado**
-	Acurácia entre 60–65%
-	Melhor equilíbrio entre treino e validação
-	Métricas por classe ainda modestas, mas mais estáveis

### **3. Augmentation agressivo (atual)**
- Acurácia de teste: ~70.5% (melhor resultado global)
- Loss mais baixo
- Porém, precisão e recall por classe continuam muito baixos
- Modelo acerta “no geral”, mas erra muito dentro de cada categoria específica
- Forte indício de que o augmentation exagerado cria variações irreais, confundindo padrões importantes

**Conclusão:**  
O augmentation agressivo elevou a acurácia total, mas não melhorou a capacidade do modelo em distinguir as classes individualmente. No contexto desse dataset, a configuração agressiva **gera diversidade** suficiente para aumentar a acurácia global, mas ao custo de **piorar a precisão por classe**. Assim, o augmentation moderado continua sendo o mais balanceado para generalização real.

---

## 7. Limitações

- Dataset **pequeno e desbalanceado**, o que prejudica o aprendizado de classes menos representadas.
- **Similaridade visual alta** entre algumas categorias, como plastic, glass e cardboard, levando a confusões frequentes mesmo com augmentation.
- Uso de uma arquitetura **relativamente simples**, que limita a capacidade do modelo de extrair padrões mais profundos — especialmente perceptível nas métricas por classe.
- Classe trash com **poucos exemplos** e grande variabilidade interna, resultando em baixo recall mesmo nas melhores configurações.
- Augmentation agressivo aumenta a acurácia geral, mas não resolve a baixa precisão e recall das categorias, já que parte das transformações pode distorcer demais os padrões reais das imagens.
- Ausência de técnicas adicionais de regularização (como class weighting ou focal loss), que poderiam ajudar no desbalanceamento entre classes.

---

## 8. Integrantes:

- Giovani Artil Oliveira de Carvalho (giovaniartil@icomp.ufam.edu.br)
- Jorge Samuel Silva Coelho (samcoelho@icomp.ufam.edu.br)
- Renata Modesto Fernandes (renata.modesto@icomp.ufam.edu.br)
- Sofia Pinho Icavino Moura (sofiaicavino@icomp.ufam.edu.br)
- Vitória Luz Edwards (vitoria.edwards@icomp.ufam.edu.br)

---
