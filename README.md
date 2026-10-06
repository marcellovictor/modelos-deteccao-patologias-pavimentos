# Treinamento de modelos de detecção de patologias em pavimentos

Scripts de preparação de dados, treino e avaliação dos modelos de detecção de
patologias em pavimento asfáltico.

O produto deste repositório são os arquivos de modelo (`.pt`) e as métricas medidas
sobre eles. A plataforma que consome esse modelo e apresenta as detecções ao
usuário fica em outro repositório.

As quatro classes detectadas, na versão atual dos scripts, são `trinca_longitudinal`, `trinca_transversal`,
`trinca_jacare` e `panela`.

## Requisitos

- Python 3.12
- GPU com CUDA. O `treinar.py` recusa a rodar sem ela.
- Dependências em `requirements.txt`:

```
pip install -r requirements.txt
```

**As versões são presas de propósito.** Atualizar a versão da ultralytics muda os valores
padrão de treino e torna as rodadas incomparáveis entre si.

## Dados

Os dados não acompanham o repositório.

Os scripts esperam o **RDD2022** na estrutura original do dataset: uma pasta por
país, cada uma com `train/images` e `train/annotations/xmls`. Os países esperados
são `China_Drone`, `China_MotorBike`, `Czech`, `India`, `Japan`, `Norway` e
`United_States`.

Colocando o dataset em `dados/rdd2022`, os comandos abaixo funcionam sem
argumento dos diretórios de origem e saída de dados. A pasta pode ser um link simbólico para onde o dataset já
estiver. Em caso de uma configuração alternativa de diretórios, todos os
caminhos são configuráveis por argumento.

As pastas `dados/` e `resultados/` estão fora do Git.

## Ordem de execução

As etapas dependem uma da outra, nesta ordem.

### 1. Análise exploratória

Conta o que existe no dataset bruto, antes de qualquer conversão: imagens por
país, rótulos encontrados, imagens sem anotação, resoluções, bounding boxes
degeneradas ou fora dos limites da imagem.

```
python preparacao/analise_exploratoria.py
python preparacao/analise_exploratoria.py --dados /outro/caminho/rdd2022
```

Serve para conferir a conversão depois, por caminho independente. Não produz
arquivo, só relatório na tela.

### 2. Conversão para o formato YOLO

Lê as anotações em PASCAL VOC (um XML por imagem, com os cantos da bounding box
em pixels) e grava no formato YOLO (um `.txt` por imagem, com classe, centro,
largura e altura em fração da imagem).

```
python preparacao/converter_voc_yolo.py
python preparacao/converter_voc_yolo.py --dados dados/rdd2022 --saida dados/dados_yolo
```

O que acontece com cada anotação:

- Rótulo entre as quatro classes adotadas: convertido.
- Rótulo conhecido do RDD2022 mas fora das quatro: descartado e contado.
- Rótulo não previsto: o script **para com erro**, em vez de descartar em
  silêncio.
- Bounding box sem área: descartada e contada.

Imagens sem nenhuma anotação das quatro classes recebem um `.txt` vazio e entram
no treino como fundo. O arquivo é sempre gravado, para que arquivo ausente seja
sinal de erro e não de imagem sem patologia.

As imagens no destino são **links simbólicos** para as originais, não cópias. O
dataset bruto precisa continuar onde está.

O relatório final traz as contagens, que devem bater com as da análise
exploratória.

### 3. Geração do split

Divide as imagens em treino, validação e teste, na proporção 70/15/15, sorteando
**dentro de cada país** para que todos apareçam nas três partes na mesma
proporção.

```
python preparacao/gerar_split.py
python preparacao/gerar_split.py --dados dados/dados_yolo --saida dados/split_aleatorio
```

Produz quatro arquivos: `treino.txt`, `validacao.txt` e `teste.txt`, com um
caminho de imagem por linha, e o `data.yaml`, que é o que a ultralytics lê.

O sorteio usa semente fixa, então rodar de novo produz o mesmo split.

### 4. Treino

```
python treinar.py --modelo yolov8n.pt
python treinar.py --modelo yolo11n.pt --epocas 100 --sufixo v2
python treinar.py --modelo yolo26n.pt --dados dados/split_aleatorio --saida resultados/treinos
```

Argumentos: `--modelo` (obrigatório), `--dados`, `--saida`, `--epocas`
(padrão 100) e `--sufixo`.

Antes de começar, o script confere que a GPU está disponível e apaga os
`labels.cache` de rodadas anteriores, porque a ultralytics os grava na pasta das
imagens e um cache velho faria o treino usar a lista errada sem avisar.

A rodada é nomeada `<arquitetura>_<split>[_<sufixo>]`, por exemplo
`yolov8n_split_aleatorio`. Ao fim, o script imprime onde ficaram o `best.pt` e o
`args.yaml` com os parâmetros usados.

Para treinar outra arquitetura, troque apenas `--modelo`. Os nomes disponíveis
são os que a ultralytics publica (`yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`,
`yolo26n.pt` e assim por diante); o arquivo é baixado automaticamente na primeira vez que utilizado.

### 5. Avaliação

```
python avaliar.py --modelo resultados/treinos/yolov8n_split_aleatorio/weights/best.pt
```

Argumentos: `--modelo` (obrigatório), `--dados` e `--saida`.

O script faz duas passadas:

1. **Na validação**, varre mil limiares de confiança e escolhe o que maximiza o
   F1 médio entre classes.
2. **No teste**, mede o desempenho nesse limiar.

Imprime a linha geral e a tabela por classe, e grava um
`<rodada>_resultado.json` com limiar e métricas.

O limiar que sai daqui é o que deve configurar o detector correspondente na
plataforma. Se a plataforma rodar em outro limiar, as métricas medidas deixam de
descrever o que o sistema faz.

Como o script só precisa de um `.pt` e de um split, serve igualmente para
avaliar modelos treinados por terceiros.
