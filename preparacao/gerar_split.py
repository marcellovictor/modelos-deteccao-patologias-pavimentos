"""Divide as imagens convertidas em treino, validacao e teste."""

import argparse
import os
import random
from pathlib import Path

from constantes import CLASSES_ALVO, PAISES

RAIZ_DO_REPOSITORIO = Path(__file__).parent.parent

SEED = 42
FRACAO_TREINO = 0.70
FRACAO_VALIDACAO = 0.15


def ler_argumentos():
    """Le as opcoes da linha de comando."""
    leitor = argparse.ArgumentParser(description="Gera o split aleatorio estratificado por pais.")
    leitor.add_argument("--dados", default="dados/dados_yolo")
    leitor.add_argument("--saida", default="dados/split_aleatorio")
    return leitor.parse_args()


def resolver_caminho(caminho):
    """Resolve um caminho relativo a partir da raiz do repositorio."""
    alvo = Path(caminho)
    if alvo.is_absolute():
        return alvo
    return RAIZ_DO_REPOSITORIO / alvo


def listar_imagens(pasta):
    """Devolve os caminhos das imagens de um pais, em ordem."""
    caminhos = []
    for nome in sorted(os.listdir(pasta)):
        if nome.endswith(".jpg"):
            caminhos.append(pasta / nome)
    return caminhos


def dividir(caminhos, sorteador):
    """Embaralha e corta a lista em treino, validacao e teste."""
    embaralhados = list(caminhos)
    sorteador.shuffle(embaralhados)

    total = len(embaralhados)
    corte_treino = int(total * FRACAO_TREINO)
    corte_validacao = int(total * (FRACAO_TREINO + FRACAO_VALIDACAO))

    treino = embaralhados[:corte_treino]
    validacao = embaralhados[corte_treino:corte_validacao]
    teste = embaralhados[corte_validacao:]

    return treino, validacao, teste


def gravar_lista(caminho_saida, caminhos):
    """Grava uma lista de caminhos absolutos, um por linha."""
    arquivo = open(caminho_saida, "w", encoding="utf-8")
    for caminho in caminhos:
        arquivo.write(str(caminho.absolute()) + "\n")
    arquivo.close()


def gravar_data_yaml(caminho_saida, pasta_split):
    """Grava o data.yaml que a Ultralytics le."""
    arquivo = open(caminho_saida, "w", encoding="utf-8")
    arquivo.write("train: " + str((pasta_split / "treino.txt").absolute()) + "\n")
    arquivo.write("val: " + str((pasta_split / "validacao.txt").absolute()) + "\n")
    arquivo.write("test: " + str((pasta_split / "teste.txt").absolute()) + "\n")
    arquivo.write("nc: " + str(len(CLASSES_ALVO)) + "\n")
    arquivo.write("names:\n")
    for indice in range(len(CLASSES_ALVO)):
        arquivo.write("  " + str(indice) + ": " + CLASSES_ALVO[indice] + "\n")
    arquivo.close()


def main():
    """Monta o split e escreve o data.yaml."""
    args = ler_argumentos()

    raiz_entrada = resolver_caminho(args.dados)
    pasta_split = resolver_caminho(args.saida)

    if not raiz_entrada.exists():
        raise SystemExit("ERRO: " + str(raiz_entrada) + " não encontrado.")

    pasta_split.mkdir(parents=True, exist_ok=True)
    sorteador = random.Random(SEED)

    treino = []
    validacao = []
    teste = []

    print("entrada:", raiz_entrada)
    print("saida:", pasta_split)
    print()

    for pais in PAISES:
        caminhos = listar_imagens(raiz_entrada / pais / "images")
        parte_treino, parte_validacao, parte_teste = dividir(caminhos, sorteador)

        treino.extend(parte_treino)
        validacao.extend(parte_validacao)
        teste.extend(parte_teste)

        print(
            "  %s: %d treino, %d validacao, %d teste"
            % (pais, len(parte_treino), len(parte_validacao), len(parte_teste))
        )

    gravar_lista(pasta_split / "treino.txt", treino)
    gravar_lista(pasta_split / "validacao.txt", validacao)
    gravar_lista(pasta_split / "teste.txt", teste)
    gravar_data_yaml(pasta_split / "data.yaml", pasta_split)

    print()
    print(
        "total: %d treino, %d validacao, %d teste"
        % (len(treino), len(validacao), len(teste))
    )


if __name__ == "__main__":
    main()
