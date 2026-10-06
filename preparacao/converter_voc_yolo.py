"""Converte o RDD2022 do formato PASCAL VOC para o formato YOLO."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from constantes import CLASSES_ALVO, MAPEAMENTO, PAISES, ROTULOS_DO_RDD2022

RAIZ_DO_REPOSITORIO = Path(__file__).parent.parent


def ler_argumentos():
    """Le as opcoes da linha de comando."""
    leitor = argparse.ArgumentParser(description="Converte o RDD2022 de PASCAL VOC para o formato YOLO.")
    leitor.add_argument("--dados", default="dados/rdd2022")
    leitor.add_argument("--saida", default="dados/dados_yolo")
    return leitor.parse_args()


def resolver_caminho(caminho):
    """Resolve um caminho relativo a partir da raiz do repositorio."""
    alvo = Path(caminho)
    if alvo.is_absolute():
        return alvo
    return RAIZ_DO_REPOSITORIO / alvo


def contadores_zerados():
    """Devolve o dicionario de contagens zerado."""
    return {
        "imagens": 0,
        "imagens_vazias": 0,
        "convertidas": 0,
        "descartados": 0,
        "degeneradas": 0,
        "sem_imagem": 0,
    }


def somar(acumulado, parcela):
    """Soma as contagens de uma parcela no acumulado."""
    for chave in parcela:
        acumulado[chave] = acumulado[chave] + parcela[chave]


def converter_bbox(xmin, ymin, xmax, ymax, largura, altura):
    """Converte cantos em pixels para centro, largura e altura em fracao."""
    centro_x = (xmin + xmax) / 2.0 / largura
    centro_y = (ymin + ymax) / 2.0 / altura
    bbox_largura = (xmax - xmin) / largura
    bbox_altura = (ymax - ymin) / altura
    return centro_x, centro_y, bbox_largura, bbox_altura


def converter_xml(caminho_xml):
    """Devolve as linhas no formato YOLO e as contagens de um XML do RDD2022."""
    raiz = ET.parse(caminho_xml).getroot()

    tamanho = raiz.find("size")
    largura = float(tamanho.find("width").text)
    altura = float(tamanho.find("height").text)

    contagens = contadores_zerados()
    linhas = []

    for objeto in raiz.findall("object"):
        rotulo = objeto.find("name").text

        if rotulo not in ROTULOS_DO_RDD2022:
            raise ValueError("rotulo nao previsto: %s em %s" % (rotulo, caminho_xml))

        if rotulo not in MAPEAMENTO:
            contagens["descartados"] += 1
            continue

        caixa = objeto.find("bndbox")
        xmin = float(caixa.find("xmin").text)
        ymin = float(caixa.find("ymin").text)
        xmax = float(caixa.find("xmax").text)
        ymax = float(caixa.find("ymax").text)

        if xmax <= xmin or ymax <= ymin:
            contagens["degeneradas"] += 1
            continue

        centro_x, centro_y, bbox_largura, bbox_altura = converter_bbox(
            xmin, ymin, xmax, ymax, largura, altura
        )

        indice = CLASSES_ALVO.index(MAPEAMENTO[rotulo])
        linhas.append(
            "%d %.6f %.6f %.6f %.6f"
            % (indice, centro_x, centro_y, bbox_largura, bbox_altura)
        )
        contagens["convertidas"] += 1

    return linhas, contagens


def converter_pais(pais, raiz_entrada, raiz_saida):
    """Converte um pais inteiro, grava o resultado e devolve as contagens."""
    pasta_xml = raiz_entrada / pais / "train" / "annotations" / "xmls"
    pasta_img = raiz_entrada / pais / "train" / "images"

    destino_img = raiz_saida / pais / "images"
    destino_rot = raiz_saida / pais / "labels"
    destino_img.mkdir(parents=True, exist_ok=True)
    destino_rot.mkdir(parents=True, exist_ok=True)

    contagens = contadores_zerados()

    nomes_xml = sorted(os.listdir(pasta_xml))
    for nome_xml in nomes_xml:
        if not nome_xml.endswith(".xml"):
            continue

        base = nome_xml[:-4]
        origem_img = pasta_img / (base + ".jpg")
        if not origem_img.exists():
            contagens["sem_imagem"] += 1
            continue

        linhas, parcela = converter_xml(pasta_xml / nome_xml)
        somar(contagens, parcela)

        arquivo_rot = open(destino_rot / (base + ".txt"), "w", encoding="utf-8")
        for linha in linhas:
            arquivo_rot.write(linha + "\n")
        arquivo_rot.close()

        if len(linhas) == 0:
            contagens["imagens_vazias"] += 1

        atalho = destino_img / (base + ".jpg")
        if not atalho.exists():
            atalho.symlink_to(origem_img.resolve())

        contagens["imagens"] += 1

    return contagens


def main():
    """Converte todos os paises e imprime o relatorio."""
    args = ler_argumentos()

    raiz_entrada = resolver_caminho(args.dados)
    raiz_saida = resolver_caminho(args.saida)

    if not raiz_entrada.exists():
        raise SystemExit("ERRO: " + str(raiz_entrada) + " não encontrado.")

    print("entrada:", raiz_entrada)
    print("saida:", raiz_saida)
    print("classes, nesta ordem:", CLASSES_ALVO)
    print()

    total = contadores_zerados()
    for pais in PAISES:
        contagens = converter_pais(pais, raiz_entrada, raiz_saida)
        somar(total, contagens)
        print(
            "  %s: %d imagens, %d vazias, %d bbox"
            % (
                pais,
                contagens["imagens"],
                contagens["imagens_vazias"],
                contagens["convertidas"],
            )
        )

    print()
    print("RELATORIO")
    for chave in total:
        print(" ", chave + ":", total[chave])


if __name__ == "__main__":
    main()
