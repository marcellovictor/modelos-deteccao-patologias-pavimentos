"""Analise exploratoria do RDD2022 bruto, antes de qualquer conversao."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from constantes import MAPEAMENTO, PAISES

RAIZ_DO_REPOSITORIO = Path(__file__).parent.parent


def ler_argumentos():
    """Le as opcoes da linha de comando."""
    leitor = argparse.ArgumentParser(description="Conta o que existe no RDD2022 bruto, antes da conversao.")
    leitor.add_argument("--dados", default="dados/rdd2022")
    return leitor.parse_args()


def resolver_caminho(caminho):
    """Resolve um caminho relativo a partir da raiz do repositorio."""
    alvo = Path(caminho)
    if alvo.is_absolute():
        return alvo
    return RAIZ_DO_REPOSITORIO / alvo


def contagens_zeradas():
    """Devolve o dicionario de contagens de um pais, zerado."""
    return {
        "xmls": 0,
        "imagens_na_pasta": 0,
        "xml_sem_imagem": 0,
        "sem_objeto": 0,
        "so_rotulo_descartado": 0,
        "com_classe_alvo": 0,
        "objetos": 0,
        "degeneradas": 0,
        "fora_dos_limites": 0,
        "rotulos": {},
        "resolucoes": {},
    }


def contar(dicionario, chave):
    """Soma um na contagem daquela chave, criando a entrada se faltar."""
    if chave not in dicionario:
        dicionario[chave] = 0
    dicionario[chave] += 1


def somar(acumulado, parcela):
    """Soma as contagens de uma parcela no acumulado."""
    for chave in parcela:
        if isinstance(parcela[chave], dict):
            for sub in parcela[chave]:
                if sub not in acumulado[chave]:
                    acumulado[chave][sub] = 0
                acumulado[chave][sub] += parcela[chave][sub]
        else:
            acumulado[chave] += parcela[chave]


def listar_jpg(pasta):
    """Devolve os nomes dos arquivos .jpg de uma pasta, em ordem."""
    nomes = []
    for nome in sorted(os.listdir(pasta)):
        if nome.endswith(".jpg"):
            nomes.append(nome)
    return nomes


def listar_xml(pasta):
    """Devolve os nomes dos arquivos .xml de uma pasta, em ordem."""
    nomes = []
    for nome in sorted(os.listdir(pasta)):
        if nome.endswith(".xml"):
            nomes.append(nome)
    return nomes


def analisar_xml(caminho_xml):
    """Le um XML e devolve as contagens dele."""
    raiz = ET.parse(caminho_xml).getroot()

    tamanho = raiz.find("size")
    largura = float(tamanho.find("width").text)
    altura = float(tamanho.find("height").text)

    contagens = contagens_zeradas()
    contar(contagens["resolucoes"], "%dx%d" % (largura, altura))

    objetos = raiz.findall("object")
    if len(objetos) == 0:
        contagens["sem_objeto"] += 1
        return contagens

    tem_classe_alvo = False
    for objeto in objetos:
        contagens["objetos"] += 1

        rotulo = objeto.find("name").text
        contar(contagens["rotulos"], rotulo)
        if rotulo in MAPEAMENTO:
            tem_classe_alvo = True

        caixa = objeto.find("bndbox")
        xmin = float(caixa.find("xmin").text)
        ymin = float(caixa.find("ymin").text)
        xmax = float(caixa.find("xmax").text)
        ymax = float(caixa.find("ymax").text)

        if xmax <= xmin or ymax <= ymin:
            contagens["degeneradas"] += 1
        if xmin < 0 or ymin < 0 or xmax > largura or ymax > altura:
            contagens["fora_dos_limites"] += 1

    if tem_classe_alvo:
        contagens["com_classe_alvo"] += 1
    else:
        contagens["so_rotulo_descartado"] += 1

    return contagens


def analisar_pais(pais, raiz_entrada):
    """Percorre os XMLs de um pais e devolve as contagens."""
    pasta_xml = raiz_entrada / pais / "train" / "annotations" / "xmls"
    pasta_img = raiz_entrada / pais / "train" / "images"

    contagens = contagens_zeradas()
    contagens["imagens_na_pasta"] = len(listar_jpg(pasta_img))

    for nome_xml in listar_xml(pasta_xml):
        contagens["xmls"] += 1

        base = nome_xml[:-4]
        if not (pasta_img / (base + ".jpg")).exists():
            contagens["xml_sem_imagem"] += 1

        somar(contagens, analisar_xml(pasta_xml / nome_xml))

    return contagens


def imprimir(titulo, contagens):
    """Imprime as contagens de um pais ou do total."""
    print("===", titulo, "===")
    print("  XMLs:", contagens["xmls"])
    print("  imagens na pasta:", contagens["imagens_na_pasta"])
    print("  XML sem imagem correspondente:", contagens["xml_sem_imagem"])
    print("  imagens sem nenhum objeto:", contagens["sem_objeto"])
    print("  imagens so com rotulo descartado:", contagens["so_rotulo_descartado"])
    print("  imagens com alguma classe alvo:", contagens["com_classe_alvo"])
    print("  objetos anotados:", contagens["objetos"])
    print("  bbox degeneradas:", contagens["degeneradas"])
    print("  bbox fora dos limites:", contagens["fora_dos_limites"])

    print("  rotulos:")
    for rotulo in sorted(contagens["rotulos"]):
        print("    ", rotulo, contagens["rotulos"][rotulo])

    print("  resolucoes:")
    ordenadas = sorted(
        contagens["resolucoes"], key=contagens["resolucoes"].get, reverse=True
    )
    for resolucao in ordenadas:
        print("    ", resolucao, contagens["resolucoes"][resolucao])
    print()


def main():
    """Analisa todos os paises e imprime o total."""
    args = ler_argumentos()
    raiz_entrada = resolver_caminho(args.dados)

    if not raiz_entrada.exists():
        raise SystemExit("ERRO: " + str(raiz_entrada) + " não encontrado.")

    print("entrada:", raiz_entrada)
    print()

    total = contagens_zeradas()
    for pais in PAISES:
        contagens = analisar_pais(pais, raiz_entrada)
        imprimir(pais, contagens)
        somar(total, contagens)

    imprimir("TOTAL", total)


if __name__ == "__main__":
    main()
