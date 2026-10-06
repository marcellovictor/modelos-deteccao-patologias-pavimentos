"""Avalia um modelo treinado num split ja gerado."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

RAIZ_DO_REPOSITORIO = Path(__file__).parent

IMGSZ = 640
CASAS_DECIMAIS = 4


def ler_argumentos():
    """Le as opcoes da linha de comando."""
    leitor = argparse.ArgumentParser(description="Avalia um modelo YOLO.")
    leitor.add_argument("--modelo", required=True)
    leitor.add_argument("--dados", default="dados/split_aleatorio")
    leitor.add_argument("--saida", default="resultados/avaliacoes")
    return leitor.parse_args()


def resolver_caminho(caminho):
    """Resolve um caminho relativo a partir da raiz do repositorio."""
    alvo = Path(caminho)
    if alvo.is_absolute():
        return alvo
    return RAIZ_DO_REPOSITORIO / alvo


def nome_da_rodada(modelo):
    """Devolve o nome da rodada de treino que produziu o modelo."""
    return Path(modelo).parent.parent.name


def nomes_das_classes(modelo):
    """Devolve os nomes das classes na ordem dos indices."""
    classes = []
    for posicao in range(len(modelo.names)):
        classes.append(modelo.names[posicao])
    return classes


def arredondar(valor):
    """Arredonda um numero, ou os numeros de um dicionario ou de uma lista."""
    if isinstance(valor, float):
        return round(valor, CASAS_DECIMAIS)
    if isinstance(valor, dict):
        saida = {}
        for chave in valor:
            saida[chave] = arredondar(valor[chave])
        return saida
    if isinstance(valor, list):
        saida = []
        for item in valor:
            saida.append(arredondar(item))
        return saida
    return valor


def escolher_limiar(resultado):
    """Devolve o indice e o valor do limiar que maximiza o F1 medio na validacao."""
    f1_medio = resultado.box.f1_curve.mean(0)
    indice_do_limiar = int(f1_medio.argmax())
    return indice_do_limiar, float(resultado.box.px[indice_do_limiar])


def metricas_gerais(resultado, indice_do_limiar):
    """Devolve as metricas medias entre classes, sem ponderar por instancias."""
    return {
        "precisao": float(resultado.box.p_curve.mean(0)[indice_do_limiar]),
        "recall": float(resultado.box.r_curve.mean(0)[indice_do_limiar]),
        "f1": float(resultado.box.f1_curve.mean(0)[indice_do_limiar]),
        "map50": float(resultado.box.map50),
        "map50_95": float(resultado.box.map),
    }


def metricas_no_limiar(resultado, indice_do_limiar, classes):
    """Devolve precisao, recall, F1 e AP por classe no limiar escolhido."""
    posicao_por_classe = {}
    for posicao in range(len(resultado.box.ap_class_index)):
        posicao_por_classe[int(resultado.box.ap_class_index[posicao])] = posicao

    metricas_por_classe = []
    for indice_classe in range(len(classes)):
        if indice_classe not in posicao_por_classe:
            metricas_por_classe.append(
                {"classe": classes[indice_classe], "ausente": True}
            )
            continue

        posicao = posicao_por_classe[indice_classe]
        metricas_por_classe.append(
            {
                "classe": classes[indice_classe],
                "ausente": False,
                "precisao": float(resultado.box.p_curve[posicao][indice_do_limiar]),
                "recall": float(resultado.box.r_curve[posicao][indice_do_limiar]),
                "f1": float(resultado.box.f1_curve[posicao][indice_do_limiar]),
                "ap50": float(resultado.box.ap50[posicao]),
                "ap50_95": float(resultado.box.maps[posicao]),
            }
        )
    return metricas_por_classe


def imprimir_linha(rotulo, dados, chave_ap50, chave_ap50_95, prefixo_ap):
    """Imprime uma linha de metricas com rotulo."""
    print(
        "  %-24s P %.3f  R %.3f  F1 %.3f  %sAP50 %.3f  %sAP50-95 %.3f"
        % (
            rotulo,
            dados["precisao"],
            dados["recall"],
            dados["f1"],
            prefixo_ap,
            dados[chave_ap50],
            prefixo_ap,
            dados[chave_ap50_95],
        )
    )


def imprimir(geral, metricas_por_classe):
    """Imprime a linha geral e a tabela por classe."""
    imprimir_linha("GERAL (media de classes)", geral, "map50", "map50_95", "m")
    for linha in metricas_por_classe:
        if linha["ausente"]:
            print("  %-24s sem instancias no conjunto" % linha["classe"])
            continue
        imprimir_linha(linha["classe"], linha, "ap50", "ap50_95", "")


def main():
    """Escolhe o limiar na validacao e mede o desempenho no teste."""
    args = ler_argumentos()

    pasta_split = resolver_caminho(args.dados)
    dados = pasta_split / "data.yaml"
    if not dados.exists():
        raise SystemExit("ERRO: " + str(dados) + " não encontrado.")

    if "/" in args.modelo and not Path(args.modelo).exists():
        raise SystemExit("ERRO: " + args.modelo + " não encontrado.")

    modelo = YOLO(args.modelo)
    classes = nomes_das_classes(modelo)
    print("classes do modelo:", classes)

    saida = resolver_caminho(args.saida)
    rodada = nome_da_rodada(args.modelo)

    res_val = modelo.val(
        data=str(dados),
        split="val",
        imgsz=IMGSZ,
        project=str(saida),
        name=rodada + "_val",
    )
    indice_do_limiar, limiar = escolher_limiar(res_val)

    res_teste = modelo.val(
        data=str(dados),
        split="test",
        imgsz=IMGSZ,
        project=str(saida),
        name=rodada + "_teste",
    )

    geral = metricas_gerais(res_teste, indice_do_limiar)
    metricas_por_classe = metricas_no_limiar(res_teste, indice_do_limiar, classes)

    print()
    print("limiar escolhido na validacao:", round(limiar, CASAS_DECIMAIS))
    print()
    print("no teste, no limiar escolhido:")
    imprimir(geral, metricas_por_classe)

    registro = {
        "rodada": rodada,
        "modelo": args.modelo,
        "dados": args.dados,
        "limiar": round(limiar, CASAS_DECIMAIS),
        "geral": arredondar(geral),
        "por_classe": arredondar(metricas_por_classe),
    }

    arquivo = saida / (rodada + "_teste") / (rodada + "_resultado.json")
    arquivo.write_text(
        json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print("resultado gravado em:", arquivo)


if __name__ == "__main__":
    main()
