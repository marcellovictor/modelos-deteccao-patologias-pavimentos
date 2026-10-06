"""Treina uma arquitetura YOLO num split ja gerado."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

RAIZ_DO_REPOSITORIO = Path(__file__).parent

IMGSZ = 640
BATCH = 16
OTIMIZADOR = "SGD"
SEED = 42

ARQUIVOS_DO_SPLIT = ["treino.txt", "validacao.txt", "teste.txt"]


def ler_argumentos():
    """Le as opcoes da linha de comando."""
    leitor = argparse.ArgumentParser(description="Treina uma arquitetura YOLO.")
    leitor.add_argument("--modelo", required=True)
    leitor.add_argument("--dados", default="dados/split_aleatorio")
    leitor.add_argument("--saida", default="resultados/treinos")
    leitor.add_argument("--epocas", type=int, default=100)
    leitor.add_argument("--sufixo", default="")
    return leitor.parse_args()


def resolver_caminho(caminho):
    """Resolve um caminho relativo a partir da raiz do repositorio."""
    alvo = Path(caminho)
    if alvo.is_absolute():
        return alvo
    return RAIZ_DO_REPOSITORIO / alvo


def conferir_gpu():
    """Interrompe se a GPU nao estiver disponivel."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "ERRO: a GPU nao está disponivel."
        )
    print("GPU:", torch.cuda.get_device_name(0))


def limpar_cache(pasta_split):
    """Apaga os labels.cache deixados por rodadas anteriores."""
    pastas = set()
    for nome in ARQUIVOS_DO_SPLIT:
        arquivo = pasta_split / nome
        if not arquivo.exists():
            continue
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            if linha.strip() != "":
                pastas.add(Path(linha.strip()).parent.parent)

    for pasta in pastas:
        alvo = pasta / "labels.cache"
        if alvo.exists():
            alvo.unlink()
            print("cache apagado:", alvo)


def montar_nome(modelo, pasta_split, sufixo):
    """Monta o nome da rodada a partir da arquitetura e do split."""
    arquitetura = Path(modelo).name.replace(".pt", "")
    nome = arquitetura + "_" + Path(pasta_split).name
    if sufixo != "":
        nome = nome + "_" + sufixo
    return nome


def main():
    """Treina a partir do modelo pre-treinado no dataset COCO."""
    args = ler_argumentos()

    pasta_split = resolver_caminho(args.dados)
    dados = pasta_split / "data.yaml"
    if not dados.exists():
        raise SystemExit("ERRO: " + str(dados) + " não encontrado.")

    if "/" in args.modelo and not Path(args.modelo).exists():
        raise SystemExit("ERRO: " + args.modelo + " não encontrado.")

    conferir_gpu()
    limpar_cache(pasta_split)

    modelo = YOLO(args.modelo)
    modelo.train(
        data=str(dados),
        epochs=args.epocas,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer=OTIMIZADOR,
        seed=SEED,
        deterministic=True,
        project=str(resolver_caminho(args.saida)),
        name=montar_nome(args.modelo, pasta_split, args.sufixo),
    )

    pasta = Path(modelo.trainer.save_dir)
    print()
    print("modelo treinado:", pasta / "weights" / "best.pt")
    print("parametros da rodada:", pasta / "args.yaml")


if __name__ == "__main__":
    main()
