"""Constantes do RDD2022 e as classes adotadas no trabalho."""

PAISES = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

CLASSES_ALVO = [
    "trinca_longitudinal",
    "trinca_transversal",
    "trinca_jacare",
    "panela",
]

MAPEAMENTO = {
    "D00": "trinca_longitudinal",
    "D10": "trinca_transversal",
    "D20": "trinca_jacare",
    "D40": "panela",
}

ROTULOS_DO_RDD2022 = {
    "D00",
    "D01",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
    "Repair",
    "Block crack",
    "D0w0",
}
