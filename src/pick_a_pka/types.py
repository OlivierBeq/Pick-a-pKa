from enum import Enum
from typing import TypedDict, Dict, List, Any

from rdkit import Chem


class BackendType(str, Enum):
    MOLGPKA = "molgpka"
    PKALEARN = "pkalearn"


class LadderStep(TypedDict):
    smiles: str
    center: int
    pka: float


class StateDistribution(TypedDict):
    smiles: str
    mol: Chem.Mol
    abundance: float



class MicrostateResult(TypedDict):
    major_state: Chem.Mol
    major_abundance: float
    distribution: list[StateDistribution]
