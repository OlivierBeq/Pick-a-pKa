from abc import ABC, abstractmethod

from rdkit import Chem

from ..core.exceptions import InvalidMoleculeError


class BasePKaModel(ABC):
    def __init__(self, device="cpu"):
        self.device = device

    @abstractmethod
    def predict_pka(self, mol: Chem.Mol | list[Chem.Mol]) -> list[dict[int, float]]:
        pass

    def predict_microstates(self, mol: Chem.Mol | list[Chem.Mol], ph: float | list[float] = 7.4,
                            ph_range: tuple = None, ph_step: float = None
                            ) -> list[dict[float, Chem.Mol] | dict[float, dict[float, Chem.Mol]]]:
        raise NotImplementedError

    def dispose(self):
        pass

    def _to_mol(self, mol_or_smiles: Chem.Mol | str | list[Chem.Mol] | list[str]) -> list[Chem.Mol]:
        """Parse a molecule or list of molecules.

        :param mol_or_smiles: molecule, SMILES, or a list of either
        :return: a list of RDKit molecule object(s)
        """
        if isinstance(mol_or_smiles, list):
            return [mol for item in mol_or_smiles for mol in self._to_mol(item)]
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise InvalidMoleculeError(f"Invalid SMILES string: {mol_or_smiles}")
            return [mol]
        if not isinstance(mol_or_smiles, Chem.Mol):
            raise InvalidMoleculeError("Input must be an RDKit Mol or a SMILES string.")
        return [mol_or_smiles]
