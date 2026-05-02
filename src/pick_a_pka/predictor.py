from typing import Literal

from rdkit import Chem

from .types import BackendType, MicrostateResult
from .exceptions import InvalidBackendError, InvalidMoleculeError
from .backends.molgpka.model import MolGpKaModel
from .backends.pkalearn.model import PkaLearnModel


class PKaPredictor:
    def __init__(
            self,
            model: Literal["molgpka", "pkalearn"] | BackendType = BackendType.MOLGPKA,
            device: str = "cpu",
            allow_amphoteric: bool = False,
    ):
        try:
            self.model_name = BackendType(model)
        except ValueError:
            raise InvalidBackendError(
                f"Unknown backend: '{model}'. Choose from: {[b.value for b in BackendType]}"
            )
        self.device = device
        self.allow_amphoteric = allow_amphoteric
        if self.model_name == BackendType.MOLGPKA:
            self.model = MolGpKaModel(device=self.device)
        elif self.model_name == BackendType.PKALEARN:
            self.model = PkaLearnModel(device=self.device,
                                       allow_amphoteric=self.allow_amphoteric
                                       )

    def __del__(self):
        # GPU cleanup
        if hasattr(self, "model") and hasattr(self.model, "dispose"):
            self.model.dispose()

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

    def predict_pka(self, mol: Chem.Mol | list[Chem.Mol] | str | list[str]) -> list[dict[int, float]]:
        """Predict the pKa values for a molecule or a list of molecules.

        :param mol: molecule, SMILES, or a list of either
        :return: a dictionary mapping each atom ID to its pKa value, for each molecule provided.
        """
        mols = self._to_mol(mol)
        results = [self.model.predict_pka(m) for m in mols]
        return results if isinstance(mol, list) else results[0]

    def predict_microstates(self, mol: Chem.Mol | list[Chem.Mol] | str | list[str],
                            ph: float | list[float] = 7.4,
                            ph_range: tuple = None, ph_step: float = None
                            ) -> list[MicrostateResult | dict[float, MicrostateResult]]:
        """Predict the relative abundances of the microstates for a molecule or a list of molecules at a given pH.

        :param mol: molecule, SMILES, or a list of either
        :param ph: A single pH value to determine the relative abundance of molecular micro-species at.
        :param ph_range: A range of pH to determine the relative abundance of molecular micro-species at. Ignored if `ph` is not None.
        :param ph_step: The incremental step to consider between values of the `ph_range`. Ignored if ph_range is None.
        :return: A list of `MicrostateResult` (if pH is not None) or of dictionaries for each molecule provided.
        """
        mols = self._to_mol(mol)
        results = [
            self.model.predict_microstates(mol_, ph=ph, ph_range=ph_range, ph_step=ph_step)
            for mol_ in mols
        ]
        return results if isinstance(mol, list) else results[0]
