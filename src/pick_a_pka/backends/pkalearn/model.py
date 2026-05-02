from importlib import resources

import torch
from rdkit import Chem

from .network import PkaLearnGNN
from ...core.base import BasePKaModel
from ...exceptions import ResourceNotFoundError


class PkaLearnModel(BasePKaModel):
    DEFAULT_CONFIG = {
        'atom_feature_element': False,
        'atom_feature_electronegativity': True,
        'atom_feature_hardness': True,
        'atom_feature_atom_size': True,
        'atom_feature_hybridization': True,
        'atom_feature_aromaticity': True,
        'atom_feature_number_of_rings': False,
        'atom_feature_ring_size': True,
        'atom_feature_number_of_Hs': True,
        'atom_feature_formal_charge': True,
        'bond_feature_bond_order': True,
        'bond_feature_conjugation': True,
        'bond_feature_polarization': True,
        'bond_feature_charge_conjugation': True,
        'bond_feature_focused': False,
        'acid_or_base': 'base',
        'mask_size': 4,
        'model_embedding_size': 128,
        'model_gnn_layers': 4,
        'model_fc_layers': 2,
        'model_dropout_rate': 0.0,
        'model_dense_neurons': 448,
        'model_attention_heads': 4
    }

    def __init__(self, device="cpu", config=None, allow_amphoteric: bool = False):
        super().__init__(device=device)
        self.config = config or self.DEFAULT_CONFIG
        self.allow_amphoteric = allow_amphoteric
        self.model = PkaLearnGNN(feature_size=19, edge_dim=7, model_params=self.config)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        pkg = "pick_a_pka.backends.pkalearn.resources"
        try:
            with resources.as_file(resources.files(pkg).joinpath("train_AAc-1_best.pth")) as path:
                ckpt = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ResourceNotFoundError(f"Cound not load pKaLearn model weights: {e}")

    @torch.no_grad()
    def predict(self, mol_or_smiles):
        """
        Runs the full iterative deprotonation ladder.
        Returns a list of dicts: [{'smiles': ..., 'center': ..., 'pka': ...}, ...]
        """
        from .microstates import predict_ladder

        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = Chem.Mol(mol_or_smiles)

        mol_clean = Chem.RemoveHs(mol)

        if self.allow_amphoteric:
            # Pre-protonate all neutral nitrogens so they enter the ladder at the top
            rw_mol = Chem.RWMol(mol_clean)
            patt = Chem.MolFromSmarts('[#7+0]')
            if patt:
                for m in rw_mol.GetSubstructMatches(patt):
                    atom = rw_mol.GetAtomWithIdx(m[0])
                    # Guard against over-bonding limits
                    if atom.GetDegree() <= 3:
                        atom.SetFormalCharge(1)
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            try:
                Chem.SanitizeMol(rw_mol)
                mol_clean = rw_mol.GetMol()
            except Exception:
                # Fallback to the original cleanly stripped molecule if RDKit rejects
                mol_clean = Chem.RemoveHs(mol)

        # Using non-canonical SMILES preserves the native node order perfectly
        smiles_str = Chem.MolToSmiles(mol_clean, canonical=False)
        return predict_ladder(self, smiles_str, self.config, allow_amphoteric=self.allow_amphoteric)

    def predict_pka(self, mol):
        mol_clean = Chem.RemoveHs(mol) if isinstance(mol, Chem.Mol) else Chem.MolFromSmiles(mol)
        ladder = self.predict(mol_clean)

        base_pka = {}
        acid_pka = {}

        for step in ladder:
            pka = step['pka']
            step_mol = Chem.MolFromSmiles(step['smiles'], sanitize=False)

            if not step_mol:
                continue

            idx = step['center']

            # Since pKaLearn parses from canonical=False SMILES, indices map 1:1.
            # Catch unexpected indexing mismatches purely as a safety mechanism
            if idx >= mol_clean.GetNumAtoms():
                continue

            # Look at the atom in its DEPROTONATED state
            fc = step_mol.GetAtomWithIdx(idx).GetFormalCharge()

            # 100% Thermodynamic Rule:
            # If deprotonation yields an anion (fc < 0), it's an ACIDIC pKa.
            # If deprotonation yields a neutral species (fc >= 0), it's a BASIC pKa.
            if fc < 0:
                acid_pka[idx] = pka
            else:
                base_pka[idx] = pka

        return {
            "base_pka": base_pka,
            "acid_pka": acid_pka,
            "mol": mol_clean
        }

    def predict_microstates(self, mol, ph=7.4, ph_range=None, ph_step=None):
        from .microstates import compute_microstates
        return compute_microstates(self, mol, ph=ph, ph_range=ph_range, ph_step=ph_step)
