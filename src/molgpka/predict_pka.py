#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize
import torch
from importlib import resources

from .utils.ionization_group import get_ionization_aid
from .utils.descriptor import mol2vec
from .utils.net import GCNNet
import molgpka.models


def _load_model(model_file, device="cpu"):
    model= GCNNet().to(device)
    # weights_only=True is strictly required in modern PyTorch for security
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    return model

def _model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def _predict_acid(mol):
    with resources.as_file(resources.files(molgpka.models).joinpath('weight_acid.pth')) as model_file:
        model_acid = _load_model(model_file)

    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = _model_pred(mol, aid, model_acid)
        acid_res.update({aid:apka})
    return acid_res

def _predict_base(mol):
    with resources.as_file(resources.files(molgpka.models).joinpath('weight_base.pth')) as model_file:
        model_base = _load_model(model_file)

    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = _model_pred(mol, aid, model_base)
        base_res.update({aid:bpka})
    return base_res

def predict(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = _predict_base(mol)
    acid_dict = _predict_acid(mol)
    # Remap without hydrogens
    mol, base_dict, acid_dict = _remap_pka_without_hs(mol, base_dict, acid_dict)
    return base_dict, acid_dict, mol

def _predict_for_protonate(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = _predict_base(mol)
    acid_dict = _predict_acid(mol)
    return base_dict, acid_dict, mol


def _remap_pka_without_hs(mol_with_hs: Chem.Mol, base_pka_dict: dict, acid_pka_dict: dict) -> tuple[Chem.Mol, dict, dict]:
    """
    Remap pKa atom indices in a molecule with explicit hydrogens to the molecule without hydrogens.

    :return: the translated pKa dictionaries and the molecule without Hs.
    """
    # 1. Tag every atom with its original index
    for atom in mol_with_hs.GetAtoms():
        atom.SetIntProp("OrigIdx", atom.GetIdx())

    # 2. Map Hydrogen indices to their Heavy Atom neighbors
    # (Just in case the pKa dictionary points to the H's instead of the O's or N's)
    h_to_heavy = {}
    for atom in mol_with_hs.GetAtoms():
        if atom.GetAtomicNum() == 1:  # If it is a Hydrogen
            neighbors = atom.GetNeighbors()
            if neighbors:
                # Store: H_index -> Heavy_Atom_index
                h_to_heavy[atom.GetIdx()] = neighbors[0].GetIdx()

    # 3. Safely remove Hydrogens
    mol_no_hs = Chem.RemoveHs(mol_with_hs)

    # 4. Create a mapping from Original Index -> New Index for the remaining atoms
    orig_to_new_idx = {}
    for atom in mol_no_hs.GetAtoms():
        if atom.HasProp("OrigIdx"):
            orig_idx = atom.GetIntProp("OrigIdx")
            orig_to_new_idx[orig_idx] = atom.GetIdx()

    # 5. Translate the pKa dictionary
    new_acid_pka_dict = {}
    new_base_pka_dict = {}
    for old_idx, pka_val in acid_pka_dict.items():
        if old_idx in orig_to_new_idx:
            # The pKa was attached to a Heavy Atom that survived
            new_idx = orig_to_new_idx[old_idx]
            new_acid_pka_dict[new_idx] = pka_val

        elif old_idx in h_to_heavy:
            # The pKa was attached to a Hydrogen!
            # Find the heavy atom it belonged to, and use THAT atom's new index
            heavy_orig_idx = h_to_heavy[old_idx]
            if heavy_orig_idx in orig_to_new_idx:
                new_idx = orig_to_new_idx[heavy_orig_idx]
                new_acid_pka_dict[new_idx] = pka_val
    for old_idx, pka_val in base_pka_dict.items():
        if old_idx in orig_to_new_idx:
            # The pKa was attached to a Heavy Atom that survived
            new_idx = orig_to_new_idx[old_idx]
            new_base_pka_dict[new_idx] = pka_val

        elif old_idx in h_to_heavy:
            # The pKa was attached to a Hydrogen!
            # Find the heavy atom it belonged to, and use THAT atom's new index
            heavy_orig_idx = h_to_heavy[old_idx]
            if heavy_orig_idx in orig_to_new_idx:
                new_idx = orig_to_new_idx[heavy_orig_idx]
                new_base_pka_dict[new_idx] = pka_val

    return mol_no_hs, new_base_pka_dict, new_acid_pka_dict
