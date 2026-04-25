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
