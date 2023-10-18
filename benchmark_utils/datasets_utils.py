import numpy as np
import pandas as pd
from nilearn import masking, maskers


def load_dataset(subject, data_path, mask):
    alignment_contrasts = mask.inverse_transform(
        np.load(data_path / "alignment" / f"{subject}.npy")
    )
    decoding_contrasts = mask.inverse_transform(
        np.load(data_path / "decoding" / f"{subject}.npy")
    )
    labels = pd.read_csv(
        data_path / "labels" / f"{subject}.csv",
        header=None,
    ).values.ravel()

    return alignment_contrasts, decoding_contrasts, labels


def load_mask(data_path, memory):
    masker_path = data_path / "masks" / "mask.nii.gz"
    connected_mask = masking.compute_background_mask(
        masker_path, connected=True
    )
    mask = maskers.NiftiMasker(connected_mask, memory=memory).fit()
    return mask
