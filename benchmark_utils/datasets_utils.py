import joblib
import pandas as pd
from nilearn import masking, maskers


def load_dataset(subject, data_path, mask):
    decoding_contrasts = mask.inverse_transform(
        joblib.load(data_path / f"{subject}.pkl")
    )
    labels = pd.read_csv(
        data_path / "labels" / f"{subject}.csv",
        header=None,
    ).values.ravel()

    return decoding_contrasts, labels


def load_mask(data_path, memory):
    masker_path = data_path / "masks" / "mask.nii.gz"
    connected_mask = masking.compute_background_mask(
        masker_path, connected=True
    )
    mask = maskers.NiftiMasker(connected_mask, memory=memory).fit()
    return mask
