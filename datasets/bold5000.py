from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.config import DATA_PATH_BOLD5000, MEMORY
    from benchmark_utils.datasets_utils import load_mask
    from pathlib import Path
    import numpy as np
    import pandas as pd


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "BOLD5000"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "target": [
            "sub-CSI1",
            "sub-CSI2",
            "sub-CSI3",
            "sub-CSI4",
        ],
        "fold": [
            "fold_01",
            "fold_02",
            "fold_03",
            "fold_04",
        ],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "pandas"]

    def __init__(self, target="sub-CSI1", fold="fold_01"):
        self.subjects = [
            "sub-CSI1",
            "sub-CSI2",
            "sub-CSI3",
            "sub-CSI4",
        ]

    def load_bold5000(self, subject, fold, data_path, mask):
        alignment_contrasts = mask.inverse_transform(
            np.load(data_path / "alignment" / f"{subject}.npy")
        )
        decoding_contrasts = mask.inverse_transform(
            np.load(data_path / "decoding" / f"{subject}_{fold}.npy")
        )
        labels = pd.read_csv(
            data_path / "labels" / f"{subject}_{fold}.csv",
            header=None,
        ).values.ravel()

        return alignment_contrasts, decoding_contrasts, labels

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        data_path = Path(DATA_PATH_BOLD5000)

        # Load the masker object
        mask = load_mask(data_path, MEMORY)

        dict_alignment = dict()
        dict_decoding = dict()
        dict_labels = dict()

        for subject in self.subjects:
            (
                alignment_contrasts,
                decoding_contrasts,
                labels,
            ) = self.load_bold5000(subject, self.fold, data_path, mask)
            dict_labels[subject] = labels

            if subject == self.target:
                data_alignment_target = alignment_contrasts
                data_decoding_target = decoding_contrasts
            else:
                dict_alignment[subject] = alignment_contrasts
                dict_decoding[subject] = decoding_contrasts

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dict_alignment=dict_alignment,
            dict_decoding=dict_decoding,
            data_alignment_target=data_alignment_target,
            data_decoding_target=data_decoding_target,
            dict_labels=dict_labels,
            target=self.target,
            mask=mask,
        )
