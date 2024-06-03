from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.config import DATA_PATH_NEUROMOD, MEMORY
    from benchmark_utils.datasets_utils import load_dataset, load_mask
    from pathlib import Path


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Neuromod"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "target": [
            "sub-01",
            "sub-02",
            "sub-03",
            "sub-05",
        ],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "pandas"]

    def __init__(self, target="sub-01"):
        self.subjects = [
            "sub-01",
            "sub-02",
            "sub-03",
            "sub-05",
        ]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        data_path = Path(DATA_PATH_NEUROMOD)

        # Load the masker object
        mask = load_mask(data_path, MEMORY)

        dict_alignment = dict()
        dict_sources = dict()
        dict_labels = dict()

        for subject in self.subjects:
            (
                alignment_contrasts,
                decoding_contrasts,
                labels,
            ) = load_dataset(subject, data_path, mask)
            dict_labels[subject] = labels

            if subject == self.target:
                data_target = alignment_contrasts
                data_target = decoding_contrasts
            else:
                dict_alignment[subject] = alignment_contrasts
                dict_sources[subject] = decoding_contrasts

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dict_alignment=dict_alignment,
            dict_sources=dict_sources,
            data_target=data_target,
            data_target=data_target,
            dict_labels=dict_labels,
            target=self.target,
            mask=mask,
        )
