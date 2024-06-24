from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.config import DATA_PATH_IBC_MATHLANG_AUDIO, MEMORY
    from benchmark_utils.datasets_utils import load_dataset, load_mask
    from pathlib import Path


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "IBC_MathLangAudio"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "target": [
            "sub-01",
            "sub-04",
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-09",
            "sub-11",
            "sub-12",
            "sub-13",
            "sub-14",
        ],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "pandas"]

    def __init__(self, target="sub-01"):
        self.subjects = [
            "sub-01",
            "sub-04",
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-09",
            "sub-11",
            "sub-12",
            "sub-13",
            "sub-14",
        ]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        data_path = Path(DATA_PATH_IBC_MATHLANG_AUDIO)

        # Load the masker object
        mask = load_mask(data_path, MEMORY)

        dict_sources = dict()
        dict_labels = dict()

        for subject in self.subjects:
            (
                decoding_contrasts,
                labels,
            ) = load_dataset(subject, data_path, mask)
            dict_labels[subject] = labels

            if subject == self.target:
                data_target = decoding_contrasts
            else:
                dict_sources[subject] = decoding_contrasts

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dict_sources=dict_sources,
            data_target=data_target,
            dict_labels=dict_labels,
            target=self.target,
            mask=mask,
        )
