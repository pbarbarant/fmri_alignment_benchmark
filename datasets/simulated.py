from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn import maskers, datasets


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "target": ["sub-01", "sub-02", "sub-03"],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_cmd = "conda"
    requirements = ["nilearn", "pandas"]

    def __init__(
        self,
        target="sub-01",
    ):
        self.subjects = ["sub-01", "sub-02", "sub-03"]
        self.n_samples_alignment = 100
        self.n_samples_decoding = 150
        self.n_features = 1876

    def generate_mock_data_subject(self):
        data_alignment = np.random.randn(
            self.n_samples_alignment, self.n_features
        )
        data_decoding = np.random.randn(
            self.n_samples_decoding, self.n_features
        )
        return data_alignment, data_decoding

    def generate_fake_labels(self):
        return np.random.randint(10, size=self.n_samples_decoding)

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Create a masker to extract the data from the brain volume.
        mask_img = datasets.load_mni152_brain_mask(resolution=10)
        mask = maskers.NiftiMasker(mask_img=mask_img).fit()
        # Generate pseudorandom data using `numpy`.
        dict_alignment = dict()
        dict_decoding = dict()
        dict_labels = dict()
        for subject in self.subjects:
            if subject == self.target:
                # Generate pseudorandom data using `numpy` for target subject.
                (
                    data_alignment_target,
                    data_decoding_target,
                ) = self.generate_mock_data_subject()
                # Convert the data to a brain volume using the masker.
                data_alignment_target = mask.inverse_transform(
                    data_alignment_target
                )
                data_decoding_target = mask.inverse_transform(
                    data_decoding_target
                )
                # Generate pseudorandom labels using `numpy` for target subject
                labels = self.generate_fake_labels()
                dict_labels[subject] = labels

            else:
                # Generate pseudorandom data using `numpy` for source subjects.
                (
                    data_alignment,
                    data_decoding,
                ) = self.generate_mock_data_subject()
                # Convert the data to a brain volume using the masker.
                dict_alignment[subject] = mask.inverse_transform(
                    data_alignment
                )
                dict_decoding[subject] = mask.inverse_transform(data_decoding)
                labels = self.generate_fake_labels()
                dict_labels[subject] = labels

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
