from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn import masking, maskers


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'subjects': [('sub-01', 'sub-02', 'sub-03')],
        'target': ['sub-01', 
                   'sub-02', 
                   'sub-03'],
        'n_samples_alignment': [10],
        'n_samples_decoding': [20],
        'n_features': [46407,],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []
    
    def __init__(self, subjects, target, n_samples_alignment, n_features, n_samples_decoding):
        self.subjects = subjects
        self.target = target
        self.n_samples_alignment = n_samples_alignment
        self.n_samples_decoding = n_samples_decoding
        self.n_features = n_features
    
    def generate_mock_data_subject(self):
        data_alignment = np.random.randn(self.n_samples_alignment, self.n_features)
        data_decoding = np.random.randn(self.n_samples_decoding, self.n_features)
        return data_alignment, data_decoding
    
    def generate_fake_labels(self):
        return np.random.randint(10, size=self.n_samples_decoding)

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Create a masker to extract the data from the brain volume.    
        masker_path = "/data/parietal/store3/work/pbarbara/public_analysis_code/ibc_data/gm_mask_3mm.nii.gz"
        connected_mask = masking.compute_background_mask(masker_path, connected=True)
        mask = maskers.NiftiMasker(connected_mask, memory="/data/parietal/store3/work/pbarbara/tmp").fit()
        
        # Generate pseudorandom data using `numpy`.
        dict_alignment = dict()
        dict_decoding = dict()
        dict_labels = dict()
        for subject in self.subjects:
            if subject == self.target:
                # Generate pseudorandom data using `numpy` for target subject.
                data_alignment_target, data_decoding_target = self.generate_mock_data_subject()
                # Convert the data to a brain volume using the masker.
                data_alignment_target = mask.inverse_transform(data_alignment_target)
                data_decoding_target = mask.inverse_transform(data_decoding_target)
                # Generate pseudorandom labels using `numpy` for target subject.
                labels = self.generate_fake_labels()
                dict_labels[subject] = labels
                
            else:
                # Generate pseudorandom data using `numpy` for source subjects.
                data_alignment, data_decoding = self.generate_mock_data_subject()
                # Convert the data to a brain volume using the masker.
                dict_alignment[subject] = mask.inverse_transform(data_alignment)
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
            mask=mask
        )
