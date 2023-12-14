from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    import os
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from benchmark_utils.config import MEMORY
    from hyperalignment.fast_int import INT as HyperAlignment


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "FastINT"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_cmd = "conda"
    requirements = ["pip:hyperalignment", "joblib"]

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dict_alignment,
        dict_decoding,
        data_alignment_target,
        data_decoding_target,
        dict_labels,
        target,
        mask,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.data_alignment_target = data_alignment_target
        self.data_decoding_target = data_decoding_target
        self.dict_labels = dict_labels
        self.target = target
        self.mask = mask

    def run(self, n_iter=None):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        X_train = []
        y_train = []
        X_test = []

        fast_int_path = os.path.join(MEMORY, "fastint")
        if not os.path.exists(fast_int_path):
            os.makedirs(fast_int_path)

        fast_int = HyperAlignment(n_jobs=5)

        # Build alignment array
        alignment_array = [
            self.mask.transform(contrasts)
            for _, contrasts in self.dict_alignment.items()
        ]

        alignment_array.append(self.mask.transform(self.data_alignment_target))
        alignment_array = np.array(alignment_array)

        # Compute the projected data into the common space (tunning matrices) for alignment data
        alignment_estimator = fast_int.fit(X_train=alignment_array)
        X_train = alignment_estimator.get_tunning_matrices()
        X_train = np.vstack(X_train)

        decoding_array = []

        # Compute the projected data into the common space (tunning matrices) for decoding data
        for subject in self.dict_alignment.keys():
            data_decoding = self.dict_decoding[subject]
            masked_data_decoding = self.mask.transform(data_decoding)
            decoding_array.append(masked_data_decoding)
            labels = self.dict_labels[subject]
            y_train.append(labels)

        decoding_array.append(self.mask.transform(self.data_decoding_target))
        decoding_array = np.array(decoding_array)
        self.y_train = np.hstack(y_train).ravel()

        # Align the test data
        X_test = alignment_estimator.fit(decoding_array).get_tunning_matrices()[
            -1
        ]  # get only the decoding target data

        # Standard scaling
        se = StandardScaler()
        self.X_train = se.fit_transform(X_train)
        self.X_test = se.transform(X_test)
        self.y_test = self.dict_labels[self.target].ravel()

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )
