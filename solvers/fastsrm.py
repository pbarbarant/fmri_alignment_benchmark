from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fastsrm.identifiable_srm import IdentifiableFastSRM
    import os
    import numpy as np
    from joblib import Memory
    from sklearn.preprocessing import StandardScaler
    from benchmark_utils.config import MEMORY


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "FastSRM"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "n_components": [50],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["fastsrm", "joblib"]

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

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        X_train = []
        y_train = []
        X_test = []

        srm_path = os.path.join(MEMORY, "fastsrm")
        if not os.path.exists(srm_path):
            os.makedirs(srm_path)

        srm = IdentifiableFastSRM(
            n_components=self.n_components,
            aggregate="mean",
            temp_dir=srm_path,
            tol=1e-10,
            n_iter=100,
            n_jobs=5,
        )

        alignment_array = [
            self.mask.transform(contrasts).T
            for _, contrasts in self.dict_alignment.items()
        ]
        alignment_array.append(self.mask.transform(self.data_alignment_target).T)
        alignment_estimator = srm.fit(alignment_array)

        for subject in self.dict_alignment.keys():
            data_decoding = self.dict_decoding[subject]
            aligned_data = alignment_estimator.transform(
                [self.mask.transform(data_decoding).T]
            ).T
            X_train.append(aligned_data)
            labels = self.dict_labels[subject]
            y_train.append(labels)

        X_train = np.vstack(X_train)
        self.y_train = np.hstack(y_train).ravel()

        # Align the test data
        X_test = self.mask.transform(self.data_decoding_target)
        X_test = alignment_estimator.transform([X_test.T]).T

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
