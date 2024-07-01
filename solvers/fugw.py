from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fmralign.alignment_methods import FugwAlignment
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from nilearn import masking


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "fugw"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "alpha": [0.8],
        "rho": [1e3],
        "eps": [1e-6],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["pip:fmralign", "joblib"]

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dict_sources,
        data_target,
        dict_labels,
        target,
        mask,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dict_sources = dict_sources
        self.data_target = data_target
        self.dict_labels = dict_labels
        self.target = target
        self.mask = mask
        # Get main connected component of segmentation
        self.segmentation = (
            masking.compute_background_mask(
                self.mask.mask_img_, connected=True
            ).get_fdata()
            > 0
        )
        print("Segmentation shape:", self.segmentation.shape)

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        X_train = []
        y_train = []
        X_test = []

        for subject in self.dict_sources.keys():
            source_data = self.dict_sources[subject]
            alignment_estimator = FugwAlignment(
                self.segmentation,
                alpha_coarse=self.alpha,
                alpha_fine=self.alpha,
                rho_coarse=self.rho,
                rho_fine=self.rho,
                eps_coarse=self.eps,
                eps_fine=self.eps,
                method="coarse-to-fine",
                anisotropy=(3, 3, 3),
                reg_mode="independent",
                divergence="kl",
                n_landmarks=100,
                n_samples=3000,
                radius=10,
                verbose=True,
                coarse_mapping_solver="mm",
                fine_mapping_solver="mm",
                coarse_mapping_solver_params={
                    "nits_bcd": 5,
                },
                fine_mapping_solver_params={
                    "nits_bcd": 5,
                },
            ).fit(
                self.mask.transform(source_data),
                self.mask.transform(self.data_target),
            )

            data_decoding = self.dict_sources[subject]
            aligned_data = alignment_estimator.transform(
                self.mask.transform(data_decoding)
            )
            X_train.append(aligned_data)
            labels = self.dict_labels[subject]
            y_train.append(labels)

        X_train = np.vstack(X_train)
        self.y_train = np.hstack(y_train).ravel()

        # Test data
        X_test = self.mask.transform(self.data_target)
        self.y_test = self.dict_labels[self.target].ravel()

        # Standard scaling
        se = StandardScaler()
        self.X_train = se.fit_transform(X_train)
        self.X_test = se.transform(X_test)

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
