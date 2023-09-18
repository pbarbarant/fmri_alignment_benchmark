import torch
import numpy as np
from fugw.mappings import FUGW, FUGWSparse
from fugw.scripts import coarse_to_fine, lmds
from nilearn import masking

class FugwAlignment():
    """Wrapper for FUGW alignment"""
    
    def __init__(
        self, 
        masker,
        method="coarse_to_fine",
        n_samples=1000,
        alpha_coarse=0.5,
        rho_coarse=1,
        eps_coarse=1e-6,
        alpha_fine=0.5,
        rho_fine=1,
        eps_fine=1e-6,
        radius=8,
    ) -> None:
        self.masker = masker
        self.method = method
        self.n_samples = n_samples
        self.alpha_coarse = alpha_coarse
        self.rho_coarse = rho_coarse 
        self.eps_coarse = eps_coarse
        self.alpha_fine = alpha_fine
        self.rho_fine = rho_fine 
        self.eps_fine = eps_fine
        self.radius = radius
    
    
    def fit(self, X, Y, verbose=False):
        """Fit FUGW alignment"""

        # Get main connected component of segmentation
        segmentation = (
            masking.compute_background_mask(
                self.masker.mask_img_, connected=True
            ).get_fdata()
            > 0
        )

        # Compute the embedding of the source and target data
        source_geometry_embeddings = lmds.compute_lmds_volume(
            segmentation
        ).nan_to_num()
        target_geometry_embeddings = source_geometry_embeddings.clone()
        source_embeddings_normalized, source_distance_max = (
            coarse_to_fine.random_normalizing(source_geometry_embeddings)
        )
        target_embeddings_normalized, target_distance_max = (
            coarse_to_fine.random_normalizing(target_geometry_embeddings)
        )

        # Subsample vertices as uniformly as possible on the surface
        source_sample = coarse_to_fine.sample_volume_uniformly(
            segmentation,
            embeddings=source_geometry_embeddings,
            n_samples=self.n_samples,
        )
        target_sample = coarse_to_fine.sample_volume_uniformly(
            segmentation,
            embeddings=target_geometry_embeddings,
            n_samples=self.n_samples,
        )

        coarse_mapping = FUGW(
            alpha=self.alpha_coarse,
            rho=self.rho_coarse,
            eps=self.eps_coarse,
            reg_mode="independent",
            divergence="kl",
        )

        fine_mapping = FUGWSparse(
            alpha=self.alpha_fine,
            rho=self.rho_fine,
            eps=self.eps_fine,
            reg_mode="independent",
            divergence="kl",
        )
        
        source_features = self.masker.transform(X)
        target_features = self.masker.transform(Y)
        source_features_normalized = source_features / np.linalg.norm(
            source_features, axis=1
        ).reshape(-1, 1)
        target_features_normalized = target_features / np.linalg.norm(
            target_features, axis=1
        ).reshape(-1, 1)

        coarse_to_fine.fit(
            # Source and target's features and embeddings
            source_features=source_features_normalized,
            target_features=target_features_normalized,
            source_geometry_embeddings=source_embeddings_normalized,
            target_geometry_embeddings=target_embeddings_normalized,
            # Parametrize step 1 (coarse alignment between source and target)
            source_sample=source_sample,
            target_sample=target_sample,
            coarse_mapping=coarse_mapping,
            coarse_mapping_solver="mm",
            coarse_mapping_solver_params={
                "nits_bcd": 10,
                "nits_uot": 100,
            },
            # Parametrize step 2 (selection of pairs of indices present in
            # fine-grained's sparsity mask)
            coarse_pairs_selection_method="topk",
            source_selection_radius=(
                self.radius / source_distance_max
            ),
            target_selection_radius=(
                self.radius / target_distance_max
            ),
            # Parametrize step 3 (fine-grained alignment)
            fine_mapping=fine_mapping,
            fine_mapping_solver="mm",
            fine_mapping_solver_params={
                "nits_bcd": 10,
                "nits_uot": 100,
            },
            # Misc
            device=torch.device("cuda:0"),
            verbose=verbose,
        )
            
        self.mapping = fine_mapping
        
        return self
    
    
    def transform(self, X):
        """Transform X"""
        
        features = self.masker.transform(X)
        transformed_features = self.mapping.transform(features)
        return self.masker.inverse_transform(transformed_features)
    