import numpy as np
import pandas as pd
from combchoice.core import Model
from combchoice.loaders import load_market_data
from combchoice.utils import get_logger

logger = get_logger(__name__)


class UnitDemandChoice(Model):

    def load_product_data(self, product_data, specification, availability_data=None):
        id_data, n_obs, n_items = load_market_data(product_data, availability_data=availability_data, covariates=[])
        self._input_data = {"id_data": id_data, "item_data": {}}
        self._specification = specification

        market_ids, bundles = id_data["market_id"], id_data["obs_bundles"]
        n_markets = int(market_ids.max()) + 1
        df_cols = [c for c in specification if c != "constant"]
        raw = product_data[df_cols].values.astype(np.float64) if df_cols else np.empty((n_obs, 0))
        if "constant" in specification:
            raw = np.insert(raw, specification.index("constant"), 1.0, axis=1)
        self._characteristics = np.zeros((n_markets, n_items, len(specification)))
        self._characteristics[market_ids, np.argmax(bundles, axis=1)] = raw

    def load_demographics(self, market_demographics, demographics=None):
        if self.comm_manager.is_root():
            nu_cols = sorted(c for c in market_demographics.columns if c.startswith('nodes'))
            demo_cols = list(demographics) if demographics is not None else [
                c for c in market_demographics.columns if c not in {'market_id', 'weights'} | set(nu_cols)]
            _, inv = np.unique(market_demographics['market_id'].values, return_inverse=True)
            counts = np.bincount(inv)
            if not np.all(counts == counts[0]):
                raise ValueError("Unbalanced simulations across markets not yet supported.")
            n_sims, n_markets = int(counts[0]), len(counts)
            self._nu = market_demographics[nu_cols].values.astype(np.float64).reshape(n_markets, n_sims, -1)
            self._demos = market_demographics[demo_cols].values.astype(np.float64).reshape(n_markets, n_sims, -1) if demo_cols else None
            parsed = (self._nu, self._demos, n_sims, demo_cols)
        else:
            parsed = None
        self._nu, self._demos, n_sims, self._demo_names = self.comm_manager.bcast(parsed)

        # configure and distribute product data
        id_data = self._input_data["id_data"]
        n_obs, n_items = len(id_data["market_id"]), id_data["obs_bundles"].shape[1]
        n_covariates = len(self._specification) 
        self.load_config({
            "dimensions": {
                "n_obs": n_obs, 
                "n_items": n_items, 
                "n_covariates": n_covariates,
                "n_simulations": n_sims,
                "covariate_names": {i: n for i, n in enumerate(self._specification)},
            },
            "subproblem": {"name": "UnitDemand"},
        })
        self.data.load_and_distribute_input_data(self._input_data)
        del self._input_data

        self._log_dims(n_obs, n_items, n_sims, n_covariates, self._specification)

    def build_rc_error_oracle(self, sigma, pi=None, seed=42):
        sigma = np.atleast_2d(sigma)

        market_idx = self.data.local_data["id_data"]["market_id"]
        sim_idx = self.comm_manager.agent_ids // self.config.dimensions.n_obs

        # c_i = Σ ν_i + Π d_i
        c = self._nu[market_idx, sim_idx] @ sigma.T
        if pi is not None and self._demos is not None:
            c += self._demos[market_idx, sim_idx] @ np.atleast_2d(pi).T

        # ε_ij = x_j' c_i + gumbel_ij
        rc_errors = np.einsum('ijk,ik->ij', self._characteristics[market_idx], c)
        gumbel = np.zeros((self.comm_manager.num_local_agent, self.config.dimensions.n_items))
        for i, id in enumerate(self.comm_manager.agent_ids):
            gumbel[i] = np.random.default_rng((seed, id)).gumbel(0, 1, self.config.dimensions.n_items)

        self.features.local_modular_errors = rc_errors + gumbel
        self.features._error_oracle = lambda bundles, ids: (self.features.local_modular_errors[ids] * bundles).sum(-1)
        self.features._error_oracle_vectorized = True
        self.features._error_oracle_takes_data = False

        if self.comm_manager.is_root():
            dim_nu = sigma.shape[1]
            n_demo = self._demos.shape[-1] if self._demos is not None else 0
            self._log_rc(dim_nu, n_demo, self._demo_names)

    def _log_dims(self, n_obs, n_items, n_sims, n_covariates, names):
        if not self.comm_manager.is_root():
            return
        n_markets = len(np.unique(self.data.local_data["id_data"]["market_id"]))
        header = f"{'n_markets':>10} | {'n_obs':>6} | {'n_items':>8} | {'n_simulations':>14} | {'n_covariates':>12}"
        values = f"{n_markets:>10} | {n_obs:>6} | {n_items:>8} | {n_sims:>14} | {n_covariates:>12}"
        logger.info(" PRODUCT DATA")
        logger.info("-" * 65)
        logger.info(header)
        logger.info(values)
        logger.info("-" * 65)
        if names:
            logger.info(" covariates: %s", ", ".join(names))

    def _log_rc(self, dim_nu, n_demographics, demo_names):
        header = f"{'dim_nu':>8} | {'n_demographics':>15}"
        values = f"{dim_nu:>8} | {n_demographics:>15}"
        logger.info(" RANDOM COEFFICIENTS")
        logger.info("-" * 28)
        logger.info(header)
        logger.info(values)
        logger.info("-" * 28)
        if demo_names:
            logger.info(" demographics: %s", ", ".join(demo_names))
