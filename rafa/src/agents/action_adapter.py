# -*- coding: utf-8 -*-
"""
Action adapter: converts raw policy outputs to headway scaling factors.

Implements the residual formulation from formulation.tex:
    alpha_i[t] = clip(alpha_rule[t] + Delta_alpha, alpha_min, alpha_max)
"""

from typing import Dict

import numpy as np

from src.agents.rl_types import RLConfig


class ActionAdapter:
    """Translates RL residual actions into adapted headway scaling factors."""

    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

    def apply(
        self,
        delta_alphas: Dict[int, float],
        alpha_rules: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Combine rule-based baseline with RL residual.

        Parameters
        ----------
        delta_alphas : dict[int, float]
            RL residual per CAV id.  Values in [-delta_alpha_max, delta_alpha_max].
        alpha_rules : dict[int, float]
            Mesoscopic rule-based alpha per CAV id.

        Returns
        -------
        dict[int, float]
            Final alpha per CAV id, in [alpha_min, alpha_max].
        """
        result: Dict[int, float] = {}
        for cav_id, da in delta_alphas.items():
            a_rule = alpha_rules.get(cav_id, 1.0)
            alpha = np.clip(
                a_rule + da,
                self.cfg.alpha_min,
                self.cfg.alpha_max,
            )
            result[cav_id] = float(alpha)
        return result

    @staticmethod
    def zero_residual(cav_ids: list) -> Dict[int, float]:
        """Return zero residuals (for baseline equivalence testing)."""
        return {cid: 0.0 for cid in cav_ids}
