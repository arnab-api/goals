"""Load and query results from adversarial_persistence.py summary.json files."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class AdversarialPersistenceResults:
    path: str   # path to summary.json
    data: dict  # raw loaded dict

    @staticmethod
    def load(path: str) -> "AdversarialPersistenceResults":
        with open(path) as f:
            data = json.load(f)
        return AdversarialPersistenceResults(path=path, data=data)

    def best_hparams(self, direction_type: str = "contrastive") -> dict | None:
        """Overall best (prefill, layer_idx, window, coeff, pass_rate).

        Reads ``overall_best_contrastive`` or ``overall_best_mean_diff`` from
        the summary, which are set to the best result across all prefills.
        """
        key = f"overall_best_{direction_type}"
        return self.data.get(key)

    def best_hparams_for_prefill(
        self, prefill: str, direction_type: str = "contrastive"
    ) -> dict | None:
        """Best (layer_idx, window, coeff, pass_rate) for a specific prefill.

        Args:
            prefill: The prefill string (must match a key in prefill_results).
            direction_type: "contrastive" or "mean_diff".

        Returns:
            {"layer_idx": int, "window": list[int], "coeff": str, "pass_rate": float}
            or None if the prefill is not found or all runs errored.
        """
        pfr = self.data["prefill_results"].get(prefill)
        if pfr is None:
            return None
        key = f"best_{direction_type}"
        return pfr.get(key)

    def best_coeff_for_layer(
        self, layer_idx: int, direction_type: str = "contrastive"
    ) -> dict:
        """Best coeff for a given layer_idx, ranked by mean pass rate across prefills.

        Returns:
            {
                "layer_idx": int,
                "window": list[int] | None,
                "coeff": str | None,
                "mean_pass_rate": float | None,
                "per_prefill": {prefill: pass_rate, ...}
            }
        """
        layer_str = str(layer_idx)
        by_coeff_key = f"{direction_type}_by_coeff"

        # For each coeff, collect rates across prefills at this layer
        coeff_scores: dict[str, list[float]] = {}
        window: list[int] | None = None

        for pfr in self.data["prefill_results"].values():
            layer_data = pfr["by_layer"].get(layer_str)
            if layer_data is None:
                continue
            if window is None:
                window = layer_data["window"]
            for coeff_str, rate in layer_data.get(by_coeff_key, {}).items():
                coeff_scores.setdefault(coeff_str, []).append(rate)

        if not coeff_scores:
            return {
                "layer_idx": layer_idx,
                "window": window,
                "coeff": None,
                "mean_pass_rate": None,
                "per_prefill": {},
            }

        mean_by_coeff = {c: sum(r) / len(r) for c, r in coeff_scores.items()}
        best_coeff = max(mean_by_coeff, key=mean_by_coeff.get)
        best_mean = mean_by_coeff[best_coeff]

        per_prefill = {}
        for prefill, pfr in self.data["prefill_results"].items():
            layer_data = pfr["by_layer"].get(layer_str)
            if layer_data is not None:
                rate = layer_data.get(by_coeff_key, {}).get(best_coeff)
                if rate is not None:
                    per_prefill[prefill] = rate

        return {
            "layer_idx": layer_idx,
            "window": window,
            "coeff": best_coeff,
            "mean_pass_rate": best_mean,
            "per_prefill": per_prefill,
        }

    def best_layer_for_coeff(
        self, coeff: str, direction_type: str = "contrastive"
    ) -> dict:
        """Best layer_idx for a given coeff, ranked by mean pass rate across prefills.

        Returns:
            {
                "layer_idx": int | None,
                "window": list[int] | None,
                "coeff": str,
                "mean_pass_rate": float | None,
                "per_prefill": {prefill: pass_rate, ...}
            }
        """
        by_coeff_key = f"{direction_type}_by_coeff"

        # For each layer, collect rates across prefills at this coeff
        layer_scores: dict[str, list[float]] = {}
        layer_windows: dict[str, list[int]] = {}

        for pfr in self.data["prefill_results"].values():
            for layer_str, layer_data in pfr["by_layer"].items():
                rate = layer_data.get(by_coeff_key, {}).get(coeff)
                if rate is not None:
                    layer_scores.setdefault(layer_str, []).append(rate)
                    layer_windows[layer_str] = layer_data["window"]

        if not layer_scores:
            return {
                "layer_idx": None,
                "window": None,
                "coeff": coeff,
                "mean_pass_rate": None,
                "per_prefill": {},
            }

        mean_by_layer = {ls: sum(r) / len(r) for ls, r in layer_scores.items()}
        best_layer_str = max(mean_by_layer, key=mean_by_layer.get)
        best_mean = mean_by_layer[best_layer_str]

        per_prefill = {}
        for prefill, pfr in self.data["prefill_results"].items():
            layer_data = pfr["by_layer"].get(best_layer_str)
            if layer_data is not None:
                rate = layer_data.get(by_coeff_key, {}).get(coeff)
                if rate is not None:
                    per_prefill[prefill] = rate

        return {
            "layer_idx": int(best_layer_str),
            "window": layer_windows[best_layer_str],
            "coeff": coeff,
            "mean_pass_rate": best_mean,
            "per_prefill": per_prefill,
        }

    def results_for(self, layer_idx: int, coeff: str) -> dict:
        """All results (both direction types, per prefill) for a layer_idx-coeff combo.

        Returns:
            {
                "layer_idx": int,
                "window": list[int] | None,
                "coeff": str,
                "contrastive": {prefill: pass_rate, ...},
                "mean_diff": {prefill: pass_rate, ...}
            }
        """
        layer_str = str(layer_idx)
        contrastive_rates: dict[str, float] = {}
        mean_diff_rates: dict[str, float] = {}
        window: list[int] | None = None

        for prefill, pfr in self.data["prefill_results"].items():
            layer_data = pfr["by_layer"].get(layer_str)
            if layer_data is None:
                continue
            if window is None:
                window = layer_data["window"]
            c_rate = layer_data.get("contrastive_by_coeff", {}).get(coeff)
            m_rate = layer_data.get("mean_diff_by_coeff", {}).get(coeff)
            if c_rate is not None:
                contrastive_rates[prefill] = c_rate
            if m_rate is not None:
                mean_diff_rates[prefill] = m_rate

        return {
            "layer_idx": layer_idx,
            "window": window,
            "coeff": coeff,
            "contrastive": contrastive_rates,
            "mean_diff": mean_diff_rates,
        }
