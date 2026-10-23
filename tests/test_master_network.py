"""Axis isolation, missingness and direct-horizon tests for MASTER's core."""
import unittest

import torch

from ts_benchmark.baselines.master.network import MasterNetwork


class MasterNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(314)
        self.values = torch.randn(3, 7, 5, 4)
        self.market = torch.randn(3, 7, 2)
        self.observed = torch.rand(3, 7, 5, 4) > .15
        self.market_observed = torch.rand(3, 7, 2) > .15

    def model(self, **kwargs):
        config = dict(n_asset_features=4, n_market_features=2, lookback=7,
                      horizon=5, d_model=16, t_heads=4, s_heads=2, dropout=.1)
        config.update(kwargs)
        return MasterNetwork(**config)

    def test_multi_horizon_shapes_and_finite_gradients(self):
        for horizon in (1, 5, 25):
            with self.subTest(horizon=horizon):
                model = self.model(horizon=horizon)
                output = model(self.values, self.market, self.observed, self.market_observed)
                self.assertEqual(tuple(output.shape), (3, horizon, 5))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_batch_origins_do_not_attend_to_each_other(self):
        model = self.model().eval()
        together = model(self.values, self.market, self.observed, self.market_observed)
        separate = torch.cat([
            model(self.values[i:i+1], self.market[i:i+1],
                  self.observed[i:i+1], self.market_observed[i:i+1])
            for i in range(3)
        ])
        torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
        changed = self.values.clone()
        changed[1:] += 100
        torch.testing.assert_close(together[0], model(changed, self.market, self.observed,
                                                    self.market_observed)[0])

    def test_assets_are_permutation_equivariant(self):
        model = self.model().eval()
        permutation = torch.tensor([3, 0, 4, 1, 2])
        expected = model(self.values, self.market, self.observed, self.market_observed)
        permuted = model(self.values[:, :, permutation], self.market,
                         self.observed[:, :, permutation], self.market_observed)
        torch.testing.assert_close(permuted, expected[:, :, permutation], atol=1e-6, rtol=1e-5)

    def test_false_observations_ignore_even_nan_payloads(self):
        model = self.model().eval()
        expected = model(self.values, self.market, self.observed, self.market_observed)
        missing_assets = self.values.masked_fill(~self.observed, float("nan"))
        missing_market = self.market.masked_fill(~self.market_observed, float("nan"))
        actual = model(missing_assets, missing_market, self.observed, self.market_observed)
        torch.testing.assert_close(actual, expected)

    def test_missingness_can_be_learned_separately_from_numeric_values(self):
        model = self.model().eval()
        with torch.no_grad():
            model.asset_missing.weight.fill_(.2)
            model.asset_missing.weight[0].fill_(.7)
        values = torch.zeros_like(self.values)
        all_observed = torch.ones_like(values, dtype=torch.bool)
        no_observed = torch.zeros_like(all_observed)
        self.assertFalse(torch.allclose(model(values, self.market, all_observed),
                                        model(values, self.market, no_observed)))

    def test_gate_is_shared_feature_distribution_and_uses_last_market_row(self):
        model = self.model().eval()
        weights = model.feature_weights(self.market)
        self.assertEqual(tuple(weights.shape), (3, 4))
        torch.testing.assert_close(weights.sum(-1), torch.full((3,), 4.0))
        self.assertTrue((weights > 0).all())
        changed_history = self.market.clone()
        changed_history[:, :-1] = 10000
        torch.testing.assert_close(model(self.values, changed_history), model(self.values, self.market))

    def test_market_ablation_has_identity_gate(self):
        model = self.model(n_market_features=0).eval()
        market = torch.empty(3, 7, 0)
        self.assertIsNone(model.market_gate)
        torch.testing.assert_close(model.feature_weights(market), torch.ones(3, 4))
        self.assertEqual(tuple(model(self.values, market).shape), (3, 5, 5))

    def test_long_lookback_and_odd_embedding_width(self):
        model = self.model(lookback=256, d_model=9, t_heads=3, s_heads=3)
        output = model(torch.randn(1, 256, 2, 4), torch.randn(1, 256, 2))
        self.assertEqual(tuple(output.shape), (1, 5, 2))

    def test_invalid_architecture_is_rejected(self):
        for override in (dict(horizon=0), dict(lookback=True), dict(d_model=15),
                         dict(n_asset_features=0), dict(n_market_features=-1),
                         dict(dropout=1), dict(dropout=float("nan")), dict(gate_beta=0)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.model(**override)

    def test_invalid_shapes_and_masks_are_rejected(self):
        model = self.model()
        for asset, market in ((self.values[:, :-1], self.market),
                              (self.values[:, :, :, :3], self.market),
                              (self.values, self.market[:2]),
                              (self.values, self.market[:, :-1]),
                              (self.values[:, :, :0], self.market)):
            with self.subTest(asset=asset.shape, market=market.shape), self.assertRaises(ValueError):
                model(asset, market)
        with self.assertRaisesRegex(ValueError, "boolean"):
            model(self.values, self.market, self.observed.float())
        with self.assertRaisesRegex(ValueError, "floating point"):
            model(self.values.long(), self.market)


if __name__ == "__main__":
    unittest.main()
