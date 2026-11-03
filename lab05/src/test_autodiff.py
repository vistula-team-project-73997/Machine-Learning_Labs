"""Independent checks for reuse, edge cases and full-model parameter gradients."""
import unittest
import numpy as np
import torch
from autodiff import Value


class AutodiffTests(unittest.TestCase):
    def test_reused_node_and_repeated_edges(self):
        x = Value(3)
        u = x * x
        y = u * u + u + x
        y.backward()
        self.assertEqual(x.grad, 4 * 3**3 + 2 * 3 + 1)
        y.backward(clear=False)
        self.assertEqual(x.grad, 2 * (4 * 3**3 + 2 * 3 + 1))
        y.backward()
        self.assertEqual(x.grad, 4 * 3**3 + 2 * 3 + 1)

    def test_relu_convention_and_power_zero(self):
        for v, expected in [(-2, 0), (0, 0), (2, 1)]:
            x = Value(v)
            x.relu().backward()
            self.assertEqual(x.grad, expected)
        x = Value(0)
        (x**0).backward()
        self.assertEqual(x.grad, 0)

    def test_long_graph(self):
        x = Value(1)
        y = x
        for _ in range(5000):
            y = y + 1
        y.backward(retain_graph=False)
        self.assertEqual(x.grad, 1)

    def test_mixed_operations(self):
        x = Value(1.7)
        y = ((3-x) / (x+2)).sigmoid().log() + x**-2
        y.backward()
        t = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
        z = torch.sigmoid((3-t)/(t+2)).log() + t**-2
        z.backward()
        self.assertAlmostEqual(y.data, z.item(), places=13)
        self.assertAlmostEqual(x.grad, t.grad.item(), places=13)

    def test_full_model_gradients(self):
        from experiments import initial_weights, scalar_model, torch_model, scalar_loss
        arrays = initial_weights()
        params, forward = scalar_model(arrays)
        model = torch_model(arrays)
        x = np.array([[.2, -.4], [.8, .6], [-.7, .3]])
        y = np.array([0., 1., 0.])
        loss = scalar_loss(forward, x, y)
        loss.backward()
        tl = torch.nn.functional.binary_cross_entropy(
            model(torch.tensor(x)), torch.tensor(y[:, None]))
        tl.backward()
        actual = np.array([p.grad for p in params])
        expected = np.concatenate([p.grad.numpy().ravel() for p in model.parameters()])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(loss.data, tl.item(), places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
