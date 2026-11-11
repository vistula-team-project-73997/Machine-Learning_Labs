"""Independent regression checks for partial accumulation and complete checkpoint recovery."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline import train_epoch, checkpoint, restore
from model import CNN


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)

    def test_partial_accumulation_matches_full_batch(self):
        # 19 examples => groups 8,8,3; final micro-batch has one example.
        data = TensorDataset(torch.randn(19, 3, dtype=torch.float64), torch.randint(0, 2, (19,)))
        a = nn.Linear(3, 2).double()
        b = copy.deepcopy(a)
        oa = torch.optim.SGD(a.parameters(), lr=0.03, momentum=0.9)
        ob = torch.optim.SGD(b.parameters(), lr=0.03, momentum=0.9)
        ra = train_epoch(a, DataLoader(data, batch_size=8), oa)
        rb = train_epoch(b, DataLoader(data, batch_size=2), ob, accumulation=4)
        self.assertEqual(ra['optimizer_steps'], 3)
        self.assertEqual(rb['optimizer_steps'], 3)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=1e-14, rtol=1e-14)

    def test_restore_recovers_next_update_scheduler_rng_and_shuffle(self):
        random.seed(8)
        np.random.seed(8)
        model = nn.Linear(3, 2).double()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        gen = torch.Generator().manual_seed(7)
        data = TensorDataset(torch.randn(11, 3, dtype=torch.float64), torch.randint(0, 2, (11,)))
        def epoch(m, o, s, g):
            row = train_epoch(m, DataLoader(data, batch_size=4, shuffle=True, generator=g), o)
            s.step()
            return row
        epoch(model, opt, scheduler, gen)
        scratch = Path(__file__).resolve().parents[2] / 'tmp' / 'lab06'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            path = Path(tmp) / 'check.pt'
            checkpoint(path, 1, model, opt, scheduler, gen, [], {}, {}, 'unit')
            expected_rng = (random.random(), np.random.random(), torch.rand(1))
            epoch(model, opt, scheduler, gen)
            expected = copy.deepcopy(model.state_dict())
            expected_lr = opt.param_groups[0]['lr']
            restored = nn.Linear(3, 2).double()
            ro = torch.optim.Adam(restored.parameters(), lr=0.99)
            rs = torch.optim.lr_scheduler.StepLR(ro, step_size=1, gamma=0.5)
            rg = torch.Generator()
            restore(path, restored, ro, rs, rg)
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.random(), expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(1), expected_rng[2]))
            epoch(restored, ro, rs, rg)
            self.assertEqual(ro.param_groups[0]['lr'], expected_lr)
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)

    def test_lab3_architecture(self):
        model = CNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20490)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 28, 28)).shape), (2, 10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
