import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.models import VisionMambaCPUFull, vim_tiny_cpu_full


class VisionMambaCPUFullTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = vim_tiny_cpu_full(pretrained=False, num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))

    def test_checkpoint_roundtrip(self) -> None:
        torch.manual_seed(42)
        model = vim_tiny_cpu_full(pretrained=False, num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            expected = model(x)

        with TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "vim_cpu_full_roundtrip.pth"
            model.save_checkpoint(ckpt_path, extra={"note": "roundtrip"})
            restored = VisionMambaCPUFull.from_checkpoint(ckpt_path, map_location="cpu")
            restored.eval()

            with torch.no_grad():
                actual = restored(x)

        torch.testing.assert_close(actual, expected)

    def test_backward_pass(self) -> None:
        model = vim_tiny_cpu_full(pretrained=False, num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 10, (2,))

        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        grad_tensors = [param.grad for param in model.parameters() if param.requires_grad]
        self.assertTrue(any(grad is not None for grad in grad_tensors))


if __name__ == "__main__":
    unittest.main()
