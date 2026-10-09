"""CPU checks for the student's NB0 implementation (no GPU or model download)."""
import ast
import math
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / "notebooks/00_dpo_loss_from_scratch.py").read_text(encoding="utf-8"))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "my_dpo_loss")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[function], type_ignores=[]), "NB0", "exec"), namespace)
loss = namespace["my_dpo_loss"]
c = torch.tensor([-20.0, -35.0])
r = c - 3
initial = loss(c, r, c, r)
assert initial is not None, "my_dpo_loss is not implemented"
assert torch.allclose(initial, torch.tensor(math.log(2)), atol=1e-6)
pc = torch.tensor([-12.0, -30.0], requires_grad=True)
pr = torch.tensor([-15.0, -28.0])
rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
expected = -torch.nn.functional.logsigmoid(0.1 * ((pc - rc) - (pr - rr))).mean()
assert torch.allclose(loss(pc, pr, rc, rr), expected, atol=1e-6)
loss(pc, pr, rc, rr).backward()
assert pc.grad is not None and torch.isfinite(pc.grad).all() and (pc.grad < 0).all()
refc, refr = torch.tensor([-20.0]), torch.tensor([-22.0])
assert torch.allclose(loss(refc + 1, refr - 1, refc, refr, beta=1), loss(refc - 3, refr - 5, refc, refr, beta=1))
print("PASS NB0: log(2), reference formula, finite gradients, likelihood displacement")
