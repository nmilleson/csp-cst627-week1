"""
Gradient-checking harness for a manually-backpropagated 2-layer MLP.

Northwind Diagnostics triage classifier: verifies a hand-derived backward
pass against PyTorch autograd and central finite differences, so a bug in
the manual backward pass shows up as a large per-parameter relative error
instead of a mysteriously rising validation loss.

Architecture: Linear(D->H) -> ReLU -> Linear(H->1) -> Sigmoid -> BCE loss.
"""

import torch

torch.manual_seed(0)

# Finite-difference step size: optimal h for central differences balances
# truncation error O(h^2) against float rounding error O(eps/h), giving
# h ~ eps^(1/3). Done in float64 (eps ~2.22e-16) so h=1e-6 keeps both error
# terms far below the thresholds we check against.
FD_STEP = 1e-6
REL_ERROR_EPS_FLOOR = 1e-8
WARN_THRESHOLD = 1e-5
FAIL_THRESHOLD = 1e-3


def make_batch(n, d, dtype=torch.float64):
    X = torch.randn(n, d, dtype=dtype)
    y = (torch.rand(n, 1, dtype=dtype) > 0.5).to(dtype)
    return X, y


def init_params(d, h, out=1, dtype=torch.float64):
    # small-scale init; only correctness matters here, not training dynamics
    W1 = torch.randn(d, h, dtype=dtype) * 0.5
    b1 = torch.zeros(h, dtype=dtype)
    W2 = torch.randn(h, out, dtype=dtype) * 0.5
    b2 = torch.zeros(out, dtype=dtype)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = torch.sigmoid(z2)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
    return y_hat, cache


def bce_loss(y_hat, y, eps=1e-12):
    y_hat = y_hat.clamp(eps, 1 - eps)
    return -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)).mean()


def manual_backward(cache, y, params):
    """Hand-derived gradients via chain rule. No autograd involved."""
    X, z1, a1, y_hat = cache["X"], cache["z1"], cache["a1"], cache["y_hat"]
    W2 = params["W2"]
    n = X.shape[0]

    # d(BCE + sigmoid)/dz2 simplifies to (y_hat - y); divide by n for the mean
    dz2 = (y_hat - y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(dim=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0).to(z1.dtype)  # ReLU'
    dW1 = X.T @ dz1
    db1 = dz1.sum(dim=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def autograd_backward(X, y, params):
    """Same computation, but let autograd compute the gradients (ground truth)."""
    ag_params = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    y_hat, _ = forward(X, ag_params)
    loss = bce_loss(y_hat, y)
    loss.backward()
    return {k: v.grad.clone() for k, v in ag_params.items()}, loss.item()


def numeric_grad(param_name, X, y, params, h=FD_STEP):
    """Central finite-difference gradient, elementwise, for one parameter tensor."""
    base = params[param_name]
    grad = torch.zeros_like(base)
    flat_base = base.view(-1)
    flat_grad = grad.view(-1)

    for i in range(flat_base.numel()):
        orig = flat_base[i].item()

        flat_base[i] = orig + h
        y_hat_plus, _ = forward(X, params)
        loss_plus = bce_loss(y_hat_plus, y).item()

        flat_base[i] = orig - h
        y_hat_minus, _ = forward(X, params)
        loss_minus = bce_loss(y_hat_minus, y).item()

        flat_base[i] = orig  # restore
        flat_grad[i] = (loss_plus - loss_minus) / (2 * h)

    return grad


def relative_error(a, b, eps_floor=REL_ERROR_EPS_FLOOR):
    return (a - b).abs() / torch.clamp(torch.max(a.abs(), b.abs()), min=eps_floor)


def run_gradcheck(n=4, d=6, h=4):
    params = init_params(d, h)
    X, y = make_batch(n, d)

    y_hat, cache = forward(X, params)
    manual_grads = manual_backward(cache, y, params)
    autograd_grads, loss_val = autograd_backward(X, y, params)

    print(f"Batch size={n}, input_dim={d}, hidden_dim={h}, loss={loss_val:.6f}")
    print(f"FD step h={FD_STEP}, dtype=float64\n")

    results = {}
    header = f"{'param':6} {'max|analytic-autograd|':>24} {'FD vs autograd (max rel err)':>30} {'FD vs autograd (mean rel err)':>30} {'status':>8}"
    print(header)
    print("-" * len(header))

    overall_status = "PASS"
    for name in ["W1", "b1", "W2", "b2"]:
        analytic = manual_grads[name]
        ag = autograd_grads[name]

        # Sanity: manual analytic backward should match autograd almost exactly
        analytic_vs_ag = (analytic - ag).abs().max().item()

        numeric = numeric_grad(name, X, y, params)
        rel_err = relative_error(numeric, ag)
        max_rel = rel_err.max().item()
        mean_rel = rel_err.mean().item()

        if max_rel > FAIL_THRESHOLD:
            status = "FAIL"
            overall_status = "FAIL"
        elif max_rel > WARN_THRESHOLD:
            status = "WARN"
            if overall_status == "PASS":
                overall_status = "WARN"
        else:
            status = "PASS"

        results[name] = {
            "analytic_vs_autograd_max_abs": analytic_vs_ag,
            "fd_vs_autograd_max_rel": max_rel,
            "fd_vs_autograd_mean_rel": mean_rel,
            "status": status,
        }
        print(f"{name:6} {analytic_vs_ag:24.3e} {max_rel:30.3e} {mean_rel:30.3e} {status:>8}")

    print(f"\nOverall: {overall_status}")
    return results, overall_status


if __name__ == "__main__":
    run_gradcheck()
