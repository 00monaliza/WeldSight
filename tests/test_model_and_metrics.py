import numpy as np
import torch

from weldsight.eval.metrics import SegConfusion, detection_metrics
from weldsight.eval.referral import referral_curve, summarize_curve
from weldsight.eval.uncertainty import cls_uncertainty
from weldsight.models.losses import SegClsLoss
from weldsight.models.unet import WeldUNet, enable_mc_dropout


def _model():
    torch.manual_seed(0)
    return WeldUNet(3, "resnet18", None, 1, 0.3, 0.3, 0.3)


def test_forward_shapes_and_loss():
    m = _model()
    x = torch.randn(2, 1, 256, 256)
    seg, cls = m(x)
    assert seg.shape == (2, 4, 256, 256) and cls.shape == (2, 3)
    masks = torch.stack([torch.randint(0, 4, (256, 256)), torch.full((256, 256), -1)])
    loss = SegClsLoss()(
        seg, cls, masks, torch.tensor([[1.0, 0, 0], [0, 0, 0]]), torch.tensor([True, False])
    )
    assert torch.isfinite(loss["total"]) and loss["dice"] > 0
    loss["total"].backward()


def test_mc_dropout_is_stochastic_but_eval_is_not():
    m = _model()
    x = torch.randn(2, 1, 256, 256)
    m.eval()
    with torch.no_grad():
        a, b = m(x)[1], m(x)[1]
    assert torch.allclose(a, b)
    enable_mc_dropout(m)
    with torch.no_grad():
        seg, cls = m.forward_mc(x, T=4)
    assert seg.shape == (4, 2, 4, 256, 256) and cls.shape == (4, 2, 3)
    assert cls.std(0).mean() > 0
    # BatchNorm must stay in eval mode under MC dropout
    assert not any(mod.training for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d))


def test_uncertainty_decomposition():
    certain = np.full((10, 5, 3), 0.99)
    disagree = np.concatenate([np.full((5, 5, 3), 0.01), np.full((5, 5, 3), 0.99)])
    assert (
        cls_uncertainty(disagree)["mutual_information"].mean()
        > cls_uncertainty(certain)["mutual_information"].mean()
    )


def test_iou_and_component_recall():
    gt = np.zeros((1, 10, 10), int)
    gt[0, :2, :2] = 1  # component A
    gt[0, 8:, 8:] = 1  # component B (missed)
    pred = np.zeros_like(gt)
    pred[0, :2, :4] = 1
    c = SegConfusion(2)
    c.update(pred, gt)
    assert np.isclose(c.iou()[1], 4 / 12)
    assert np.isclose(c.component_recall()[1], 0.5)


def test_detection_metrics():
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    p = np.array([[0.9, 0.1], [0.2, 0.1], [0.1, 0.8], [0.6, 0.1]])
    d = detection_metrics(y, p, 0.5, ["a", "b"])
    assert d["a"]["recall"] == 0.5 and d["a"]["precision"] == 0.5 and d["b"]["recall"] == 1.0


def test_referral_curve_endpoints_and_monotonic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, 3))
    pred = y * (rng.random((200, 3)) > 0.4)
    unc = rng.random(200)
    groups = np.repeat(np.arange(20), 10)
    df = referral_curve(y, pred, unc, groups, n_random=10)
    for curve in ("model", "random", "oracle"):
        r = (
            df[(df.curve == curve) & (df["class"] == "all")]
            .sort_values("fraction")["recall"]
            .to_numpy()
        )
        assert np.isclose(r[0], (y & pred).sum() / y.sum())
        assert np.isclose(r[-1], 1.0)
        assert np.all(np.diff(r) >= -1e-12)
    s = summarize_curve(df, ["a", "b", "c"])
    assert s["oracle"]["all"]["area"] >= s["model"]["all"]["area"] - 1e-9
