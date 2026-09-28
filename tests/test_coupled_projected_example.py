import pytest

from deflation_example.coupled_projected_example import run


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("hybrid", marks=pytest.mark.gpu),
    ],
)
def test_self_contained_projected_example(device):
    report, fields = run(device)
    assert report["status"] == "verified", report
    assert report["state_maximum_difference"] < 2e-7
    assert fields["control"].shape == fields["state"].shape
