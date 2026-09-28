from deflation_example.coupled_projected_example import run


def test_self_contained_projected_example():
    report, fields = run()
    assert report["status"] == "verified", report
    assert report["state_maximum_difference"] < 2e-7
    assert fields["control"].shape == fields["state"].shape
