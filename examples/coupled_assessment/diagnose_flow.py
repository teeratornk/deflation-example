"""Inspect pointwise divergence and the discrete kinetic-energy balance."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from deflation_example.assess_transformer import (
    checked_restart,
    import_velocity,
    load_controls,
    transformer_boundaries,
)
from deflation_example.axisymmetric_flow import AxisymmetricFlow
from deflation_example.oil_properties import momentum_reference
from deflation_example.reporting import environment, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(1):
        inputs = load_controls(args.controls_directory, "nominal", 0)
        properties = momentum_reference()
        flow = AxisymmetricFlow(inputs.mesh, properties["kinematic_viscosity_m2_s"])
        fixed, values = transformer_boundaries(
            flow, inputs.parameters["physical"]["inlet_velocity_m_s"]
        )
        current, factor, origin = checked_restart(
            args.checkpoint_directory,
            flow,
            fixed,
            values,
            properties,
            inputs.manifest["input_sha256"],
            0,
            1e-9,
        )
        original = import_velocity(flow, inputs.velocity)
        report = {
            "scope": "flow-initialization diagnostics; intermediate viscosity differs from the physical assessment value",
            "environment": environment(),
            "origin": origin,
            "original_prescribed_at_physical_viscosity": flow.kinetic_diagnostics(original),
        }
        flow.viscosity *= factor
        report["continuation_state_at_declared_viscosity"] = flow.kinetic_diagnostics(
            current.velocity
        )
        write_report(args.output, report)
    print(args.output.name)


if __name__ == "__main__":
    main()
