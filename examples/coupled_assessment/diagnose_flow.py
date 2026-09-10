"""Inspect pointwise divergence and the discrete kinetic-energy balance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from deflation_example.assess_transformer import (
    import_velocity,
    load_controls,
    transformer_boundaries,
)
from deflation_example.axisymmetric_flow import AxisymmetricFlow, FlowResult
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
        record = json.loads((args.checkpoint_directory / "record.json").read_text())
        checkpoint = json.loads(
            (args.checkpoint_directory / "baseline-checkpoint.json").read_text()
        )
        config = record["configuration"]
        inputs = load_controls(args.controls_directory, "nominal", config["level"])
        properties = record["momentum_properties"]
        if (
            checkpoint["momentum_properties"] != properties
            or checkpoint["input_sha256"] != inputs.manifest["input_sha256"]
            or checkpoint["level"] != config["level"]
        ):
            raise ValueError("Checkpoint inputs differ from the assessment record")
        factor = checkpoint["stage"].get("viscosity_factor", 1.0)
        flow = AxisymmetricFlow(
            inputs.mesh,
            properties["kinematic_viscosity_m2_s"],
            convection_form=config.get("convection_form", "advective"),
            grad_div=record.get("grad_div_coefficient_m2_s", 0.0),
        )
        fixed, values = transformer_boundaries(
            flow, inputs.parameters["physical"]["inlet_velocity_m_s"]
        )
        field = args.checkpoint_directory / checkpoint.get("file", "baseline-checkpoint.npz")
        if field.parent.resolve() != args.checkpoint_directory.resolve():
            raise ValueError("Checkpoint fields must lie inside their record directory")
        digest = hashlib.sha256(field.read_bytes()).hexdigest()
        if "field_sha256" in checkpoint and digest != checkpoint["field_sha256"]:
            raise ValueError("Checkpoint field checksum differs")
        with np.load(field, allow_pickle=False) as data:
            current = FlowResult(data["velocity"], data["pressure"], "diagnostic", [])
        original = import_velocity(flow, inputs.velocity)
        report = {
            "scope": "Flow-initialization diagnostics. Temporal-step convergence and intermediate-viscosity convergence are recorded separately from the physical steady residual.",
            "environment": environment(),
            "checkpoint_field_sha256": digest,
            "checkpoint_stage": checkpoint["stage"],
            "convection_form": flow.convection_form,
            "grad_div_coefficient_m2_s": flow.grad_div,
            "level": config["level"],
            "viscosity_factor": factor,
            "original_prescribed_at_physical_viscosity": flow.kinetic_diagnostics(original),
            "physical_steady_residuals": flow.verify(
                current, np.zeros_like(flow.quadrature_points), fixed, values
            ),
        }
        flow.viscosity *= factor
        report["checkpoint_at_declared_viscosity"] = flow.kinetic_diagnostics(current.velocity)
        write_report(args.output, report)
    print(args.output.name)


if __name__ == "__main__":
    main()
