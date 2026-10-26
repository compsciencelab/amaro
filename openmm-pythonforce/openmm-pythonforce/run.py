import os

import numpy as np
import openmm as mm
import openmm.app as app
from openmm.app.xtcreporter import XTCReporter
from openmm.unit import femtoseconds, kelvin, nanoseconds, picosecond

from config import get_args
from tmdnet_force import add_python_force
from reporters import ForceReporter
from restraints import add_restraints, write_pdb_with_original_ids
from system import (
    build_full_nnp_system,
    compute_dof,
    configure_periodic_box,
    constrain_hydrogen_bonds,
    repartition_hydrogen_masses,
)


def main():
    outdir, args = get_args()
    steps = args.steps
    output_freq = args.output_freq
    time_step = args.time_step * femtoseconds
    temperature = args.temperature * kelvin
    friction = args.friction_coeff / picosecond
    outname = args.outname
    model_ckpt = os.path.abspath(args.model_ckpt)

    print(f"noh-filter-str: {args.noh}")
    system, top, species_ids, pos, box_vectors = build_full_nnp_system(
        args.pdb,
        psf=args.psf,
        init_coords=args.init_coords,
        noh_mode=args.noh,
        embedding=args.embedding,
        use_increased_masses=args.use_increased_masses,
    )
    periodic = configure_periodic_box(system, top, args, box_vectors)

    if args.seed is not None:
        np.random.seed(args.seed)
    if args.constrainHBonds:
        n_constraints = constrain_hydrogen_bonds(system, top, pos)
        print(f"Added {n_constraints} hydrogen-bond constraint(s).")
    if args.hydrogen_mass is not None:
        n_repartitioned = repartition_hydrogen_masses(system, top, args.hydrogen_mass)
        print(f"Repartitioned mass for {n_repartitioned} hydrogen atom(s).")

    add_python_force(system, species_ids, model_ckpt, args, periodic)
    restraint_info = {}
    if args.k_harmonic > 0 or args.k_flat_bottom > 0:
        restraint_info = add_restraints(system, args, pos)
    system.addForce(mm.CMMotionRemover())

    with open(os.path.join(outdir, "params.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"steps: {steps}\n")
        handle.write(f"output_freq: {output_freq}\n")
        handle.write(f"time_step: {time_step}\n")
        handle.write(f"temperature: {temperature}\n")
        handle.write(f"friction: {friction}\n")
        handle.write(f"model_ckpt: {model_ckpt}\n")
        handle.write(f"model_energy_unit: {args.model_energy_unit}\n")
        handle.write(f"model_length_unit: {args.model_length_unit}\n")
        handle.write(f"periodic: {periodic}\n")
        handle.write(f"periodic_box: {args.periodic_box}\n")
        handle.write(f"periodic_box_unit: {args.periodic_box_unit}\n")
        handle.write(f"coulomb_cutoff_nm: {args.coulomb_cutoff}\n")
        handle.write(f"reaction_field_dielectric: {args.reaction_field_dielectric}\n")
        handle.write(f"include_atomref_energy: {args.include_atomref_energy}\n")
        handle.write(f"seed: {args.seed}\n")
        handle.write(f"device: {args.device}\n")
        handle.write(f"platform: {args.platform}\n")
        handle.write(f"compile: {args.compile}\n")
        handle.write(f"constrainHBonds: {args.constrainHBonds}\n")
        handle.write(f"hydrogen_mass: {args.hydrogen_mass}\n")
        for key, value in restraint_info.items():
            handle.write(f"{key}: {value}\n")
        handle.write(f"dof: {compute_dof(system)}\n")

    nvt_integrator = mm.LangevinMiddleIntegrator(temperature, friction, time_step)
    if args.seed is not None:
        nvt_integrator.setRandomNumberSeed(args.seed)

    platform_properties = {"Precision": "single"} if args.platform == "CUDA" else {}
    if args.seed is not None and args.platform == "CUDA":
        platform_properties["DeterministicForces"] = "true"
    simulation = app.Simulation(
        top,
        system,
        nvt_integrator,
        platform=mm.Platform.getPlatformByName(args.platform),
        platformProperties=platform_properties,
    )
    simulation.context.setPositions(pos)

    if args.minimize_steps is not None and args.minimize_steps > 0:
        print("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=args.minimize_steps)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        write_pdb_with_original_ids(
            top,
            state.getPositions(),
            os.path.join(outdir, f"{outname}_structure_minimized.pdb"),
        )
    else:
        print("Skipping minimization (minimize_steps <= 0).")

    if args.seed is None:
        simulation.context.setVelocitiesToTemperature(temperature)
    else:
        simulation.context.setVelocitiesToTemperature(temperature, args.seed)

    report_args = dict(
        step=True,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
        progress=False,
        remainingTime=True,
        kineticEnergy=True,
        totalSteps=steps,
    )
    if args.steps is None:
        report_args.pop("progress")
        report_args.pop("remainingTime")
        report_args["totalSteps"] = None
        report_args["elapsedTime"] = True
    simulation.reporters.append(
        app.StateDataReporter(
            os.path.join(outdir, f"nvt_metrics_{outname}.csv"),
            output_freq,
            **report_args,
        )
    )
    if XTCReporter is None:
        raise RuntimeError("XTCReporter is not available in this OpenMM installation")
    simulation.reporters.append(
        XTCReporter(
            os.path.join(outdir, f"traj_{outname}.xtc"),
            output_freq,
            enforcePeriodicBox=False,
        )
    )

    # report forces to a separate file
    simulation.reporters.append(
        ForceReporter(os.path.join(outdir, f"forces_{outname}.csv"), output_freq)
    )

    if args.steps is not None:
        time_ns = steps * time_step.value_in_unit(nanoseconds)
        duration = (
            f"{time_ns:.2f} ns" if time_ns >= 0.01 else f"{time_ns * 1000:.2f} ps"
        )
        print(f"Running simulation for {steps} steps ({duration})...")
        simulation.step(steps)
    else:
        print(f"Running simulation for {args.comp_time}h computational time...")
        simulation.runForClockTime(time=args.comp_time)
    if args.save_final_state:
        print("Saving final state to .xml...")
        simulation.saveState(os.path.join(outdir, f"{outname}_final_state.xml"))


if __name__ == "__main__":
    main()
