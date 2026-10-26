"""Report per-particle forces alongside OpenMM coordinate trajectories."""

import csv

from openmm.unit import kilojoules_per_mole, nanometer, picosecond


class ForceReporter:
    """Write total potential forces in topology order, with zero-based indices."""

    def __init__(self, file, reportInterval):
        if reportInterval <= 0:
            raise ValueError("reportInterval must be positive")
        self._file = file
        self._reportInterval = reportInterval
        with open(self._file, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "step",
                    "time_ps",
                    "atom_index",
                    "fx_kJ_mol_nm",
                    "fy_kJ_mol_nm",
                    "fz_kJ_mol_nm",
                ]
            )

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "periodic": None, "include": ["forces"]}

    def report(self, simulation, state):
        forces = state.getForces(asNumpy=True).value_in_unit(
            kilojoules_per_mole / nanometer
        )
        time_ps = state.getTime().value_in_unit(picosecond)
        # Close after each frame so completed reports are flushed even if a later step fails.
        with open(self._file, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(
                (simulation.currentStep, time_ps, index, *force)
                for index, force in enumerate(forces)
            )
