"""Unit tests for the opt-in ExogenousLabourForcePath demography and the workforce-entry
hooks. Also pins NoAging (default) as an exact no-op for backward compatibility."""

import numpy as np

from macromodel.agents.individuals.func.demography import ExogenousLabourForcePath, NoAging
from macromodel.agents.individuals.individual_properties import ActivityStatus


def _activity(n_emp, n_nea, n_unemp):
    return np.array(
        [ActivityStatus.EMPLOYED] * n_emp
        + [ActivityStatus.NOT_ECONOMICALLY_ACTIVE] * n_nea
        + [ActivityStatus.UNEMPLOYED] * n_unemp,
        dtype=object,
    )


class TestNoAgingDefault:
    def test_noaging_population_constant_and_activity_untouched(self):
        n = NoAging()
        assert n.update(1000.0) == 1000.0
        act = _activity(3, 2, 1)
        before = act.copy()
        n.individuals_joining_the_workforce(current_individuals_activity=act)
        n.individuals_leaving_the_workforce(current_individuals_activity=act)
        assert (act == before).all()


class TestExogenousLabourForcePath:
    def test_inert_without_a_path(self):
        e = ExogenousLabourForcePath()  # no labour_force_index -> behaves like NoAging
        act = _activity(3, 2, 1)
        before = act.copy()
        e.update(1000.0)
        e.individuals_joining_the_workforce(current_individuals_activity=act)
        e.individuals_leaving_the_workforce(current_individuals_activity=act)
        assert (act == before).all()

    def test_entry_reclassifies_nea_to_unemployed_reproducibly(self):
        # +50% labour force at step 1: NEA -> UNEMPLOYED; employed never touched.
        act = _activity(8, 10, 2)  # labour force = 8 + 2 = 10 (investors excluded, none here)
        e = ExogenousLabourForcePath(labour_force_index=[1.0, 1.5], seed=7)
        for _ in range(2):
            e.update(0.0)
            e.individuals_joining_the_workforce(current_individuals_activity=act)
            e.individuals_leaving_the_workforce(current_individuals_activity=act)
        log = e.log[-1]
        assert log["labour_force"] == 15  # 10 -> 15
        assert log["target_labour_force"] == 15
        assert log["entries"] == 5
        assert int((act == ActivityStatus.EMPLOYED).sum()) == 8  # employed untouched

    def test_declining_path_never_removes_employed(self):
        # -50% target with only 2 unemployed: exits capped at 2, shortfall logged,
        # employed workers never reclassified.
        act = _activity(8, 0, 2)  # labour force = 10, wants to shed 5, only 2 unemployed
        e = ExogenousLabourForcePath(labour_force_index=[1.0, 0.5], seed=7)
        for _ in range(2):
            e.update(0.0)
            e.individuals_joining_the_workforce(current_individuals_activity=act)
            e.individuals_leaving_the_workforce(current_individuals_activity=act)
        log = e.log[-1]
        assert log["exits"] == 2
        assert log["exit_shortfall"] == 3
        assert int((act == ActivityStatus.EMPLOYED).sum()) == 8  # employed preserved
