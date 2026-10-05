import numpy as np
import pytest

from leo_routing.resource.allocator import allocate_capacity


def test_closed_form_kkt_and_budget():
    workloads = {0: 1.0, 1: 4.0, 2: 9.0}
    rates = allocate_capacity(workloads, 60.0)
    assert rates == pytest.approx({0: 10.0, 1: 20.0, 2: 30.0})
    assert sum(rates.values()) == pytest.approx(60.0)
    duals = [workloads[key] / rates[key] ** 2 for key in workloads]
    assert np.ptp(duals) < 1e-12


def test_weighted_solution_and_equal_mode():
    assert allocate_capacity({0: 1.0, 1: 1.0}, 30.0, weights={0: 1.0, 1: 4.0}) == pytest.approx({0: 10, 1: 20})
    assert allocate_capacity({0: 1.0, 1: 9.0}, 30.0, "equal") == pytest.approx({0: 15, 1: 15})
    assert allocate_capacity({}, 1.0) == {}


@pytest.mark.parametrize("workload,capacity", [({0: 0.0}, 10.0), ({0: -1.0}, 10.0),
                                              ({0: float('nan')}, 10.0), ({0: 1.0}, 0.0)])
def test_invalid_allocations_rejected(workload, capacity):
    with pytest.raises(ValueError):
        allocate_capacity(workload, capacity)
