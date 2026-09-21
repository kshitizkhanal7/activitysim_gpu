from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from phase62_batch_worker import release_invocation_state, ResidentMemorySampler, memory_snapshot


def test_explicit_teardown_drops_services_flows_and_arrays_not_programs(monkeypatch):
    service = SimpleNamespace(_SERVICE=object())
    scheduling = SimpleNamespace(_SERVICE=object())
    flow = SimpleNamespace(_FLOWS={"live_input":object()})
    cuda = SimpleNamespace(_COEFFICIENT_CACHE={"device":object()},_HOST_COEFFICIENT_CACHE={"host":object()},
                           _COMPILED_PLAN_CACHE={"workspace":object()},_KERNEL_CACHE={"source":"program"})
    for name,module in (("choiceforge.modelwide_service",service),("choiceforge.activitysim_trip_scheduling",scheduling),
                        ("activitysim.core.flow",flow),("choiceforge.sharrow_cuda",cuda)):
        monkeypatch.setitem(sys.modules,name,module)
    released = release_invocation_state()
    assert service._SERVICE is None and scheduling._SERVICE is None
    assert not flow._FLOWS and not cuda._COEFFICIENT_CACHE and not cuda._COMPILED_PLAN_CACHE
    assert cuda._KERNEL_CACHE=={"source":"program"}
    assert "choiceforge.modelwide_service._SERVICE" in released
    release_invocation_state()  # idempotent even if stale module references survive


def test_memory_sampler_reports_observations_and_stops_thread():
    sampler = ResidentMemorySampler().start()
    result = sampler.finish()
    assert result["observed_peak_rss_bytes"]>0
    assert result["instantaneous_peak_guaranteed"] is False
    assert not sampler.thread.is_alive()


def test_cpu_memory_measurement_does_not_initialize_cuda(monkeypatch):
    class NoGPU:
        def __getattr__(self,name):
            raise AssertionError("CPU telemetry touched CUDA")
    monkeypatch.setitem(sys.modules,"cupy",NoGPU())
    snapshot = memory_snapshot(False)
    assert snapshot["uss_bytes"]>0 and "gpu_pool_used_bytes" not in snapshot
