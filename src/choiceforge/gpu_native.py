"""Fail-closed building blocks for a GPU-native travel-model runtime.

The boundary is deliberately explicit: host code may read files, upload an
input partition, launch kernels, and download final outputs.  Between
``seal_ingress`` and ``egress_table``, every modeled array must remain a CUDA
array.  Scalar synchronization needed to launch or validate a kernel is
control-plane work, not an alternative CPU implementation of model logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from .cuda_backend import CudaChoiceBackend, _cupy


class GpuOnlyViolation(RuntimeError):
    """Raised when modeled work attempts to cross the GPU-only boundary."""


@dataclass(frozen=True)
class GpuMemoryBudget:
    """An auditable allocation of device memory, expressed in bytes."""

    total_bytes: int
    reserve_bytes: int
    hot_skims_bytes: int
    persistent_state_bytes: int
    workspace_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.total_bytes,
            self.reserve_bytes,
            self.hot_skims_bytes,
            self.persistent_state_bytes,
            self.workspace_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("memory budget values cannot be negative")
        if self.committed_bytes > self.total_bytes:
            raise ValueError(
                f"memory budget overcommits device: {self.committed_bytes} > {self.total_bytes}"
            )

    @property
    def committed_bytes(self) -> int:
        return (
            self.reserve_bytes
            + self.hot_skims_bytes
            + self.persistent_state_bytes
            + self.workspace_bytes
        )

    @property
    def unallocated_bytes(self) -> int:
        return self.total_bytes - self.committed_bytes

    def max_entities(self, bytes_per_entity: int, *, utilization: float = 0.85) -> int:
        """Return a conservative entity count for the unallocated pool."""

        if bytes_per_entity <= 0:
            raise ValueError("bytes_per_entity must be positive")
        if not 0.0 < utilization <= 1.0:
            raise ValueError("utilization must be in (0, 1]")
        return int((self.unallocated_bytes * utilization) // bytes_per_entity)


def plan_household_partitions(total_households: int, max_per_partition: int) -> list[tuple[int, int]]:
    """Create deterministic, complete, half-open household ranges."""

    if total_households < 0:
        raise ValueError("total_households cannot be negative")
    if max_per_partition <= 0:
        raise ValueError("max_per_partition must be positive")
    return [
        (start, min(start + max_per_partition, total_households))
        for start in range(0, total_households, max_per_partition)
    ]


def _is_cuda_array(value: Any) -> bool:
    return hasattr(value, "__cuda_array_interface__")


@dataclass(frozen=True)
class DeviceTable:
    """A named collection of equal-length CUDA columns."""

    columns: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("a device table must contain at least one column")
        lengths: set[int] = set()
        for name, value in self.columns.items():
            if not name:
                raise ValueError("device column names cannot be empty")
            if not _is_cuda_array(value):
                raise GpuOnlyViolation(f"column {name!r} is not a CUDA array")
            if getattr(value, "ndim", 0) < 1:
                raise ValueError(f"column {name!r} must have at least one dimension")
            lengths.add(int(value.shape[0]))
        if len(lengths) != 1:
            raise ValueError("all device columns must have the same first dimension")

    @property
    def nrows(self) -> int:
        return int(next(iter(self.columns.values())).shape[0])

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.columns.values())


@dataclass
class GpuRuntimeTelemetry:
    input_bytes: int = 0
    output_bytes: int = 0
    modeled_host_to_device_bytes: int = 0
    modeled_device_to_host_bytes: int = 0
    modeled_cpu_fallbacks: int = 0
    kernel_stages: list[str] = field(default_factory=list)


class GpuNativeRuntime:
    """Own GPU state and reject host modeled data after ingress is sealed."""

    def __init__(self) -> None:
        self.cp = _cupy()
        self.tables: dict[str, DeviceTable] = {}
        self.telemetry = GpuRuntimeTelemetry()
        self._sealed = False
        self._choice_backend = CudaChoiceBackend()

    @property
    def sealed(self) -> bool:
        return self._sealed

    def ingress_table(self, name: str, columns: Mapping[str, Any]) -> DeviceTable:
        if self._sealed:
            self.telemetry.modeled_host_to_device_bytes += sum(
                int(np.asarray(value).nbytes) for value in columns.values()
            )
            raise GpuOnlyViolation("host ingress is closed during modeled execution")
        device_columns = {
            column_name: self.cp.ascontiguousarray(self.cp.asarray(value))
            for column_name, value in columns.items()
        }
        table = DeviceTable(device_columns)
        self.tables[name] = table
        self.telemetry.input_bytes += table.nbytes
        return table

    def register_device_table(self, name: str, columns: Mapping[str, Any]) -> DeviceTable:
        table = DeviceTable(columns)
        self.tables[name] = table
        return table

    def seal_ingress(self) -> None:
        self._sealed = True

    def run_stage(
        self,
        stage_name: str,
        operation: Callable[..., Mapping[str, Any] | DeviceTable],
        *args: Any,
        output_table: str | None = None,
        **kwargs: Any,
    ) -> DeviceTable:
        if not self._sealed:
            raise GpuOnlyViolation("seal ingress before running modeled stages")
        result = operation(*args, **kwargs)
        table = result if isinstance(result, DeviceTable) else DeviceTable(result)
        if output_table is not None:
            self.tables[output_table] = table
        self.telemetry.kernel_stages.append(stage_name)
        return table

    def linear_choice(
        self,
        chooser_features: Any,
        coefficients: Any,
        constants: Any,
        uniforms: Any,
    ) -> DeviceTable:
        for label, value in (
            ("chooser_features", chooser_features),
            ("coefficients", coefficients),
            ("constants", constants),
            ("uniforms", uniforms),
        ):
            if not _is_cuda_array(value):
                self.telemetry.modeled_host_to_device_bytes += int(np.asarray(value).nbytes)
                raise GpuOnlyViolation(f"{label} crossed the sealed boundary from host memory")
        result = self._choice_backend.linear_choice(
            chooser_features,
            coefficients,
            constants,
            uniforms,
            return_device=True,
        )
        self.telemetry.kernel_stages.append("fused_linear_choice")
        return DeviceTable({"choice": result.choices, "logsum": result.logsums})

    def cpu_fallback(self, stage_name: str) -> None:
        self.telemetry.modeled_cpu_fallbacks += 1
        raise GpuOnlyViolation(f"CPU fallback is forbidden for modeled stage {stage_name!r}")

    def egress_table(self, table: DeviceTable, columns: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        selected = columns or tuple(table.columns)
        result = {name: self.cp.asnumpy(table.columns[name]) for name in selected}
        self.telemetry.output_bytes += sum(value.nbytes for value in result.values())
        return result

    def assert_gpu_only(self) -> None:
        if (
            self.telemetry.modeled_cpu_fallbacks
            or self.telemetry.modeled_host_to_device_bytes
            or self.telemetry.modeled_device_to_host_bytes
        ):
            raise GpuOnlyViolation(f"GPU-only contract failed: {self.telemetry}")


GPU_NATIVE_CUDA = r"""
extern "C" __device__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void entity_uniform_f32(
    const long long* entity_ids,
    int n,
    unsigned long long seed,
    unsigned long long stream,
    float* output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    unsigned long long key = (unsigned long long)entity_ids[row]
        ^ (seed * 0xD2B74407B1CE6E93ULL)
        ^ (stream * 0xCA5A826395121157ULL);
    unsigned long long value = splitmix64(key);
    // The upper 24 bits are exactly representable as float32.  Half a unit
    // keeps the result strictly inside (0, 1), independent of partitioning.
    output[row] = ((float)(value >> 40) + 0.5f) * 5.9604644775390625e-8f;
}

// The first RandomState.random_sample() value for one NumPy MT19937 seed.
// ActivitySim resets each entity's stream at a model step, so the calibrated
// Phase 19 components require offset zero only.  The first twist outputs need
// just state words 0, 1, 2, 397, and 398; generating those words avoids a
// 624-word per-thread local array while retaining NumPy's exact bit semantics.
extern "C" __global__ void activitysim_uniform_f64_offset0(
    const long long* entity_ids,
    int n,
    unsigned int combined_name_seed,
    unsigned int base_seed,
    double* output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    unsigned int seed = (unsigned int)(
        (unsigned long long)base_seed
        + (unsigned long long)combined_name_seed
        + (unsigned long long)entity_ids[row]);

    unsigned int mt0 = seed;
    unsigned int mt1 = 0U, mt2 = 0U, mt397 = 0U, mt398 = 0U;
    unsigned int value = mt0;
    for (unsigned int i = 1U; i <= 398U; ++i) {
        value = 1812433253U * (value ^ (value >> 30)) + i;
        if (i == 1U) mt1 = value;
        else if (i == 2U) mt2 = value;
        else if (i == 397U) mt397 = value;
        else if (i == 398U) mt398 = value;
    }
    const unsigned int upper = 0x80000000U;
    const unsigned int lower = 0x7fffffffU;
    const unsigned int matrix = 0x9908b0dfU;
    unsigned int y0 = (mt0 & upper) | (mt1 & lower);
    unsigned int y1 = (mt1 & upper) | (mt2 & lower);
    unsigned int first = mt397 ^ (y0 >> 1) ^ ((y0 & 1U) ? matrix : 0U);
    unsigned int second = mt398 ^ (y1 >> 1) ^ ((y1 & 1U) ? matrix : 0U);

    first ^= first >> 11;
    first ^= (first << 7) & 0x9d2c5680U;
    first ^= (first << 15) & 0xefc60000U;
    first ^= first >> 18;
    second ^= second >> 11;
    second ^= (second << 7) & 0x9d2c5680U;
    second ^= (second << 15) & 0xefc60000U;
    second ^= second >> 18;

    unsigned long long numerator =
        ((unsigned long long)(first >> 5) << 26)
        + (unsigned long long)(second >> 6);
    output[row] = (double)numerator * (1.0 / 9007199254740992.0);
}

extern "C" __global__ void segmented_sum_sorted_f32(
    const long long* group_ids,
    const float* values,
    int n,
    unsigned char* is_start,
    float* sums)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    bool start = row == 0 || group_ids[row - 1] != group_ids[row];
    is_start[row] = start ? 1 : 0;
    if (!start) { sums[row] = 0.0f; return; }
    float acc = 0.0f;
    long long group = group_ids[row];
    for (int cursor = row; cursor < n && group_ids[cursor] == group; ++cursor) {
        acc += values[cursor];
    }
    sums[row] = acc;
}
"""


@lru_cache(maxsize=1)
def _gpu_native_kernels() -> tuple[Any, Any, Any]:
    cp = _cupy()
    module = cp.RawModule(
        code=GPU_NATIVE_CUDA,
        options=("--std=c++11",),
        name_expressions=(
            "entity_uniform_f32",
            "segmented_sum_sorted_f32",
            "activitysim_uniform_f64_offset0",
        ),
    )
    return (
        module.get_function("entity_uniform_f32"),
        module.get_function("segmented_sum_sorted_f32"),
        module.get_function("activitysim_uniform_f64_offset0"),
    )


def entity_uniforms_gpu(entity_ids: Any, seed: int, stream: int = 0) -> Any:
    """Generate stable float32 draws entirely on the GPU from entity IDs."""

    cp = _cupy()
    if not _is_cuda_array(entity_ids):
        raise GpuOnlyViolation("entity IDs must already reside on the GPU")
    ids = cp.ascontiguousarray(entity_ids, dtype=cp.int64)
    output = cp.empty(ids.shape, dtype=cp.float32)
    if ids.size == 0:
        return output
    threads = 256
    blocks = (int(ids.size) + threads - 1) // threads
    kernel, _, _ = _gpu_native_kernels()
    kernel(
        (blocks,),
        (threads,),
        (ids, np.int32(ids.size), np.uint64(seed), np.uint64(stream), output),
    )
    return output


def activitysim_hash32(value: str) -> int:
    """Return ActivitySim's stable low-32-bit MD5 name hash."""

    if not isinstance(value, str) or not value:
        raise ValueError("ActivitySim random channel and step names must be nonempty")
    return int(hashlib.md5(value.encode("utf8")).hexdigest(), 16) & 0xFFFFFFFF


def activitysim_uniforms_gpu(
    entity_ids: Any,
    channel_name: str,
    step_name: str,
    *,
    base_seed: int = 0,
    offset: int = 0,
) -> Any:
    """Generate ActivitySim/NumPy RandomState draws on the GPU, bit exactly.

    The implemented contract is deliberately narrow: one draw at offset zero,
    which is the random usage of the calibrated auto-ownership and mandatory-
    tour-frequency MNL components.  Any later stream offset fails closed.
    """

    cp = _cupy()
    if not _is_cuda_array(entity_ids):
        raise GpuOnlyViolation("ActivitySim random entity IDs must reside on the GPU")
    if offset != 0:
        raise GpuOnlyViolation("the ActivitySim GPU random kernel currently supports offset zero only")
    if not 0 <= int(base_seed) <= 0xFFFFFFFF:
        raise ValueError("base_seed must fit uint32")
    ids = cp.ascontiguousarray(entity_ids, dtype=cp.int64)
    output = cp.empty(ids.shape, dtype=cp.float64)
    if ids.size == 0:
        return output
    combined = (
        activitysim_hash32(channel_name) + activitysim_hash32(step_name)
    ) & 0xFFFFFFFF
    threads = 256
    blocks = (int(ids.size) + threads - 1) // threads
    _, _, kernel = _gpu_native_kernels()
    kernel(
        (blocks,),
        (threads,),
        (ids, np.int32(ids.size), np.uint32(combined), np.uint32(base_seed), output),
    )
    return output


def activitysim_uniforms_cpu(
    entity_ids: Any,
    channel_name: str,
    step_name: str,
    *,
    base_seed: int = 0,
) -> np.ndarray:
    """Independent NumPy oracle for ActivitySim's first per-entity draw."""

    ids = np.asarray(entity_ids, dtype=np.int64)
    combined = (
        activitysim_hash32(channel_name) + activitysim_hash32(step_name)
    ) & 0xFFFFFFFF
    seeds = (ids + int(base_seed) + combined) % (1 << 32)
    result = np.empty(ids.shape, dtype=np.float64)
    generator = np.random.RandomState()
    for index, seed in np.ndenumerate(seeds):
        generator.seed(int(seed))
        result[index] = generator.rand()
    return result


def entity_uniforms_cpu(entity_ids: Any, seed: int, stream: int = 0) -> np.ndarray:
    """Independent CPU oracle for the counter-based GPU random stream."""

    ids = np.asarray(entity_ids, dtype=np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        key = ids ^ (np.uint64(seed) * np.uint64(0xD2B74407B1CE6E93))
        key ^= np.uint64(stream) * np.uint64(0xCA5A826395121157)
        value = key + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return (((value >> np.uint64(40)).astype(np.float32) + np.float32(0.5)) * np.float32(2.0**-24))


def segmented_sum_sorted_gpu(group_ids: Any, values: Any) -> DeviceTable:
    """Sum adjacent equal-ID runs in fixed row order without GPU atomics.

    The returned arrays have one element per input row. ``is_start`` marks the
    rows containing valid sums; non-start rows contain zero. This avoids a host
    round trip merely to discover the compact output size.
    """

    cp = _cupy()
    if not _is_cuda_array(group_ids) or not _is_cuda_array(values):
        raise GpuOnlyViolation("segmented-sum inputs must already reside on the GPU")
    groups = cp.ascontiguousarray(group_ids, dtype=cp.int64)
    data = cp.ascontiguousarray(values, dtype=cp.float32)
    if groups.shape != data.shape or groups.ndim != 1:
        raise ValueError("group_ids and values must be equal-length vectors")
    starts = cp.empty(groups.shape, dtype=cp.uint8)
    sums = cp.empty(data.shape, dtype=cp.float32)
    if groups.size == 0:
        return DeviceTable({"group_id": groups, "is_start": starts, "sum": sums})
    if groups.size > 1 and bool(cp.any(groups[1:] < groups[:-1]).item()):
        raise ValueError("group_ids must be sorted in nondecreasing order")
    threads = 256
    blocks = (int(groups.size) + threads - 1) // threads
    _, kernel, _ = _gpu_native_kernels()
    kernel((blocks,), (threads,), (groups, data, np.int32(groups.size), starts, sums))
    return DeviceTable({"group_id": groups, "is_start": starts, "sum": sums})
