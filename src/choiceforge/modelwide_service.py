"""Persistent exact runtime state for the Phase 46 destination service.

The service owns reusable CUDA workspaces and reproduces ActivitySim's scalar-
seeded NumPy ``RandomState`` streams on the GPU.  ActivitySim remains the owner
of row seeds and offsets; the service reads and advances that ledger under the
same one-call contract as ``random_for_df``.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .cuda_backend import _cupy


_SERVICE = None


_MT19937_SOURCE = r'''
__device__ __forceinline__ unsigned int phase46_mt_next(
    unsigned int* mt, int* position)
{
    if (*position >= 624) {
        const unsigned int upper = 0x80000000U;
        const unsigned int lower = 0x7fffffffU;
        for (int i = 0; i < 227; ++i) {
            const unsigned int y = (mt[i] & upper) | (mt[i + 1] & lower);
            mt[i] = mt[i + 397] ^ (y >> 1) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
        }
        for (int i = 227; i < 623; ++i) {
            const unsigned int y = (mt[i] & upper) | (mt[i + 1] & lower);
            mt[i] = mt[i - 227] ^ (y >> 1) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
        }
        const unsigned int y = (mt[623] & upper) | (mt[0] & lower);
        mt[623] = mt[396] ^ (y >> 1) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
        *position = 0;
    }
    unsigned int value = mt[*position];
    *position += 1;
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680U;
    value ^= (value << 15) & 0xefc60000U;
    value ^= value >> 18;
    return value;
}

extern "C" __global__ void phase46_mt19937_rows(
    const unsigned int* seeds,
    const int* offsets,
    unsigned int* states,
    double* output,
    int rows,
    int draws)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int* mt = states + (long long)row * 624;
    mt[0] = seeds[row];
    for (int i = 1; i < 624; ++i) {
        const unsigned int previous = mt[i - 1];
        mt[i] = 1812433253U * (previous ^ (previous >> 30)) + (unsigned int)i;
    }
    int position = 624;
    const int stop = offsets[row] + draws;
    for (int draw = 0; draw < stop; ++draw) {
        const unsigned int first = phase46_mt_next(mt, &position) >> 5;
        const unsigned int second = phase46_mt_next(mt, &position) >> 6;
        if (draw >= offsets[row]) {
            output[(long long)row * draws + draw - offsets[row]] =
                ((double)first * 67108864.0 + (double)second)
                * (1.0 / 9007199254740992.0);
        }
    }
}
'''


class Phase46DestinationService:
    """Own compiled kernels and grow-only scratch buffers for one model run."""

    def __init__(self, cp=None):
        self.cp = cp or _cupy()
        self._rng_kernel = self.cp.RawKernel(
            _MT19937_SOURCE,
            "phase46_mt19937_rows",
            options=("--std=c++11", "--fmad=false"),
        )
        self._rng_kernel.compile()
        self._rng_capacity_rows = 0
        self._rng_capacity_values = 0
        self._rng_states = None
        self._rng_seeds = None
        self._rng_offsets = None
        self._rng_output = None
        self._cell_capacity = 0
        self._sample_capacity = 0
        self._row_capacity = 0
        self._utility = None
        self._weights = None
        self._choices = None
        self._probabilities = None
        self._first = None
        self._counts = None
        self._guard = None
        self._bad = None
        self._risk = None
        self._random_events = []
        self._workspace_growths = 0
        self.phase47_device_final = False

    def _ensure_rng(self, rows: int, draws: int) -> None:
        cp = self.cp
        if rows > self._rng_capacity_rows:
            self._rng_states = cp.empty((rows, 624), dtype=cp.uint32)
            self._rng_seeds = cp.empty(rows, dtype=cp.uint32)
            self._rng_offsets = cp.empty(rows, dtype=cp.int32)
            self._rng_capacity_rows = rows
            self._workspace_growths += 1
        values = rows * draws
        if values > self._rng_capacity_values:
            self._rng_output = cp.empty(values, dtype=cp.float64)
            self._rng_capacity_values = values
            self._workspace_growths += 1

    def generate_from_seeds(self, seeds, offsets, draws: int):
        """Generate exact RandomState uniforms without mutating a ledger."""
        seeds = np.ascontiguousarray(seeds, dtype=np.uint32)
        offsets = np.ascontiguousarray(offsets, dtype=np.int32)
        draws = int(draws)
        if seeds.ndim != 1 or offsets.shape != seeds.shape:
            raise ValueError("Phase 46 seeds and offsets must be aligned vectors")
        if draws <= 0 or np.any(offsets < 0) or np.any(offsets + draws > 4096):
            raise ValueError("Phase 46 supports 1..4096 consumed double draws per row")
        rows = len(seeds)
        self._ensure_rng(rows, draws)
        self._rng_seeds[:rows].set(seeds)
        self._rng_offsets[:rows].set(offsets)
        output = self._rng_output[: rows * draws].reshape(rows, draws)
        block = 128
        self._rng_kernel(
            ((rows + block - 1) // block,),
            (block,),
            (
                self._rng_seeds,
                self._rng_offsets,
                self._rng_states,
                output,
                np.int32(rows),
                np.int32(draws),
            ),
        )
        return output

    def random_for_df(self, state, frame, draws: int):
        """Generate draws and advance ActivitySim's authoritative row ledger."""
        started = time.perf_counter()
        rng = state.get_rn_generator()
        if not getattr(rng, "channels", None):
            raise ValueError("Phase 46 requires ActivitySim's keyed random channels")
        channel = rng.get_channel_for_df(frame)
        if getattr(channel, "step_name", None) != getattr(rng, "step_name", None):
            raise ValueError("Phase 46 random channel is outside its active step")
        row_states = channel.row_states.loc[frame.index, ["row_seed", "offset"]]
        seeds = np.ascontiguousarray(row_states["row_seed"], dtype=np.uint32)
        offsets = np.ascontiguousarray(row_states["offset"], dtype=np.int32)
        prepared = time.perf_counter()
        device = self.generate_from_seeds(seeds, offsets, draws)
        self.cp.cuda.Stream.null.synchronize()
        generated = time.perf_counter()
        host = self.cp.asnumpy(device)
        transferred = time.perf_counter()
        diagnostic_id = os.environ.get("CHOICEFORGE_PHASE46_RNG_DIAGNOSTIC_ID")
        if diagnostic_id is not None:
            target = int(diagnostic_id)
            positions = np.flatnonzero(np.asarray(frame.index) == target)
            if len(positions):
                position = int(positions[0])
                generator = np.random.RandomState()
                # Match ActivitySim exactly: seed with the pandas row-state
                # scalar, then consume its recorded double-draw offset.
                generator.seed(row_states.iloc[position]["row_seed"])
                generator.rand(int(offsets[position]))
                expected = generator.rand(int(draws))
                print(
                    "PHASE46_RNG_DIAGNOSTIC "
                    + repr(
                        {
                            "id": target,
                            "seed": int(seeds[position]),
                            "offset": int(offsets[position]),
                            "draws": int(draws),
                            "bit_mismatches": int(np.count_nonzero(
                                expected.view(np.uint64)
                                != host[position].view(np.uint64)
                            )),
                            "expected_first": float(expected[0]),
                            "gpu_first": float(host[position, 0]),
                        }
                    ),
                    flush=True,
                )
        shadow_rows = os.environ.get("CHOICEFORGE_PHASE46_RNG_SHADOW_ROWS")
        if shadow_rows is not None and len(frame) == int(shadow_rows):
            expected = []
            for row in row_states.itertuples(index=False):
                generator = np.random.RandomState()
                generator.seed(row.row_seed)
                generator.rand(int(row.offset))
                expected.append(generator.rand(int(draws)))
            expected = np.asarray(expected, dtype=np.float64)
            mismatch = expected.view(np.uint64) != host.view(np.uint64)
            print(
                "PHASE46_RNG_SHADOW "
                + repr(
                    {
                        "rows": len(frame),
                        "draws": int(draws),
                        "bit_mismatches": int(np.count_nonzero(mismatch)),
                        "mismatch_rows": int(np.count_nonzero(np.any(mismatch, axis=1))),
                    }
                ),
                flush=True,
            )
        channel.row_states.loc[frame.index, "offset"] += int(draws)
        finished = time.perf_counter()
        self._random_events.append(
            {
                "rows": len(frame),
                "draws_per_row": int(draws),
                "draw_values": int(len(frame) * draws),
                "ledger_read_seconds": prepared - started,
                "gpu_generation_seconds": generated - prepared,
                "host_transfer_seconds": transferred - generated,
                "ledger_update_seconds": finished - transferred,
                "total_seconds": finished - started,
            }
        )
        return host, device

    def sample_workspace(self, rows: int, alternatives: int, draws: int) -> dict:
        """Return reusable views sized for one dense sampling invocation."""
        cp = self.cp
        cells = int(rows) * int(alternatives)
        samples = int(rows) * int(draws)
        if cells > self._cell_capacity:
            self._utility = cp.empty(cells, dtype=cp.float32)
            self._weights = cp.empty(cells, dtype=cp.float32)
            self._cell_capacity = cells
            self._workspace_growths += 1
        if samples > self._sample_capacity:
            self._choices = cp.empty(samples, dtype=cp.int32)
            self._probabilities = cp.empty(samples, dtype=cp.float32)
            self._first = cp.empty(samples, dtype=cp.uint8)
            self._counts = cp.empty(samples, dtype=cp.uint32)
            self._sample_capacity = samples
            self._workspace_growths += 1
        if rows > self._row_capacity:
            self._guard = cp.empty(rows, dtype=cp.uint8)
            self._bad = cp.empty(rows, dtype=cp.uint8)
            self._risk = cp.empty(rows, dtype=cp.float32)
            self._row_capacity = rows
            self._workspace_growths += 1
        utility = self._utility[:cells].reshape(rows, alternatives)
        weights = self._weights[:cells].reshape(rows, alternatives)
        choices = self._choices[:samples].reshape(rows, draws)
        probabilities = self._probabilities[:samples].reshape(rows, draws)
        first = self._first[:samples].reshape(rows, draws)
        counts = self._counts[:samples].reshape(rows, draws)
        guard = self._guard[:rows]
        bad = self._bad[:rows]
        risk = self._risk[:rows]
        guard.fill(0)
        bad.fill(0)
        risk.fill(0)
        return {
            "utility": utility,
            "weights": weights,
            "choices": choices,
            "probabilities": probabilities,
            "first": first,
            "counts": counts,
            "guard": guard,
            "bad": bad,
            "risk": risk,
        }

    def final_workspace(self, rows: int, alternative_rows: int, width: int) -> dict:
        """Return Phase 47 views without adding another dense allocation."""
        rows = int(rows)
        alternative_rows = int(alternative_rows)
        width = int(width)
        padded_cells = rows * width
        required_cells = max(alternative_rows, padded_cells)
        required_samples = max(rows, padded_cells)
        cp = self.cp
        if required_cells > self._cell_capacity:
            self._utility = cp.empty(required_cells, dtype=cp.float32)
            self._weights = cp.empty(required_cells, dtype=cp.float32)
            self._cell_capacity = required_cells
            self._workspace_growths += 1
        if required_samples > self._sample_capacity:
            self._choices = cp.empty(required_samples, dtype=cp.int32)
            self._probabilities = cp.empty(required_samples, dtype=cp.float32)
            self._first = cp.empty(required_samples, dtype=cp.uint8)
            self._counts = cp.empty(required_samples, dtype=cp.uint32)
            self._sample_capacity = required_samples
            self._workspace_growths += 1
        if rows > self._row_capacity:
            self._guard = cp.empty(rows, dtype=cp.uint8)
            self._bad = cp.empty(rows, dtype=cp.uint8)
            self._risk = cp.empty(rows, dtype=cp.float32)
            self._row_capacity = rows
            self._workspace_growths += 1
        self._guard[:rows].fill(0)
        self._bad[:rows].fill(0)
        self._risk[:rows].fill(0)
        return {
            "utilities": self._utility[:padded_cells].reshape(rows, width),
            "weights": self._weights[:padded_cells].reshape(rows, width),
            "positions": self._choices[:rows],
            "selected_probabilities": self._probabilities[:rows],
            "guard": self._guard[:rows],
            "bad": self._bad[:rows],
            "row_maxima": self._risk[:rows],
        }

    def summary(self) -> dict:
        random_seconds = sum(item["total_seconds"] for item in self._random_events)
        allocated = (
            self._rng_capacity_rows * (624 * 4 + 4 + 4)
            + self._rng_capacity_values * 8
            + self._cell_capacity * 8
            + self._sample_capacity * (4 + 4 + 1 + 4)
            + self._row_capacity * (1 + 1 + 4)
        )
        return {
            "random_calls": len(self._random_events),
            "random_rows": sum(item["rows"] for item in self._random_events),
            "random_draw_values": sum(
                item["draw_values"] for item in self._random_events
            ),
            "random_seconds": random_seconds,
            "workspace_growths": self._workspace_growths,
            "workspace_bytes": int(allocated),
            "cell_capacity": self._cell_capacity,
            "row_capacity": self._row_capacity,
            "events": list(self._random_events),
        }


def get_phase46_service() -> Phase46DestinationService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = Phase46DestinationService()
    return _SERVICE


def reset_phase46_service() -> None:
    global _SERVICE
    _SERVICE = None


def phase46_service_summary() -> dict | None:
    return None if _SERVICE is None else _SERVICE.summary()
