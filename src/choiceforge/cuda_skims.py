"""Exact-layout CUDA adapter for standard ActivitySim dense skim dictionaries.

The adapter preserves ActivitySim's existing ``OffsetMapper`` on the host,
then gathers its dense skim cube on the GPU.  Keeping zone mapping in the
reviewed ActivitySim mapper is deliberate: it covers sparse/noncontiguous zone
IDs without reimplementing a second, subtly different mapping rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np


NOT_IN_SKIM_ZONE_ID = -1
_SKIM_CACHE = {}
_DATASET_ARRAY_CACHE = {}
_DATASET_BINDING_CACHE = {}
_DATASET_CACHE_STATS = {"binding_hits": 0, "binding_misses": 0, "array_uploads": 0}


@dataclass(frozen=True)
class CudaDatasetSkimBinding:
    """Compact kernel ABI for one skim cube and its mapped row positions."""

    data: object
    orig: object
    dest: object
    time: object | None
    dest_count: int
    time_count: int
    choiceforge_device_skim_binding: bool = True


def clear_cuda_dataset_cache():
    """Release immutable skim cubes and unused CuPy pool blocks after a model step."""
    from .cuda_backend import _cupy

    _DATASET_ARRAY_CACHE.clear()
    _DATASET_BINDING_CACHE.clear()
    for key in _DATASET_CACHE_STATS:
        _DATASET_CACHE_STATS[key] = 0
    cp = _cupy()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def cuda_dataset_cache_stats():
    """Return monotonic in-process metadata/gather-cache counters."""
    return {
        **_DATASET_CACHE_STATS,
        "binding_entries": len(_DATASET_BINDING_CACHE),
        "array_entries": len(_DATASET_ARRAY_CACHE),
    }


class CudaChooserColumns:
    """Lazy device view of a pandas-like chooser table's numeric columns."""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self._cache = {}

    def __getitem__(self, column):
        from .cuda_backend import _cupy

        if column not in self._cache:
            self._cache[column] = _cupy().asarray(np.asarray(self.dataframe[column]))
        return self._cache[column]


class CudaSkimWrapper:
    """Lazy GPU equivalent of ActivitySim's 2D/3D skim wrappers."""

    def __init__(self, skims, dataframe, origin_column, destination_column, period_column=None):
        self.skims = skims
        self.dataframe = dataframe
        self.origin_column = origin_column
        self.destination_column = destination_column
        self.period_column = period_column
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            orig = self.dataframe[self.origin_column].to_numpy(copy=False)
            dest = self.dataframe[self.destination_column].to_numpy(copy=False)
            if self.period_column is None:
                self._cache[key] = self.skims.lookup(orig, dest, key)
            else:
                period = self.dataframe[self.period_column].to_numpy(copy=False)
                self._cache[key] = self.skims.lookup_3d(orig, dest, period, key)
        return self._cache[key]


class CudaDatasetWrapper:
    """GPU gather view of Sharrow/ActivitySim's targeted DatasetWrapper.

    It consumes the wrapper's already-prepared ordinal ``positions`` rather
    than recreating zone or period mapping.  That makes the gather semantics
    identical to the Sharrow wrapper under comparison.
    """

    def __init__(self, wrapper, *, reverse=False):
        if getattr(wrapper, "df", None) is None or not hasattr(wrapper, "positions"):
            raise ValueError("DatasetWrapper has no currently targeted chooser frame")
        self.wrapper = wrapper
        self.reverse = bool(reverse)
        self.dataframe = wrapper.df
        self._cache = {}
        self._device_arrays = {}
        self._device_positions = {}

    def _position(self, name):
        positions = self.wrapper.positions
        value = positions[name] if isinstance(positions, dict) else positions[name].to_numpy(copy=False)
        return np.asarray(value, dtype=np.int64)

    def _device_position(self, name):
        from .cuda_backend import _cupy

        if name not in self._device_positions:
            self._device_positions[name] = _cupy().asarray(self._position(name))
        return self._device_positions[name]

    def __getitem__(self, key):
        from .cuda_backend import _cupy

        if key in self._cache:
            return self._cache[key]
        cp = _cupy()
        if self.reverse:
            self._cache[key] = cp.asarray(np.asarray(self.wrapper.reverse(key)))
            return self._cache[key]
        array = self.wrapper.dataset[key]
        odim, ddim = self.wrapper.odim, self.wrapper.ddim
        required = (odim, ddim)
        if not all(dim in array.dims for dim in required):
            raise ValueError(f"skim {key!r} does not have expected OD dimensions")
        has_time = "time_period" in array.dims
        order = required + (("time_period",) if has_time else ())
        if set(array.dims) != set(order):
            raise ValueError(f"skim {key!r} has unsupported dimensions {array.dims!r}")
        host_values = np.asarray(array.transpose(*order).values)
        interface = host_values.__array_interface__
        # Directional DatasetWrapper objects are recreated for each batch, but
        # their xarray variables share the same underlying NumPy allocation.
        # Key by that allocation/view, not by the ephemeral Dataset identity,
        # so OD/DOT/ODT wrappers upload each immutable skim array only once.
        device_key = (
            int(interface["data"][0]),
            host_values.shape,
            host_values.strides,
            host_values.dtype.str,
        )
        if device_key not in _DATASET_ARRAY_CACHE:
            _DATASET_ARRAY_CACHE[device_key] = cp.ascontiguousarray(
                cp.asarray(host_values)
            )
        data = _DATASET_ARRAY_CACHE[device_key]
        orig, dest = self._device_position(odim), self._device_position(ddim)
        if has_time:
            if "time_period" not in self.wrapper.positions:
                raise ValueError(f"time-dependent skim {key!r} has no mapped period positions")
            result = data[orig, dest, self._device_position("time_period")]
        else:
            result = data[orig, dest]
        self._cache[key] = result
        return result

    def strict_binding(self, key):
        """Return cube-plus-index inputs without materializing a gathered vector."""
        from .cuda_backend import _cupy

        cp = _cupy()
        odim, ddim = self.wrapper.odim, self.wrapper.ddim
        binding_key = (id(self.wrapper.dataset), key, odim, ddim, self.reverse)
        cached = _DATASET_BINDING_CACHE.get(binding_key)
        if cached is None:
            _DATASET_CACHE_STATS["binding_misses"] += 1
            array = self.wrapper.dataset[key]
            required = (odim, ddim)
            if not all(dim in array.dims for dim in required):
                raise ValueError(f"skim {key!r} does not have expected OD dimensions")
            has_time = "time_period" in array.dims
            order = required + (("time_period",) if has_time else ())
            if set(array.dims) != set(order):
                raise ValueError(f"skim {key!r} has unsupported dimensions {array.dims!r}")
            host_values = np.asarray(array.transpose(*order).values)
            if host_values.dtype != np.float32:
                raise ValueError(
                    f"strict compact skim {key!r} requires float32, got {host_values.dtype}"
                )
            interface = host_values.__array_interface__
            device_key = (
                int(interface["data"][0]),
                host_values.shape,
                host_values.strides,
                host_values.dtype.str,
            )
            if device_key not in _DATASET_ARRAY_CACHE:
                _DATASET_ARRAY_CACHE[device_key] = cp.ascontiguousarray(
                    cp.asarray(host_values)
                )
                _DATASET_CACHE_STATS["array_uploads"] += 1
            cached = (
                _DATASET_ARRAY_CACHE[device_key],
                has_time,
                int(host_values.shape[1]),
                int(host_values.shape[2]) if has_time else 1,
            )
            _DATASET_BINDING_CACHE[binding_key] = cached
        else:
            _DATASET_CACHE_STATS["binding_hits"] += 1
        data, has_time, dest_count, time_count = cached
        orig_name, dest_name = (ddim, odim) if self.reverse else (odim, ddim)
        return CudaDatasetSkimBinding(
            data=data,
            orig=self._device_position(orig_name),
            dest=self._device_position(dest_name),
            time=self._device_position("time_period") if has_time else None,
            dest_count=dest_count,
            time_count=time_count,
        )


def cuda_wrapper_from_activitysim(wrapper, *, reverse=False):
    """Adapt a currently-targeted ActivitySim SkimWrapper or Skim3dWrapper.

    ActivitySim sets ``wrapper.df`` immediately before evaluating a chunk; this
    function must therefore be called within that evaluation scope.
    """
    if not hasattr(wrapper, "df") or wrapper.df is None:
        raise ValueError("ActivitySim skim wrapper has no currently targeted chooser frame")
    if hasattr(wrapper, "dataset") and not hasattr(wrapper, "skim_dict"):
        return CudaDatasetWrapper(wrapper, reverse=reverse)
    skim_dict = wrapper.skim_dict
    cache_key = id(skim_dict)
    if cache_key not in _SKIM_CACHE:
        _SKIM_CACHE[cache_key] = CudaSkimDictionary.from_activitysim(skim_dict)
    origin_key = wrapper.dest_key if reverse else wrapper.orig_key
    destination_key = wrapper.orig_key if reverse else wrapper.dest_key
    return CudaSkimWrapper(
        _SKIM_CACHE[cache_key], wrapper.df, origin_key, destination_key,
        getattr(wrapper, "dim3_key", None),
    )


def activitysim_cuda_environment(choosers, locals_dict, **skim_wrappers):
    """Build a lazy device environment for the reviewed expression compiler.

    ``skim_wrappers`` maps names such as ``odt_skims`` to a configured
    :class:`CudaSkimWrapper`.  Numeric locals are copied to the device only
    when they are array-like; scalar locals retain their exact Python value.
    Other locals are omitted and therefore cause a clear fail-closed error if
    an expression unexpectedly depends on them.
    """
    from .cuda_backend import _cupy

    cp = _cupy()
    environment = {"df": CudaChooserColumns(choosers), **skim_wrappers}
    # ActivitySim expressions may refer to chooser columns either as
    # ``df.column`` or directly as ``column``.  Preserve both namespaces.
    # CudaChooserColumns caches the conversion, so each column is uploaded at
    # most once per chunk environment.
    for column in choosers.columns:
        if np.asarray(choosers[column]).dtype.kind in "biuf":
            environment[column] = environment["df"][column]
    for name, value in locals_dict.items():
        if isinstance(value, (int, float, bool, np.number)):
            environment[name] = value
        elif isinstance(value, np.ndarray) and value.ndim <= 1:
            environment[name] = cp.asarray(value)
    return environment


@dataclass
class CudaSkimDictionary:
    """GPU view of an ActivitySim row-major ``SkimDict`` dense cube.

    ``skim_data`` uses ActivitySim's `(block, origin, destination)` layout.
    This adapter intentionally excludes sparse MAZ overlays: those need their
    own equivalence proof before being considered for GPU execution.
    """

    skim_data: object
    block_offsets: dict
    skim_dim3: dict
    offset_mapper: object
    omx_shape: tuple[int, int]
    _device_data: object = field(default=None, init=False, repr=False)

    @classmethod
    def from_activitysim(cls, skim_dict):
        """Create a device-ready adapter for a normal dense ``SkimDict``.

        The required attributes are intentionally checked structurally so this
        module does not import ActivitySim during ordinary package use.
        """
        required = ("skim_data", "skim_info", "skim_dim3", "offset_mapper", "omx_shape")
        if any(not hasattr(skim_dict, name) for name in required):
            raise TypeError("expected an ActivitySim dense SkimDict")
        if type(skim_dict).__name__ == "MazSkimDict":
            raise ValueError("sparse MazSkimDict is not yet a supported CUDA skim layout")
        return cls(
            skim_data=skim_dict.skim_data,
            block_offsets=dict(skim_dict.skim_info.block_offsets),
            skim_dim3={key: dict(value) for key, value in skim_dict.skim_dim3.items()},
            offset_mapper=skim_dict.offset_mapper,
            omx_shape=tuple(skim_dict.omx_shape[:2]),
        )

    def lookup(self, orig, dest, key, *, return_device=True):
        """Gather a 2D skim, matching SkimDict's invalid-zone contract."""
        if key not in self.block_offsets:
            raise KeyError(f"skim key {key!r} is not available")
        return self._gather(orig, dest, self.block_offsets[key], return_device=return_device)

    def lookup_3d(self, orig, dest, dim3, key, *, return_device=True):
        """Gather a stacked 3D skim using ActivitySim's period-to-block map."""
        if key not in self.skim_dim3:
            raise KeyError(f"3D skim key {key!r} is not available")
        labels = np.asarray(dim3)
        mapping = self.skim_dim3[key]
        try:
            blocks = np.asarray([mapping[label] for label in labels], dtype=np.int64)
        except KeyError as err:
            raise KeyError(f"unknown {key!r} skim period {err.args[0]!r}") from err
        return self._gather(orig, dest, blocks, return_device=return_device)

    def _gather(self, orig, dest, blocks, *, return_device):
        from .cuda_backend import _cupy

        cp = _cupy()
        orig = np.asarray(orig, dtype=np.int64)
        dest = np.asarray(dest, dtype=np.int64)
        if orig.shape != dest.shape or orig.ndim != 1:
            raise ValueError("orig and dest must be same-length one-dimensional arrays")
        mapped_orig = np.asarray(self.offset_mapper.map(orig), dtype=np.int64)
        mapped_dest = np.asarray(self.offset_mapper.map(dest), dtype=np.int64)
        valid = (
            (mapped_orig >= 0) & (mapped_orig < self.omx_shape[0])
            & (mapped_dest >= 0) & (mapped_dest < self.omx_shape[1])
        )
        permitted_missing = (orig == NOT_IN_SKIM_ZONE_ID) | (dest == NOT_IN_SKIM_ZONE_ID)
        if np.any(~valid & ~permitted_missing):
            bad = np.flatnonzero(~valid & ~permitted_missing)[:5]
            pairs = ", ".join(f"{orig[i]}->{dest[i]}" for i in bad)
            raise AssertionError(f"OD pairs not in skim: {pairs}")
        blocks = np.broadcast_to(np.asarray(blocks, dtype=np.int64), orig.shape)
        if np.any((blocks < 0) | (blocks >= self.skim_data.shape[0])):
            raise ValueError("skim block offset is out of range")
        # Safe zero indices avoid NumPy's negative-index wrapping for the
        # permitted -1 sentinel; those entries are replaced with NaN below.
        safe_orig = np.where(valid, mapped_orig, 0)
        safe_dest = np.where(valid, mapped_dest, 0)
        # Skim cubes are invariant throughout a model run.  Upload once and
        # retain the device view for all directional expressions/batches.
        if self._device_data is None:
            self._device_data = cp.ascontiguousarray(cp.asarray(self.skim_data))
        cube = self._device_data
        result = cube[cp.asarray(blocks), cp.asarray(safe_orig), cp.asarray(safe_dest)]
        result = cp.where(cp.asarray(valid), result, cp.nan)
        return result if return_device else cp.asnumpy(result)
