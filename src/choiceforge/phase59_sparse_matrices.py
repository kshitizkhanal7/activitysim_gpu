"""Standard gzip/shuffle OMX with zero-only chunks left at their fill value.

This is a CPU output-layout optimization, not a GPU hardware claim. It retains
every logical matrix cell and uses only standard HDF5 filters. No reports or
matrices are omitted; sparse storage is invisible to ordinary OMX readers.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import zlib
import h5py
import numpy as np


def compress_tile(tile):
    # HDF5's shuffle filter groups bytes by byte-position within each element.
    shuffled = np.ascontiguousarray(tile.view(np.uint8).reshape(-1, 8).T)
    return zlib.compress(shuffled.tobytes(), level=1)


def write_matrices(state, aggregate_trips, zone_index, orig_index, dest_index, model_settings,
                   *, chunk_size=128, workers=4):
    started = time.perf_counter()
    n = len(zone_index)
    if n < 1 or chunk_size < 1 or workers < 1 or not isinstance(zone_index.name, str):
        raise ValueError("Phase 59 sparse OMX requires named nonempty zones and positive resource limits")
    orig, dest = np.asarray(orig_index), np.asarray(dest_index)
    if (orig.shape != dest.shape or len(orig) != len(aggregate_trips)
            or np.any(orig < 0) or np.any(orig >= n) or np.any(dest < 0) or np.any(dest >= n)):
        raise ValueError("Phase 59 sparse OMX coordinates are outside the zone mapping")
    if len(np.unique(orig.astype(np.int64)*n+dest)) != len(orig):
        raise ValueError("Phase 59 sparse OMX requires unique aggregated OD pairs")
    block = min(chunk_size, n)
    blocks = (n+block-1)//block
    event = {"omx_files":0, "matrices":0, "written_chunks":0, "compressed_chunk_bytes":0,
             "dense_logical_bytes":0, "compression_workers":workers, "chunk_size":block,
             "gpu_acceleration_claimed":False, "zero_only_chunks_omitted":0}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="choiceforge-omx") as pool:
        for matrix in model_settings.MATRICES:
            with h5py.File(state.get_output_file_path(str(matrix.file_name)), "w") as file:
                file.attrs["OMX_VERSION"] = np.bytes_("0.2")
                file.attrs["OMX_CREATED_WITH"] = "ChoiceForge standard sparse-chunk OMX"
                file.attrs["SHAPE"] = np.array([n, n], dtype=np.int32)
                data = file.create_group("data")
                lookup = file.create_group("lookup")
                # OpenMatrix create_mapping always writes UInt32Atom, even when
                # the input pandas index is signed/int64. Match that contract.
                lookup.create_dataset(zone_index.name, data=zone_index.to_numpy(dtype=np.uint32))
                for table in matrix.tables:
                    col = table.data_field
                    if col not in aggregate_trips:
                        raise ValueError(f"Phase 59 matrix field missing: {col}")
                    if model_settings.HH_EXPANSION_WEIGHT_COL:
                        aggregate_trips[col] = aggregate_trips[col]/aggregate_trips[model_settings.HH_EXPANSION_WEIGHT_COL]
                    values = aggregate_trips[col].to_numpy(dtype=np.float64)
                    # Negative zero is data too under the exact byte contract.
                    nonzero = (values != 0) | np.signbit(values)
                    rows, cols = orig[nonzero], dest[nonzero]
                    chunk_ids, inverse = np.unique((rows//block)*blocks+cols//block, return_inverse=True)
                    tiles = np.zeros((len(chunk_ids), block, block), dtype="<f8")
                    tiles[inverse, rows % block, cols % block] = values[nonzero]
                    dataset = data.create_dataset(table.name, shape=(n,n), dtype="<f8", chunks=(block,block),
                                                  compression="gzip", compression_opts=1, shuffle=True, fillvalue=0.)
                    for chunk_id, compressed in zip(chunk_ids, pool.map(compress_tile, tiles)):
                        dataset.id.write_direct_chunk((int(chunk_id//blocks)*block, int(chunk_id % blocks)*block),
                                                      compressed, filter_mask=0)
                        event["compressed_chunk_bytes"] += len(compressed)
                    event["written_chunks"] += len(chunk_ids)
                    event["zero_only_chunks_omitted"] += blocks*blocks-len(chunk_ids)
                    event["dense_logical_bytes"] += n*n*8
                    event["matrices"] += 1
                event["omx_files"] += 1
    event["seconds"] = time.perf_counter()-started
    return event
