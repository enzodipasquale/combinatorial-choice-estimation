import numpy as np


class BundleStore:
    """Bit-packed, deduplicated bundle store aligned with Gurobi cut rows."""

    def __init__(self, n_items):
        self.n_items = int(n_items)
        nb = (self.n_items + 7) // 8
        self.packed = np.empty((0, nb), dtype=np.uint8)
        self.cut_to_bundle = np.empty(0, dtype=np.int32)
        self.refcount = np.empty(0, dtype=np.int64)
        self._idx = {}

    def add_cuts(self, bundles):
        if len(bundles) == 0:
            return np.empty(0, dtype=np.int32)
        packed = np.packbits(np.asarray(bundles, dtype=bool), axis=1, bitorder='little')
        out, new, n = np.empty(len(packed), dtype=np.int32), [], len(self.packed)
        for i, row in enumerate(packed):
            key = row.tobytes()
            idx = self._idx.get(key)
            if idx is None:
                idx = n + len(new)
                self._idx[key] = idx
                new.append(row)
            out[i] = idx
        if new:
            self.packed = np.vstack([self.packed, np.stack(new)])
            self.refcount = np.concatenate([self.refcount, np.zeros(len(new), dtype=np.int64)])
        self.cut_to_bundle = np.concatenate([self.cut_to_bundle, out])
        np.add.at(self.refcount, out, 1)
        return out

    def prune(self, keep_mask):
        keep_mask = np.asarray(keep_mask, dtype=bool)
        np.subtract.at(self.refcount, self.cut_to_bundle[~keep_mask], 1)
        self.cut_to_bundle = self.cut_to_bundle[keep_mask]
        alive = self.refcount > 0
        if alive.all():
            return
        remap = (np.cumsum(alive) - 1).astype(np.int32)
        self._idx = {self.packed[i].tobytes(): int(remap[i]) for i in np.nonzero(alive)[0]}
        self.packed, self.refcount = self.packed[alive], self.refcount[alive]
        if self.cut_to_bundle.size:
            self.cut_to_bundle = remap[self.cut_to_bundle]

    def get(self, cut_indices):
        idx = self.cut_to_bundle[np.asarray(cut_indices, dtype=np.int64)]
        return np.unpackbits(self.packed[idx], axis=1, bitorder='little',
                             count=self.n_items).astype(bool)

    def state(self):
        return {'packed': self.packed, 'cut_to_bundle': self.cut_to_bundle,
                'refcount': self.refcount}

    @classmethod
    def from_state(cls, n_items, s):
        store = cls(n_items)
        store.packed = np.ascontiguousarray(s['packed'], dtype=np.uint8)
        store.cut_to_bundle = np.ascontiguousarray(s['cut_to_bundle'], dtype=np.int32)
        store.refcount = np.ascontiguousarray(s['refcount'], dtype=np.int64)
        store._idx = {store.packed[i].tobytes(): i for i in range(len(store.packed))}
        return store

    def save(self, path):
        np.savez(path, n_items=self.n_items, **self.state())

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls.from_state(int(d['n_items']), d)


def n_bundle_float_slots(n_items):
    """Float64 slots needed to hold one bit-packed bundle (padded to 8-byte alignment)."""
    return ((n_items + 7) // 8 + 7) // 8


def pack_bundles_to_float(bundles_bool):
    """(n, n_items) bool -> (n, n_slots) float64. Packed + padded to 8-byte alignment."""
    packed = np.packbits(np.asarray(bundles_bool, dtype=bool), axis=-1, bitorder='little')
    nb = packed.shape[-1]
    pad = (8 - nb % 8) % 8
    if pad:
        packed = np.pad(packed, [(0, 0)] * (packed.ndim - 1) + [(0, pad)])
    return np.ascontiguousarray(packed).view(np.float64)


def unpack_bundles_from_float(float_slots, n_items):
    """(n, n_slots) float64 -> (n, n_items) bool."""
    nb = (n_items + 7) // 8
    bv = np.ascontiguousarray(float_slots).view(np.uint8).reshape(float_slots.shape[0], -1)
    return np.unpackbits(bv[:, :nb], axis=-1, bitorder='little', count=n_items).astype(bool)
