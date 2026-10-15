"""Bounded, pickle-free optional NPZ checkpoint transport (format version 1).

This is a transport, not a replacement for NeuralTrainer's semantic validation.
Only .npz paths select it; existing JSON files keep their original representation.
"""
import json
import math
from pathlib import Path
import zipfile
import numpy as np

MAX_METADATA = 16 * 1024 * 1024
MAX_ARRAY = 128 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024
MAX_MEMBERS = 100000


def json_tree(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_tree(v) for v in value]
    return value


def _array_spec(path, shape, dtype):
    # Policy parameter groups, or sparse feature index/value pairs in decisions.
    if len(path) >= 2 and path[-2] in ('params', 'm', 'v'):
        if dtype != np.dtype('float64') or len(shape) not in (1, 2):
            raise ValueError('invalid compact policy array dtype/shape')
    elif ('pending' in path and 'decisions' in path and
          path[-2:] in (('$packed_features', 'indices'), ('$packed_features', 'values'), ('$packed_features', 'offsets'))):
        expected = np.dtype('float64' if path[-1] == 'values' else 'int64')
        if dtype != expected or len(shape) != 1:
            raise ValueError('invalid packed sparse feature dtype/shape')
    elif 'pending' in path and 'decisions' in path and path[-1] in (0, 1):
        expected = np.dtype('int64' if path[-1] == 0 else 'float64')
        if dtype != expected or len(shape) != 1:
            raise ValueError('invalid compact sparse feature dtype/shape')
    else:
        raise ValueError('unexpected compact array location')
    if math.prod(shape) * dtype.itemsize > MAX_ARRAY:
        raise ValueError('compact tensor exceeds size limit')


def save_compact(path, data):
    arrays = {}
    def encode(value, path=()):
        if isinstance(value, np.ndarray):
            _array_spec(path, value.shape, value.dtype)
            name = 'a%06d' % len(arrays)
            arrays[name] = value
            return {'$array': name}
        if isinstance(value, dict):
            if len(path) == 4 and path[0] == 'pending' and path[2] == 'decisions':
                pairs = [value['observation'], *value['candidates']]
                lengths = [len(pair[0]) for pair in pairs]
                if any(len(pair[1]) != length for pair, length in zip(pairs, lengths)):
                    raise ValueError('invalid sparse feature pair')
                value = {k: v for k, v in value.items() if k not in ('observation', 'candidates')}
                value['$packed_features'] = {
                    'indices': np.concatenate([p[0] for p in pairs]),
                    'values': np.concatenate([p[1] for p in pairs]),
                    'offsets': np.asarray([0, *np.cumsum(lengths, dtype=np.int64)], dtype=np.int64)}
            if '$array' in value:
                raise ValueError('reserved compact metadata key')
            return {k: encode(v, path + (k,)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v, path + (i,)) for i, v in enumerate(value)]
        return value
    metadata = json.dumps({'format': 'hexwar-npz', 'version': 1, 'data': encode(data)},
                          allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(metadata) > MAX_METADATA or len(arrays) + 1 > MAX_MEMBERS:
        raise ValueError('compact metadata exceeds size limit')
    if sum(a.nbytes + 256 for a in arrays.values()) + len(metadata) > MAX_TOTAL:
        raise ValueError('compact checkpoint exceeds size limit')
    # Exclusive creation retains the original no-overwrite contract. Uncompressed
    # ZIP writes arrays one at a time, without float lists or compression buffers.
    with Path(path).open('xb') as stream:
        np.savez(stream, metadata=np.frombuffer(metadata, dtype=np.uint8), **arrays)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate compact metadata key')
        result[key] = value
    return result


def read_checkpoint(path):
    if Path(path).suffix.lower() != '.npz':
        return json.loads(Path(path).read_text('utf-8'))
    if Path(path).stat().st_size > MAX_TOTAL + MAX_METADATA:
        raise ValueError('compact archive exceeds physical size limit')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if (len(names) > MAX_MEMBERS or len(set(names)) != len(names) or
                'metadata.npy' not in names or sum(m.file_size for m in members) > MAX_TOTAL):
            raise ValueError('invalid compact archive size/members')
        headers = {}
        for member in members:
            if member.filename != 'metadata.npy' and not (member.filename.startswith('a') and
                    member.filename[1:-4].isdigit() and member.filename.endswith('.npy')):
                raise ValueError('unexpected compact archive member')
            limit = MAX_METADATA if member.filename == 'metadata.npy' else MAX_ARRAY
            if member.file_size > limit + 65536 or member.flag_bits & 1:
                raise ValueError('invalid compact archive member size')
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError('unsupported compact NPY version')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                if (dtype.hasobject or dtype not in (np.dtype('uint8'), np.dtype('float64'), np.dtype('int64'))
                        or len(shape) > 2 or any(type(d) is not int or d < 0 for d in shape)
                        or math.prod(shape) * dtype.itemsize > limit or fortran
                        or stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size):
                    raise ValueError('invalid compact NPY header')
                headers[member.filename[:-4]] = (shape, dtype)
        if headers['metadata'][1] != np.dtype('uint8') or len(headers['metadata'][0]) != 1:
            raise ValueError('invalid compact metadata dtype/shape')
        class NumericMembers:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __getitem__(self, name):
                with archive.open(name + '.npy') as member:
                    return np.load(member, allow_pickle=False)
        with NumericMembers() as values:
            envelope = json.loads(values['metadata'].tobytes().decode('utf-8'), object_pairs_hook=_unique,
                                  parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
            if (set(envelope) != {'format', 'version', 'data'} or envelope['format'] != 'hexwar-npz'
                    or type(envelope['version']) is not int or envelope['version'] != 1):
                raise ValueError('unsupported compact checkpoint format')
            seen = set()
            def decode(value, path=(), depth=0):
                if depth > 64:
                    raise ValueError('compact metadata nesting exceeds limit')
                if isinstance(value, dict):
                    if '$array' in value:
                        name = value['$array']
                        if set(value) != {'$array'} or not isinstance(name, str) or name not in headers or name == 'metadata' or name in seen:
                            raise ValueError('invalid compact array reference')
                        seen.add(name)
                        _array_spec(path, *headers[name])
                        return values[name]
                    decoded = {k: decode(v, path + (k,), depth + 1) for k, v in value.items()}
                    if '$packed_features' in decoded:
                        if len(path) != 4 or path[0] != 'pending' or path[2] != 'decisions':
                            raise ValueError('unexpected packed feature location')
                        if not {'$packed_features', 'selected'} <= set(decoded) or set(decoded) - {'$packed_features', 'selected', 'public_score_offset'}:
                            raise ValueError('invalid packed decision metadata')
                        packed = decoded.pop('$packed_features')
                        if not isinstance(packed, dict) or set(packed) != {'indices', 'values', 'offsets'}:
                            raise ValueError('invalid packed feature fields')
                        indices, amounts, offsets = (packed[k] for k in ('indices', 'values', 'offsets'))
                        if not all(isinstance(a, np.ndarray) for a in (indices, amounts, offsets)):
                            raise ValueError('packed features require numeric arrays')
                        if (len(indices) != len(amounts) or len(offsets) < 3 or offsets[0] != 0 or
                                offsets[-1] != len(indices) or np.any(offsets[1:] < offsets[:-1])):
                            raise ValueError('invalid packed sparse offsets')
                        pairs = [[indices[a:b], amounts[a:b]] for a, b in zip(offsets[:-1], offsets[1:])]
                        decoded['observation'], decoded['candidates'] = pairs[0], pairs[1:]
                    return decoded
                if isinstance(value, list):
                    return [decode(v, path + (i,), depth + 1) for i, v in enumerate(value)]
                return value
            data = decode(envelope['data'])
            if seen != set(headers) - {'metadata'}:
                raise ValueError('unreferenced compact arrays')
    # Bound configuration allocations before constructing a policy. All scoring,
    # source/snapshot, counter and pending-decision guards still run in the loader.
    policies = [data] if 'config' in data else [data['policy'], data['initial_policy']]
    for policy in policies:
        config = policy['config']
        for key in ('buckets', 'hidden_dim'):
            if type(config[key]) is not int or not 1 <= config[key] <= 1048576:
                raise ValueError('invalid compact policy dimensions')
        if config['buckets'] * config['hidden_dim'] * 16 > MAX_ARRAY:
            raise ValueError('compact policy dimensions exceed limit')
        for group in ('params', 'm', 'v'):
            if not all(isinstance(a, np.ndarray) for a in policy[group].values()):
                raise ValueError('compact policy tensors must be numeric arrays')
    if 'policy_config' in data and any(data['policy_config'].get(k, default) != data['policy']['config'][k]
                                      for k, default in [('buckets', 2048), ('hidden_dim', 32)]):
        raise ValueError('compact trainer policy dimensions differ')
    return data


def state_digest(value):
    """SHA256 of a typed, length-delimited state tree, without array float lists.

    This digest is for before/after invariance, not the on-disk file identity.
    Arrays include dtype, shape and exact C-order bytes (including signed zero).
    """
    import hashlib
    digest = hashlib.sha256()
    def token(kind, payload=b''):
        digest.update(kind)
        digest.update(str(len(payload)).encode('ascii') + b':' + payload)
    def walk(node):
        if isinstance(node, np.ndarray):
            if node.dtype.hasobject or not np.isfinite(node).all():
                raise ValueError('invalid state digest tensor')
            token(b'A', node.dtype.str.encode('ascii'))
            walk(list(node.shape))
            array = np.ascontiguousarray(node)
            view = memoryview(array).cast('B')
            digest.update(str(len(view)).encode('ascii') + b':')
            digest.update(view)
        elif isinstance(node, dict):
            token(b'D', str(len(node)).encode('ascii'))
            for key in sorted(node):
                if not isinstance(key, str):
                    raise ValueError('state dictionary keys must be strings')
                token(b'K', key.encode('utf-8')); walk(node[key])
        elif isinstance(node, (list, tuple)):
            token(b'L', str(len(node)).encode('ascii'))
            for item in node: walk(item)
        else:
            token(b'J', json.dumps(node, allow_nan=False, separators=(',', ':')).encode('utf-8'))
    walk(value)
    return digest.hexdigest()
