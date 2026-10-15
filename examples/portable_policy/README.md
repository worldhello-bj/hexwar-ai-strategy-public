# Portable policy inference source

Source-only companion to separately delivered private policy exports. This directory
contains no trained models, scenario fixtures, provenance manifests, or run results.

Use Python 3.10+ and install `requirements-neural.txt` for NumPy inference, or
`requirements-sparse.txt` for exports configured with the SciPy CSR backend.
The pinned versions match the private export runtime; do not silently change the
saved backend when comparing exact probabilities.

From this directory:

```sh
python -m pip install -r requirements-sparse.txt
python infer.py /path/to/private-model.npz /path/to/public-input.json --seed 123
python smoke_test.py
```

Input JSON has `observation` and `candidates` keys. The trusted host must supply
only the active seat's detached schema-3 public observation and currently legal
candidate views from `SpatialObservationAdapter`. The result includes a sampled
index, its exact advertised engine action, and all candidate probabilities.
See `integration_example.py` for host integration. The host owns legality and
public-information guarantees; this facade is not a security sandbox.

`hexwar_policy/neural_policy.py` and `compact_checkpoint.py` are byte-for-byte
copies of training source revision `fabd416e150c276d04e4b8f10c9feef364682892`.
Use the matching simulator observation/scoring contract separately. The portable
runtime imports neither the simulator nor training opponent pools for inference.
The copied policy retains training methods for source compatibility, but this
package supports inference only; optimizer diagnostics are not packaged.

The `hexwar-inference-weights` version-1 NPZ format preserves policy configuration
and learned tensors, omitting optimizer state, training RNG, counters and pending
trajectories. Load it using `load_weights` or `GamePolicy`, not `NeuralPolicy.load`.
The facade uses an independent sampling RNG. These exports cannot resume training.
Loading uses numeric arrays with `allow_pickle=False`, size/dtype/shape/finite
checks; accept model files only from a trusted source.

`smoke_test.py` constructs tiny, untrained synthetic weights in a temporary
folder, tests exact roundtrip probabilities, deterministic sampling and input
non-mutation, then removes the temporary files. It requires no game data or
trained model. It is an inference-contract check, not a playing-strength test.
Separate private-pack verification covered 36 public-view probability checks
across six exports; those private fixtures and results are not included here.

LICENSE covers the original project code. NOTICE is preserved from the source
project and describes the wider repository; its third-party material listing
does not mean those materials are included in this source-only directory.
