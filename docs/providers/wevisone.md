# WeVisOne-VL provider

ASPIRE's `WeVisOneProvider` is a standard text-segmentation provider client. Its
service loads Tencent's released `WeVisOne-VL-2B` checkpoint in a provider-only
environment and uses the official `task="segmentation"` inference path. Model
weights stay outside Git under `third_party/models/wevisone-vl-2b`. With no
`--model` argument or `WEVISONE_MODEL_PATH`, the service selects that local
directory when it exists and verifies the required files and weight header
before loading. A present but incomplete snapshot fails locally instead of
falling back to a new Hub download. Local paths force Hugging Face offline mode.
Set `WEVISONE_MODEL_PATH` to select another local location or Hub ID;
`third_party/...` paths are relative to the ASPIRE checkout.

The client omits empty thresholded masks. The service reports the model's
pre-limit `total_instances` and a `truncated` flag. The client returns
`ErrorCode.TRUNCATED` whenever the service omitted instances, so a single
returned mask cannot be mistaken for a unique target.
The client thresholds the returned probabilities at `0.5` by default because
ASPIRE's `Segmentation` contract carries a binary mask. The official model does
not emit an instance confidence score, so the required ASPIRE score is a neutral
`0.5` and diagnostics explicitly state that it is not model confidence.
Point-prompted segmentation is currently unsupported by this provider.

Run through the environment supervisor after registering the `wevisone`
provider in `configs/environments.json`. The provider is separate from the
robot SDK environment. Its model ID, host, port, and device are explicit service
arguments; no policy runner is started by this service. Validate the local files
without loading model weights or using a GPU:

```bash
PYTHONPATH=src python -m cap_harness.providers.wevisone.snapshot
```

The standalone service uses the offline-first resolver:

```bash
WEVISONE_MODEL_PATH=third_party/models/wevisone-vl-2b \
PYTHONPATH=third_party/WeVisOne-VL:src \
  python -m cap_harness.providers.wevisone.service --port 8117
```

The official WeVisOne source pins PyTorch 2.11.0 and Transformers 5.12.1. This
target currently has JetPack R36.3 with NVIDIA PyTorch 2.4.0a0; Transformers
4.57.0 fails to parse the release's tokenizer configuration, while 5.12.1
rejects the installed PyTorch prerelease version. Keep the service in a separate
runtime and do not launch it in the Open-RAIL environment until a compatible
Jetson runtime is available.
