# Stage Initialization Hangs With Partially Overlapping GPU Device Sets

## Description

When running `ming_flash_omni`, stage initialization can get stuck before any vLLM worker process is launched if two stages use partially overlapping `devices` configs.

The default `ming_flash_omni.yaml` has this layout:

```yaml
stages:
  - stage_id: 0
    tensor_parallel_size: 4
    devices: "0,1,2,3"

  - stage_id: 1
    devices: "3"
```

Stage 0 and stage 1 both need GPU 3. Because the initialization grouping appears to use the raw `devices` string as the group key, these two stages are treated as independent groups:

```text
stage 0 -> device:0,1,2,3
stage 1 -> device:3
```

They are initialized in parallel, but both need `/tmp/vllm_omni_device_3_init.lock`. This causes stage 1 to wait in `acquire_device_locks()` while stage 0 is still in the stage launch path. In practice this looks like the process is stuck before model loading.

## Reproduction

Environment:

```text
GPUs: 8x NVIDIA H20
Model: Jonathan1909/Ming-flash-omni-2.0
Config: vllm_omni/deploy/ming_flash_omni.yaml
```

Run:

```bash
cd examples/offline_inference/ming_flash_omni
python end2end.py --query-type text --modalities audio --output-dir output_ming_omni_speech
```

Relevant config:

```yaml
pipeline: ming_flash_omni
async_chunk: false

stages:
  - stage_id: 0
    tensor_parallel_size: 4
    devices: "0,1,2,3"

  - stage_id: 1
    devices: "3"
```

After the model files are downloaded/reconstructed, the process remains alive, but no GPU worker starts.

## Observed Behavior

`nvidia-smi` shows:

```text
No running processes found
```

No child process exists:

```bash
pgrep -P <pid> -a
```

`py-spy dump --pid <pid>` shows:

```text
MainThread:
  wait
  result
  _wait_for_orchestrator_init
  AsyncOmniEngine.__init__

orchestrator:
  as_completed
  _initialize_stage_replicas
  initialize

stage-init_0:
  scoped_spawn_device_env
  launch_stage_replica
  _initialize_local_llm_replica

stage-init_1:
  acquire_device_locks
  _initialize_local_llm_replica
```

Device lock files are held by the same parent process:

```text
/tmp/vllm_omni_device_0_init.lock: <pid>
/tmp/vllm_omni_device_1_init.lock: <pid>
/tmp/vllm_omni_device_2_init.lock: <pid>
/tmp/vllm_omni_device_3_init.lock: <pid>
```

`lslocks` confirms:

```text
python <pid> FLOCK WRITE /tmp/vllm_omni_device_0_init.lock
python <pid> FLOCK WRITE /tmp/vllm_omni_device_1_init.lock
python <pid> FLOCK WRITE /tmp/vllm_omni_device_2_init.lock
python <pid> FLOCK WRITE /tmp/vllm_omni_device_3_init.lock
```

## Expected Behavior

Stages with overlapping physical devices should not be initialized concurrently. In this case, stage 0 and stage 1 should be serialized because both include GPU 3.

## Actual Behavior

The stages are initialized concurrently because their `devices` strings are different, even though their physical device sets overlap. Stage 1 then waits on the GPU 3 init lock.

## Likely Cause

In `StageRuntime._replica_init_group_key`, the init group key is based on the raw `devices` value:

```python
devices = runtime_cfg.get("devices")
return f"device:{devices}"
```

So these are treated as different groups:

```text
"0,1,2,3"
"3"
```

But they overlap on physical GPU 3.

## Workaround

Move stage 1 to a non-overlapping device:

```yaml
stage 1:
  devices: "4"
```

With 8 GPUs this avoids the issue.

## Suggested Fix

Group local stage initialization by physical device overlap, not by exact `devices` string equality. For example, build init groups so that any replicas whose resolved physical device sets intersect are serialized.

In this case:

```text
stage 0 devices = {0,1,2,3}
stage 1 devices = {3}
```

should be placed into the same initialization group.
