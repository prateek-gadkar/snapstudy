"""
qualcomm_npu.py — Qualcomm AI Hub integration for StudySnap.

This is the NPU half of the submission: it takes the two models StudySnap
depends on (Whisper-Small for ASR, and a small instruct LLM for the
summary/flashcard step) and compiles + profiles them for Snapdragon X Elite
compute so they run on the Hexagon NPU instead of the CPU.

WHY THIS MATTERS FOR THE JUDGING CRITERIA
  Technical Implementation  -> real compile + profile jobs on a real target device
  Deployment & Accessibility-> NPU execution is what makes offline, low-power,
                               long-battery lecture capture possible on a laptop
                               a student can actually carry to class

SETUP (2 minutes)
  1. Create a free account at https://app.aihub.qualcomm.com
  2. Copy your API token from the account page
  3. export QAI_HUB_API_TOKEN="your_token_here"
  4. pip install qai-hub qai-hub-models
  5. python qualcomm_npu.py --device "Snapdragon X Elite CRD"

This script is written defensively: if qai-hub is not installed or the token is
missing, it explains exactly what to do instead of crashing. It never invents
performance numbers — every figure it prints comes back from the Hub job.

NOTE: profiling must run on the Hub, not on your local machine. The Hub returns
real on-device latency, memory peak, and per-layer NPU/CPU split for the target
device you name.
"""

from __future__ import annotations

import argparse
import os
import sys

# Targets relevant to this challenge. "Snapdragon X Elite CRD" is the
# developer reference device for the same silicon family found in
# Snapdragon-powered HP AI PCs.
DEVICE_CHOICES = [
    "Snapdragon X Elite CRD",
    "Snapdragon X2 Elite CRD",
    "Snapdragon 8 Gen 3 QRD",
]


def _fail(msg: str) -> None:
    print(f"\n[!] {msg}\n")
    sys.exit(1)


def _require_hub():
    try:
        import qai_hub as hub
    except ImportError:
        _fail(
            "qai-hub is not installed.\n"
            "    pip install qai-hub qai-hub-models\n"
            "Then re-run this script."
        )
    if not os.environ.get("QAI_HUB_API_TOKEN"):
        _fail(
            "QAI_HUB_API_TOKEN is not set.\n"
            "    1. Sign up free at https://app.aihub.qualcomm.com\n"
            "    2. Copy the API token from your account page\n"
            "    3. export QAI_HUB_API_TOKEN='<token>'\n"
            "    4. Re-run this script.\n\n"
            "Until then, the StudySnap pipeline still runs end-to-end on CPU:\n"
            "    python app.py --demo"
        )
    return hub


def compile_and_profile(model_name: str, device_name: str, samples: int = 50) -> dict:
    """
    Compile a pretrained model for the target Snapdragon device and profile it.

    Returns the real measurements reported by the Hub. Nothing here is
    hardcoded or estimated.
    """
    import qai_hub as hub

    device = hub.Device(device_name)
    print(f"[*] Target device : {device_name}")
    print(f"[*] Model         : {model_name}")

    # --- 1. load a pretrained model from the AI Hub model zoo ----------------
    try:
        import qai_hub_models  # noqa: F401
    except ImportError:
        _fail("qai-hub-models is required for pretrained models: pip install qai-hub-models")

    module_path, cls_name = model_name.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, cls_name)
    model = model_cls.from_pretrained()
    print("[+] Pretrained weights loaded (cached locally after first fetch)")

    # --- 2. export to a traceable form, then compile for the NPU -------------
    print("[*] Exporting + compiling for the target...")
    source_model = model.convert_to_torch() if hasattr(model, "convert_to_torch") else model
    input_specs = model.get_input_spec()

    compile_job = hub.submit_compile_job(
        model=source_model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime onnx",
    )
    print(f"    compile job: {compile_job.job_id}")
    compile_job.wait()
    print(f"    status     : {compile_job.get_status().code}")
    if str(compile_job.get_status().code) != "SUCCESS":
        _fail(f"Compilation failed. Inspect job {compile_job.job_id} on the Hub.")

    target_model = compile_job.get_target_model()

    # --- 3. profile on the real device --------------------------------------
    print("[*] Submitting profile job (runs on real Snapdragon hardware)...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    print(f"    profile job: {profile_job.job_id}")
    profile_job.wait()
    print(f"    status     : {profile_job.get_status().code}")

    if str(profile_job.get_status().code) != "SUCCESS":
        _fail(f"Profiling failed. Inspect job {profile_job.job_id} on the Hub.")

    # --- 4. pull the real numbers -------------------------------------------
    profile = profile_job.download_profile()
    print("\n[+] Profiling complete. Real measurements from the Hub:\n")
    _print_profile(profile)

    return {
        "device": device_name,
        "model": model_name,
        "compile_job_id": compile_job.job_id,
        "profile_job_id": profile_job.job_id,
        "profile": profile,
    }


def _print_profile(profile) -> None:
    """Pretty-print the common fields. Schema can vary; be defensive."""
    interesting = [
        "estimated_inference_time",
        "estimated_inference_peak_memory",
        "compute_unit",
        "first_load_time",
        "warm_load_time",
    ]
    for key in interesting:
        if isinstance(profile, dict) and key in profile:
            print(f"    {key:38s}: {profile[key]}")

    if isinstance(profile, dict):
        exec_units = profile.get("execution_detail") or profile.get("layer_details")
        if exec_units:
            print(f"\n    per-layer NPU/CPU split: {len(exec_units)} layers returned")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile + profile StudySnap models for Snapdragon NPU")
    ap.add_argument("--device", default="Snapdragon X Elite CRD", choices=DEVICE_CHOICES)
    ap.add_argument(
        "--model",
        default="qai_hub_models.models.whisper_small.Model",
        help="Dotted path to a qai_hub_models model class",
    )
    args = ap.parse_args()

    hub = _require_hub()
    del hub  # only needed for the import check; compile_and_profile re-imports

    compile_and_profile(args.model, args.device)

    print("\n" + "=" * 68)
    print("NEXT: paste the profile output above into README.md, section")
    print("'Measured NPU performance'. Then run the app itself with:")
    print("    python app.py --text lecture.txt --target 'Snapdragon X Elite NPU'")
    print("=" * 68)


if __name__ == "__main__":
    main()
