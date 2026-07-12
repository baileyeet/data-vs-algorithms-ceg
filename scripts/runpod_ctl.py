"""Minimal RunPod REST client for this project. Money-touching actions require
--confirm AND print their cost estimate first; without --confirm every command
is a dry run that shows exactly what it would do.

Env: RUNPOD_API_KEY (never commit it; keep in .env.local or shell env).

Commands:
  list-gpus                      show on-demand H100 SXM availability/pricing
  list-pods / list-volumes       current account state
  create-volume --name N --size-gb 300 --datacenter DC          [--confirm]
  create-pod --preset cpu|h100x1|h100x8 --volume-id V [--name N] [--confirm]
  terminate --pod-id ID                                          [--confirm]

Presets (image: runpod pytorch 2.x; volume mounted at /workspace):
  cpu     cpu3c-2-4 (2 vCPU) — data prep       (~$0.10-0.30/hr)
  h100x1  1x H100 SXM — smoke tests            (~$3-4/hr)
  h100x8  8x H100 SXM one node NVLink — tiers  (~$21-32/hr)
"""

import argparse
import json
import os
import sys
import urllib.request

BASE = "https://rest.runpod.io/v1"

PRESETS = {
    "cpu":    {"computeType": "CPU", "cpuFlavorIds": ["cpu3c"], "vcpuCount": 2,
               "est": "$0.10-0.30/hr"},
    "h100x1": {"computeType": "GPU", "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
               "gpuCount": 1, "est": "$3-4/hr"},
    "h100x8": {"computeType": "GPU", "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
               "gpuCount": 8, "est": "$21-32/hr"},
}
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def req(method, path, body=None):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY not set")
    r = urllib.request.Request(
        f"{BASE}{path}", method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.loads(resp.read() or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {path}: {e.read().decode()[:500]}")


def gate(args, description, est):
    print(f"ACTION: {description}\nESTIMATED COST: {est}")
    if not args.confirm:
        print("DRY RUN — re-run with --confirm to execute (spends money)")
        sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list-gpus", "list-pods", "list-volumes",
                                    "create-volume", "create-pod", "terminate"])
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--volume-id")
    ap.add_argument("--pod-id")
    ap.add_argument("--name", default="ceg")
    ap.add_argument("--size-gb", type=int, default=300)
    ap.add_argument("--datacenter", default=None)
    args = ap.parse_args()

    if args.cmd == "list-gpus":
        gpus = req("GET", "/gputypes")
        for g in gpus:
            if "H100" in g.get("id", "") or "H100" in g.get("displayName", ""):
                print(json.dumps(g, indent=2))
    elif args.cmd == "list-pods":
        print(json.dumps(req("GET", "/pods"), indent=2))
    elif args.cmd == "list-volumes":
        print(json.dumps(req("GET", "/networkvolumes"), indent=2))
    elif args.cmd == "create-volume":
        if not args.datacenter:
            sys.exit("--datacenter required (pick one with H100 SXM availability "
                     "via list-gpus; volume and pods must share a datacenter)")
        est = f"~${args.size_gb * 0.07:.0f}/month storage"
        gate(args, f"create {args.size_gb}GB network volume '{args.name}' "
                   f"in {args.datacenter}", est)
        print(json.dumps(req("POST", "/networkvolumes",
                             {"name": args.name, "size": args.size_gb,
                              "dataCenterId": args.datacenter}), indent=2))
    elif args.cmd == "create-pod":
        if not args.preset:
            sys.exit("--preset required")
        p = PRESETS[args.preset]
        body = {"name": f"{args.name}-{args.preset}", "imageName": IMAGE,
                "cloudType": "SECURE", "computeType": p["computeType"],
                "containerDiskInGb": 50, "supportPublicIp": True,
                "startSsh": True}
        if p["computeType"] == "GPU":
            body |= {"gpuTypeIds": p["gpuTypeIds"], "gpuCount": p["gpuCount"],
                     "interruptible": False}
        else:
            body |= {"cpuFlavorIds": p["cpuFlavorIds"], "vcpuCount": p["vcpuCount"]}
        if args.volume_id:
            body |= {"networkVolumeId": args.volume_id, "volumeMountPath": "/workspace"}
        gate(args, f"create pod: {json.dumps(body, indent=2)}", p["est"])
        out = req("POST", "/pods", body)
        print(json.dumps(out, indent=2))
        print(f"\nssh: check `list-pods` for connection details once RUNNING")
    elif args.cmd == "terminate":
        if not args.pod_id:
            sys.exit("--pod-id required")
        gate(args, f"TERMINATE pod {args.pod_id} (unsaved container disk is lost; "
                   f"network volume persists)", "stops billing")
        req("DELETE", f"/pods/{args.pod_id}")
        print("terminated")


if __name__ == "__main__":
    main()
