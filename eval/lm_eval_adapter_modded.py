"""lm-eval adapter for modded-nanoGPT (A1) checkpoints. CUDA-only.

The modded model classes are defined inside train_new/train_gpt_ceg.py, which
is a script that begins distributed training at import. We therefore exec only
its prefix — everything up to (excluding) the Hyperparameters instantiation —
in an isolated namespace to obtain the class definitions, then rebuild the
model from the checkpoint's saved `model_args` and load its state dict
(stripping torch.compile's `_orig_mod.` prefixes).

Usage (on a GPU pod):
  python eval/lm_eval_adapter_modded.py --ckpt /workspace/runs/small_a1d1/ckpt_XXXX.pt \
    --tasks hellaswag --limit 500 --out results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUT_MARKER = "args = Hyperparameters()"


def load_modded_classes(trainer_file, pre_init_dist=False):
    """Exec the class-definition prefix of the CEG trainer script."""
    import os

    # the script prefix reads torchrun env at import; single-process defaults
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # PID-unique port: parallel single-process loaders must not share the
    # rendezvous port (7 of 8 crash on bind otherwise)
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
    # triton_kernels lives in the upstream clone next to the trainer
    upstream = str(Path(trainer_file).parent / "modded-nanogpt")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    import torch.distributed as dist

    if pre_init_dist and not dist.is_initialized():
        # medium track's classes assume an initialized default group; the
        # small track's prefix initializes it itself (would collide)
        dist.init_process_group("nccl", rank=0, world_size=1)
    src = Path(trainer_file).read_text()
    idx = src.find(CUT_MARKER)
    assert idx > 0, f"cut marker {CUT_MARKER!r} not found in {trainer_file}"
    prefix = src[:idx]
    # globals the class bodies read but which are defined after our cut point
    # (single-process eval values)
    ns = {"__name__": "modded_classes", "__file__": str(trainer_file),
          "device": "cuda", "grad_accum_steps": 1, "world_size": 1}
    # the prefix imports `ceg` (the wrapper registers itself under that name
    # when launched normally); provide a minimal stand-in
    import types

    ceg_stub = types.ModuleType("ceg")
    ceg_stub.CONFIG = types.SimpleNamespace(model_max_seq_len=0, out_dir="/tmp")
    ceg_stub.DataExhausted = type("DataExhausted", (Exception,), {})
    sys.modules.setdefault("ceg", ceg_stub)
    exec(compile(prefix, str(trainer_file), "exec"), ns)
    # the model classes read the module-global `args` (Hyperparameters);
    # instantiate it with defaults — eval only touches structural fields
    ns["args"] = ns["Hyperparameters"]()
    return ns


def restore_yarn(model, ckpt, is_medium):
    """Exact yarn restoration from checkpoints that carry yarn_state (runs
    after the reload-fidelity fix). Returns (ws_short, ws_long) or None if
    the checkpoint predates yarn saving (caller falls back to _replay_yarn,
    which is approximate — label results accordingly)."""
    ys = ckpt.get("yarn_state")
    if not ys:
        return None
    for name, state in ys.items():
        y = getattr(model, name, None)
        if y is None:
            continue
        for attr in ("angular_freq", "factor1", "factor2", "cos", "sin"):
            if attr in state and getattr(y, attr, None) is not None:
                getattr(y, attr).copy_(state[attr].to(getattr(y, attr).device))
        if "attn_scale" in state and hasattr(y, "attn_scale"):
            y.attn_scale = state["attn_scale"]
    ws = ckpt.get("ws_state") or {}
    block = 1 if is_medium else 128
    if ws.get("ws_short") is not None:
        return ws["ws_short"] * block, ws["ws_long"] * block
    return None


def _set_split_embed(model, ckpt, ckpt_path, sd, split_embed_frac):
    """medium ties embed to lm_head.weight until split_step, then unties; the
    flag is a plain attribute (never in state_dict), so a fresh model would
    run post-split checkpoints with the diverged lm_head as the embedding
    (measured: +0.38..0.57 BPB, growing with training). Tied weights make
    the flag irrelevant (set False for a stable compile guard); diverged
    weights need the schedule to disambiguate post-split from pre-split
    (where embed still holds its unused init)."""
    if torch.equal(sd["embed.weight"], sd["lm_head.weight"]):
        model.split_embed = False
        return
    cfg_p = Path(ckpt_path).parent / "run_config.json"
    if not cfg_p.exists():
        sys.exit("embed/lm_head diverged but no run_config.json to "
                 "locate split_step — cannot set split_embed")
    sched = json.loads(cfg_p.read_text())["num_scheduled_iterations"]
    split_step = math.ceil(split_embed_frac * sched) | 1
    model.split_embed = ckpt["step"] >= split_step


def _replay_yarn(model, ckpt_path, ckpt, is_medium):
    """Replay the training-time YaRN mutations up to this checkpoint's step.

    The rotary factor buffers are persistent=False (never saved) and training
    permanently mutates angular_freq at each long-window growth via
    Yarn.apply(old, new). A fresh model has the un-mutated state; without the
    replay, reloaded models attend with mis-calibrated rotary (measured: +0.18
    BPB small, +0.84 medium vs the training-recorded evals).

    Returns (ws_short, ws_long) for the checkpoint's forward config.
    """
    import json

    step = ckpt["step"]
    cfg_p = Path(ckpt_path).parent / "run_config.json"
    total = max(json.loads(cfg_p.read_text())["ckpt_steps"]) if cfg_p.exists() else step
    is_final = step >= total
    if is_medium:
        # scheduled steps: total minus extension at the upstream 40/4740 ratio
        sched = total - max(1, round(total * 40 / 4740))
        ws_sched = (3, 7, 11, 13, 15, 17, 19, 21, 23, 23, 23, 23)
        transitions = []
        prev = ws_sched[0]
        for k in range(1, len(ws_sched)):
            if ws_sched[k] != prev:
                transitions.append((-(-k * sched // len(ws_sched)), prev, ws_sched[k]))
                prev = ws_sched[k]
        cur = ws_sched[0]
        for boundary, old, new in transitions:
            if step >= boundary and new <= 13:  # training only yarns while <=13
                model.yarn.apply(old, new)
            if step >= boundary:
                cur = new
        if is_final:
            return 11, 27  # ws_final//2, ws_validate_post_yarn_ext
        return min(11, cur // 2), cur
    else:
        sched = total - max(1, round(total * 10 / 1390))
        stages = [(1, 3), (3, 7), (5, 11), (6, 13)]  # (short, long) per stage
        bounds = [round(sched / 3), round(2 * sched / 3), sched]
        cur_s, cur_l = stages[0]
        for i, b in enumerate(bounds):
            if step >= b:
                old_l, (new_s, new_l) = cur_l, stages[i + 1]
                model.yarn.apply(old_l * 128, new_l * 128)
                model.yarn_paired_head.apply(old_l * 128, new_l * 128)
                cur_s, cur_l = new_s, new_l
        if is_final:
            return 6 * 128, 20 * 128  # ext-stage short, post-yarn-ext long
        return cur_s * 128, cur_l * 128


def build_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = ckpt["model_args"]
    # checkpoints carry "size", not "algorithm" — route the class source by it
    is_medium = ckpt.get("size") == "medium" or "medium" in ckpt.get("algorithm", "")
    trainer = (ROOT / "train_new" /
               ("train_gpt_medium_ceg.py" if is_medium else "train_gpt_ceg.py"))
    ns = load_modded_classes(trainer, pre_init_dist=is_medium)
    # older checkpoints omit max_seq_len from model_args; infer from the saved
    # rotary table (yarn.factor1 has 2*max_seq_len rows) or default to 262144
    if "max_seq_len" not in margs:
        key = next((k for k in ckpt["model"]
                    if k.removeprefix("_orig_mod.").endswith("yarn.factor1")), None)
        margs["max_seq_len"] = (ckpt["model"][key].shape[0] // 2 if key else 262144)
    model = ns["GPT"](**margs).cuda()
    sd = {k.removeprefix("_orig_mod."): v.cuda() for k, v in ckpt["model"].items()}
    # distributed training pads the weight banks to multiples of world_size
    # (e.g. qk_bank 60->64 rows at world 8); single-process reload expects the
    # unpadded size — padding rows are appended, so truncate from the end.
    # Sanity of this assumption is checked by the caller (score vs chance).
    msd = model.state_dict()
    for k, v in list(sd.items()):
        if k in msd and v.shape != msd[k].shape:
            # world-size sharding pads exactly one dimension; slice it back
            diff = [i for i in range(v.ndim) if v.shape[i] != msd[k].shape[i]]
            assert len(diff) == 1 and v.shape[diff[0]] > msd[k].shape[diff[0]], \
                f"unexpected mismatch for {k}: {v.shape} vs {msd[k].shape}"
            sd[k] = v.narrow(diff[0], 0, msd[k].shape[diff[0]])
    # assign=True: parameters adopt the checkpoint's dtypes (the trainer casts
    # banks/linears to bf16 post-construction; plain load would upcast to the
    # fresh model's fp32 and break the bf16/fp8-only kernels)
    model.load_state_dict(sd, assign=True)
    model.eval()
    if is_medium:
        _set_split_embed(model, ckpt, ckpt_path, sd,
                         ns["args"].split_embed_frac)
    # validated instrument formula (fidelity delta 0.0024 vs recorded eval):
    # exact yarn restoration BEFORE torch.compile; compiled numerics match the
    # training eval's fp8 paths where eager does not (+0.02 BPB eager bias)
    ws = restore_yarn(model, ckpt, is_medium)
    model = torch.compile(model)
    return model, ckpt, ns, ws


class ModdedLM:
    """Minimal loglikelihood interface over the modded model's forward API.

    The modded forward is loss-oriented (returns per-position losses in eval
    mode via the CEG patch), so continuation logprob = sum of target-position
    losses (negated). Requests are packed one sequence per forward with varlen
    cu_seqlens, mirroring the wrapper's BPB shim.
    """

    def __init__(self, model, ns, batch_tokens=16384, is_medium=False):
        import tiktoken

        self.model = model
        self.ns = ns
        self.enc = tiktoken.get_encoding("gpt2")
        self.batch_tokens = batch_tokens
        self.is_medium = is_medium
        self._cfg = None

    def _cfg_lazy(self):
        if self._cfg is None:
            # end-of-training forward config, matching each track's final
            # in-training eval state (pure next-token MTP weights)
            if self.is_medium:
                # medium: ws in block units passed raw; final ws_short =
                # ws_final//2 = 11, ws_long = ws_validate_post_yarn_ext = 27
                self._cfg = self.ns["ForwardScheduleConfig"](
                    mtp_weights=torch.tensor([1.0], device="cuda"),
                    ws_short=11, ws_long=27)
            else:
                # small: extension-stage windows with final YaRN ext
                # (short 6 blocks, long 20 blocks, block=128)
                self._cfg = self.ns["ForwardScheduleConfig"](
                    mtp_weights=torch.tensor([1.0], device="cuda"),
                    ws_short=6 * 128, ws_long=20 * 128, train_max_seq_len=2048)
        return self._cfg

    def _packed_losses(self, seqs):
        """One forward over many sequences packed with varlen cu_seqlens
        (mirrors the training-eval shim's shape — the fused CE kernel needs
        large flat batches, not tiny per-request calls). Returns per-position
        losses for each sequence."""
        flat, cu = [], [0]
        for ids in seqs:
            flat.extend(ids)
            cu.append(len(flat))
        pad = (-len(flat)) % 128
        if pad:
            flat.extend([50256] * pad)
            cu.append(len(flat))  # padding tail is its own ignored segment
        toks_cpu = torch.tensor(flat, dtype=torch.int32)
        toks = toks_cpu.cuda()
        targets = torch.roll(toks, -1).to(torch.int64)  # per-segment shift handled below
        cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
        with torch.no_grad():
            if self.is_medium:
                # medium forward takes no bigram inputs
                losses = self.model(toks, targets, cu_t, self._cfg_lazy())
            else:
                bigram = self.ns["get_bigram_hash"](toks_cpu).cuda(non_blocking=True)
                losses = self.model(toks, targets, cu_t, bigram, self._cfg_lazy())
        # losses[i] = loss predicting flat[i+1] from prefix within its segment;
        # positions at segment ends predict across boundaries — never read them
        out = []
        for s, e in zip(cu, cu[1:]):
            out.append(losses[s : e - 1])
            if len(out) == len(seqs):
                break
        return out

    def loglikelihood(self, requests):
        encoded = []
        for inst in requests:
            context, continuation = inst.args
            ctx = self.enc.encode_ordinary(context) if context else [50256]
            cont = self.enc.encode_ordinary(continuation)
            encoded.append((ctx, cont))
        out = []
        batch, meta = [], []
        def flush():
            if not batch:
                return
            for losses, (ctx, cont) in zip(self._packed_losses(batch), meta):
                cl = losses[len(ctx) - 1 : len(ctx) + len(cont) - 1]
                out.append((-cl.float().sum().item(), False))
            batch.clear(); meta.clear()
        for ctx, cont in encoded:
            ids = ctx + cont
            if sum(len(b) for b in batch) + len(ids) > self.batch_tokens:
                flush()
            batch.append(ids)
            meta.append((ctx, cont))
        flush()
        return out


def load_into(model, ckpt_path):
    """Swap another checkpoint's weights into an already-built (compiled)
    model — same architecture required. Returns (ckpt, ws)."""
    inner = getattr(model, "_orig_mod", model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_medium = ckpt.get("size") == "medium" or "medium" in ckpt.get("algorithm", "")
    sd = {k.removeprefix("_orig_mod."): v.cuda() for k, v in ckpt["model"].items()}
    msd = inner.state_dict()
    for k, v in list(sd.items()):
        if k in msd and v.shape != msd[k].shape:
            diff = [i for i in range(v.ndim) if v.shape[i] != msd[k].shape[i]]
            assert len(diff) == 1 and v.shape[diff[0]] > msd[k].shape[diff[0]]
            sd[k] = v.narrow(diff[0], 0, msd[k].shape[diff[0]])
    inner.load_state_dict(sd)  # dtypes already correct in the built model
    if is_medium:
        # default 2/3/4 matches Hyperparameters.split_embed_frac (the wrapper
        # never overrides it); flag flip recompiles once per branch, amortized
        _set_split_embed(inner, ckpt, ckpt_path, sd, 2 / 3 / 4)
    ws = restore_yarn(inner, ckpt, is_medium)
    return ckpt, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="first checkpoint (defines the architecture)")
    ap.add_argument("--more-ckpts", nargs="*", default=[],
                    help="additional same-arch checkpoints evaluated in the same "
                         "process (compile amortized); each writes <out-dir>/<name>.json")
    ap.add_argument("--out-dir", default=None, help="required with --more-ckpts")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, ckpt, ns, ws = build_model(args.ckpt)
    is_medium = ckpt.get("size") == "medium" or "medium" in ckpt.get("algorithm", "")
    if ws is None:
        sys.exit("checkpoint lacks yarn_state — pre-fix legacy checkpoint; post-hoc "
                 "eval is only validated for yarn-saving runs (rerun the arm)")
    print(f"loaded {args.ckpt}: step={ckpt.get('step')} size={ckpt.get('size')} ws={ws}")

    import lm_eval
    from lm_eval.api.model import LM

    inner = ModdedLM(model, ns, is_medium=is_medium)

    class Wrapper(LM):
        def loglikelihood(self, requests):
            return inner.loglikelihood(requests)

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

        def generate_until(self, requests):
            raise NotImplementedError

    def set_cfg(ws_pair):
        if is_medium:
            inner._cfg = ns["ForwardScheduleConfig"](
                mtp_weights=torch.tensor([1.0], device="cuda"),
                ws_short=ws_pair[0], ws_long=ws_pair[1])
        else:
            inner._cfg = ns["ForwardScheduleConfig"](
                mtp_weights=torch.tensor([1.0], device="cuda"),
                ws_short=ws_pair[0], ws_long=ws_pair[1], train_max_seq_len=2048)

    def eval_one(name, out_path):
        results = lm_eval.simple_evaluate(model=Wrapper(), tasks=args.tasks.split(","),
                                          limit=args.limit)
        payload = json.dumps(results["results"], indent=2, default=str)
        if out_path:
            Path(out_path).write_text(payload)
        print(f"done {name}")

    set_cfg(ws)
    eval_one(args.ckpt, args.out)
    for extra in args.more_ckpts:
        ck2, ws2 = load_into(model, extra)
        if ws2 is None:
            print(f"SKIP {extra}: no yarn_state")
            continue
        set_cfg(ws2)
        out2 = str(Path(args.out_dir) / (Path(extra).parent.name + "_" +
                                         Path(extra).stem + ".json"))
        eval_one(extra, out2)


if __name__ == "__main__":
    main()
