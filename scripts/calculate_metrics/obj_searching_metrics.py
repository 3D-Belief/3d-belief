#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

EPS = 1e-8

def safe_load(p: Path):
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def per_episode_metrics(ep):
    """Compute per-episode metrics (mirrors the aggregate definitions)."""
    s = 1.0 if ep.get("success", False) else 0.0

    # SPL term: s * d*/max(l, d*, EPS)
    d_star = ep.get("oracle_distance_traveled")
    l = ep.get("distance_traveled")
    spl = None
    if d_star is not None and l is not None:
        spl = s * (float(d_star) / max(float(l), float(d_star), EPS))

    # SWNT term: s * m/max(t, m, 1)
    m = ep.get("oracle_length")
    t = ep.get("num_steps")
    swnt = None
    if m is not None and t is not None:
        swnt = s * (float(m) / max(int(t), int(m), 1))

    # ToO term: t / max(m, 1)
    too = None
    if m is not None and t is not None and int(m) > 0:
        too = float(t) / max(int(m), 1)

    # Distance progress: (d0 - dT) / max(d0, EPS)
    d0 = ep.get("initial_distance_to_goal")
    dT = ep.get("final_distance_to_goal")
    dp = None
    if d0 is not None and dT is not None and float(d0) > 0:
        dp = (float(d0) - float(dT)) / max(float(d0), EPS)

    # Per-decision inference time
    infer_t_0 = float(ep.get("model_inference_time", 0.0))
    infer_t_1 = float(ep.get("model_inference_time_obs", 0.0))
    infer_t = infer_t_0 + infer_t_1
    plan_t  = ep.get("planning_time")
    obs_steps = ep.get("num_steps")
    if obs_steps is None:
        # fallback for vlm agents (make decision every 5 steps)
        num_steps = ep.get("num_steps")
        obs_steps = (int(num_steps) // 5) if num_steps is not None else None

    avg_infer_per_decision = None
    if (infer_t is not None or plan_t is not None) and obs_steps is not None and int(obs_steps) > 0:
        total = float(infer_t or 0.0) + float(plan_t or 0.0)
        avg_infer_per_decision = total / int(obs_steps)

    # Per-decision VLM tokens
    vin  = ep.get("vlm_input_tokens")
    vout = ep.get("vlm_output_tokens")
    avg_vlm_in_per_decision  = None
    avg_vlm_out_per_decision = None
    if (vin is not None or vout is not None) and obs_steps is not None and int(obs_steps) > 0:
        avg_vlm_in_per_decision  = int(vin or 0)  / int(obs_steps)
        avg_vlm_out_per_decision = int(vout or 0) / int(obs_steps)

    return {
        "SR": s,
        "SPL": spl,
        "SWNT": swnt,
        "ToO": too,
        "DP": dp,
        "avg_infer_decision": avg_infer_per_decision,
        "avg_vlm_in_decision": avg_vlm_in_per_decision,
        "avg_vlm_out_decision": avg_vlm_out_per_decision,
    }

def fmt(x, places=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)

def main(root: Path, per_episode: bool):
    logs = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        jf = sub / "0_mug" / "final_log.json"
        if jf.is_file():
            data = safe_load(jf)
            if data is not None:
                logs.append((sub.name, data))

    N = len(logs)
    if N == 0:
        print("episodes counted: 0")
        return

    sr_sum = 0.0
    spl_terms, swnt_terms, too_terms, dp_terms = [], [], [], []
    skipped_spl = skipped_swnt = skipped_too = skipped_dp = 0

    # totals for per-decision averages across all episodes
    total_infer_time = 0.0
    total_obs_steps_for_infer = 0
    total_vlm_in = 0
    total_vlm_out = 0
    total_obs_steps_for_tokens = 0

    # track success/failure episode names
    success_eps = []
    failure_eps = []

    if per_episode:
        print("episode, SR, SPL, SWNT, ToO, DP, avg_infer_time_decision(s), avg_vlm_in_decision, avg_vlm_out_decision")

    for name, ep in logs:
        # per-episode metrics
        epm = per_episode_metrics(ep)

        # record success/failure by episode name
        if ep.get("success", False):
            success_eps.append(name)
        else:
            failure_eps.append(name)

        if per_episode:
            print(
                f"{name}, "
                f"{fmt(epm['SR'])}, "
                f"{fmt(epm['SPL'])}, "
                f"{fmt(epm['SWNT'])}, "
                f"{fmt(epm['ToO'])}, "
                f"{fmt(epm['DP'])}, "
                f"{fmt(epm['avg_infer_decision'], 6)}, "
                f"{fmt(epm['avg_vlm_in_decision'])}, "
                f"{fmt(epm['avg_vlm_out_decision'])}"
            )

        # accumulate for aggregates
        s = epm["SR"]
        sr_sum += s

        if epm["SPL"] is not None:
            spl_terms.append(epm["SPL"])
        else:
            skipped_spl += 1

        if epm["SWNT"] is not None:
            swnt_terms.append(epm["SWNT"])
        else:
            skipped_swnt += 1

        if epm["ToO"] is not None:
            too_terms.append(epm["ToO"])
        else:
            skipped_too += 1

        if epm["DP"] is not None:
            dp_terms.append(epm["DP"])
        else:
            skipped_dp += 1

        # Reuse raw fields to aggregate per-decision totals
        infer_t_0 = float(ep.get("model_inference_time", 0.0))
        infer_t_1 = float(ep.get("model_inference_time_obs", 0.0))
        infer_t = infer_t_0 + infer_t_1
        plan_t  = ep.get("planning_time")
        obs_steps = ep.get("obs_steps")
        if obs_steps is None:
            # fallback for vlm agents (make decision every 5 steps)
            num_steps = ep.get("num_steps")
            obs_steps = (int(num_steps) // 5) if num_steps is not None else None
        if (infer_t is not None or plan_t is not None) and obs_steps is not None and int(obs_steps) > 0:
            total_infer_time += float(infer_t or 0.0) + float(plan_t or 0.0)
            total_obs_steps_for_infer += int(obs_steps)

        vin  = ep.get("vlm_input_tokens")
        vout = ep.get("vlm_output_tokens")
        if (vin is not None or vout is not None) and obs_steps is not None and int(obs_steps) > 0:
            total_vlm_in  += int(vin or 0)
            total_vlm_out += int(vout or 0)
            total_obs_steps_for_tokens += int(obs_steps)

    SR   = sr_sum / N
    SPL  = sum(spl_terms) / N
    SWNT = sum(swnt_terms) / N
    ToO  = sum(too_terms) / N
    DP   = sum(dp_terms) / N

    avg_infer_per_decision = (
        total_infer_time / total_obs_steps_for_infer if total_obs_steps_for_infer > 0 else float("nan")
    )
    avg_vlm_in_per_decision = (
        total_vlm_in / total_obs_steps_for_tokens if total_obs_steps_for_tokens > 0 else float("nan")
    )
    avg_vlm_out_per_decision = (
        total_vlm_out / total_obs_steps_for_tokens if total_obs_steps_for_tokens > 0 else float("nan")
    )

    print(f"episodes counted: {N}")
    print(f"SR   (success rate)                    : {SR:.4f}")
    print(f"SPL  (success weighted by path length): {SPL:.4f}  (skipped {skipped_spl})")
    print(f"SWNT (success-weighted norm. time)    : {SWNT:.4f}  (skipped {skipped_swnt})")
    print(f"ToO  (time over oracle, steps ratio)  : {ToO:.4f}  (skipped {skipped_too})")
    print(f"DP   (distance progress)              : {DP:.4f}  (skipped {skipped_dp})")
    print(f"avg_inference_time_per_decision (s)   : {avg_infer_per_decision:.6f}")
    print(f"avg_vlm_input_tokens_per_decision     : {avg_vlm_in_per_decision:.4f}")
    print(f"avg_vlm_output_tokens_per_decision    : {avg_vlm_out_per_decision:.4f}")

    # print episode names
    print(f"successful episodes ({len(success_eps)}):")
    if success_eps:
        print("  " + ", ".join(success_eps))
    else:
        print("  (none)")

    print(f"failed episodes ({len(failure_eps)}):")
    if failure_eps:
        print("  " + ", ".join(failure_eps))
    else:
        print("  (none)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="root directory containing rollout subfolders")
    ap.add_argument("--per-episode", action="store_true",
                    help="print per-episode metrics for each rollout")
    args = ap.parse_args()
    main(args.root, per_episode=args.per_episode)
