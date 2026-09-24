#!/usr/bin/env python3
"""Audit D05 provenance, complete fields, pairs, references and work accounting."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import numpy as np
import scipy
from support_contract import quality, pair_quality, compare, certificate_issues, independent_certificate, decode

ROOT = Path(__file__).resolve().parents[2]


def audit(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    records=[json.loads(line) for line in (directory/"records.jsonl").read_text().splitlines()]
    problems=[]
    objects={}
    def load(sha):
        if sha not in objects:
            data=gzip.decompress((directory/"objects"/(sha+".json.gz")).read_bytes())
            if hashlib.sha256(data).hexdigest()!=sha:
                raise ValueError("payload digest mismatch: " + sha)
            objects[sha]=json.loads(data)
        return objects[sha]
    revision=manifest["source_revision"]
    for path,expected in manifest["files"].items():
        source=subprocess.check_output(["git","show",revision+":"+path],cwd=ROOT)
        if hashlib.sha256(source).hexdigest()!=expected:
            problems.append("historical source digest mismatch: " + path)
    if manifest["builds"]["source_revision"]!=revision:
        problems.append("build source mismatch")
    for label in ("unprofiled","profiled"):
        built=manifest["builds"]["artifacts"][label]
        if manifest["final_binary_digests"][label]!=built["sha256"]:
            problems.append("executable changed during sampling: " + label)
        path=Path(built["path"])
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=built["sha256"]:
            problems.append("retained executable changed: " + label)
    expected=len(manifest["design"])*(2*(manifest["pairs"]+1)+2)+8
    if len(records)!=expected or manifest["record_count"]!=len(records):
        problems.append("incomplete record ledger")
    # Replay the declared randomization rather than trusting pair labels alone.
    rng=random.Random(manifest["seed"])
    expected_order=[]
    for spec in manifest["design"]:
        for pair in range(-1,manifest["pairs"]):
            labels=["baseline","candidate"];rng.shuffle(labels)
            for label in labels:
                expected_order.append((spec["name"],pair,label,"warmup" if pair<0 else "primary"))
    observed_order=[(r["study"],r["pair"],r["label"],r["phase"]) for r in records if r["phase"] in ("warmup","primary")]
    if observed_order!=expected_order:
        problems.append("pair order differs from frozen randomized design")
    primary_positions=[i for i,r in enumerate(records) if r["phase"] in ("warmup","primary")]
    later_positions=[i for i,r in enumerate(records) if r["phase"] not in ("warmup","primary")]
    if primary_positions and later_positions and max(primary_positions)>min(later_positions):
        problems.append("profiling/negative controls overlap primary collection")
    for row in records:
        for key in ("numerics_sha256","references_sha256","failed_stdout_sha256","timeout_output_sha256"):
            if key in row:
                load(row[key])
    checks={sha:quality(value) for sha,value in list(objects.items()) if "operator_identity" in value and value["error"] is None}
    comparisons=[]
    profile_diagnostics=[]
    for spec in manifest["design"]:
        rows=[r for r in records if r["study"]==spec["name"] and not r["negative_control"]]
        expected_config={"case":spec["case"],"grid":[spec["grid"]]*2,"bands":spec["bands"],"radius":spec["radius"],"polarization":spec["polarization"]}
        failures=[]
        signatures={}
        for row in rows:
            label=row["label"]
            if row["returncode"]!=0 or "numerics_sha256" not in row:
                failures.append(dict(pair=row["pair"],phase=row["phase"],label=label,issues=["process/parse failure"]))
                continue
            sha=row["numerics_sha256"];n=load(sha);m=row["measurement"]
            issues=checks.get(sha,{"issues":["solve rejected: "+str(n["error"])]})["issues"].copy()
            if any(n.get(key)!=value for key,value in expected_config.items()) or n["mode"]!=spec[label]:
                problems.append(spec["name"]+" executed another configuration")
            if n["eigensolver"]["tol"]!=1e-14 or n["eigensolver"]["max_iter"]!=1000:
                problems.append(spec["name"]+" changed tolerance/iteration contract")
            if not math.isfinite(m["seconds"]) or m["seconds"]<=0 or not isinstance(m["peak_rss_kib"],int) or m["peak_rss_kib"]<=0:
                problems.append(spec["name"]+" invalid time/RSS")
            if row["phase"]=="primary":
                signatures.setdefault(label,set()).add(sha)
            if row["phase"]!="profile" and (m["counters"] or m["profile"] or m["stage_counters"]):
                problems.append(spec["name"]+" instrumented primary/warmup")
            if row["phase"]=="profile":
                result=n.get("result")
                iterations=result["report"]["iterations"] if result else 0
                if m["counters"].get("iterations",0)!=iterations or m["counters"].get("profiler.unbalanced_timers",0):
                    problems.append(spec["name"]+" profile iteration/timer mismatch")
                if not m["profile"]:
                    problems.append(spec["name"]+" empty explanatory profile")
                diagnostic=dict(study=spec["name"],label=label,numerics_sha256=sha)
                if "references_sha256" in row:
                    ref=load(row["references_sha256"])
                    projected=ref["projected"]
                    if "error" in projected:
                        diagnostic["same_pencil_issues"]=[projected["error"]]
                    else:
                        nref=dict(n,result=projected,bands=len(projected["spectrum"]),sectors=None,mode="unreduced",error=None)
                        diagnostic["same_pencil_quality"]=quality(nref)
                        diagnostic["same_pencil_issues"]=diagnostic["same_pencil_quality"]["issues"].copy()
                        if result:
                            cmp=compare(n,result,projected)
                            diagnostic["same_pencil_comparison"]=cmp
                            if cmp["max_scaled_frequency_error"]>1e-6 or cmp["guarded_prefix"]<=0 or cmp["max_sin_angle"]>1e-3:
                                diagnostic["same_pencil_issues"].append("guarded same-pencil reference disagrees")
                    issues += diagnostic["same_pencil_issues"]
                    full=ref["full_rectangle"]
                    k=decode(n["support_identity"]["k_bits"])
                    kernel=any(np.array_equal(k+np.asarray(g),[0.0,0.0]) for g in n["support_identity"]["modes"])
                    full_issues=certificate_issues(full["certificate"],full["bands"],kernel,"full rectangle reference")
                    if not full["solver_converged"]:
                        full_issues.append("full rectangle reference did not converge")
                    cert=independent_certificate(n,full,projected=False)
                    full_issues += certificate_issues(cert,full["bands"],kernel,"independent full rectangle")
                    diagnostic["full_rectangle_issues"]=full_issues
                    diagnostic["full_rectangle_independent_certificate"]=cert
                    if result:
                        diagnostic["full_rectangle_comparison"]=compare(n,result,full)
                else:
                    issues.append("missing supplementary references")
                profile_diagnostics.append(diagnostic)
            if issues:
                failures.append(dict(pair=row["pair"],phase=row["phase"],label=label,issues=issues))
        for label in ("baseline","candidate"):
            sig=signatures.get(label,set())
            if len(sig)!=1:
                failures.append(dict(label=label,issues=["primary complete numerical snapshots do not repeat exactly"]))
            prof=[r for r in rows if r["phase"]=="profile" and r["label"]==label]
            if len(prof)!=1 or prof[0].get("numerics_sha256") not in sig:
                failures.append(dict(label=label,issues=["profile snapshot differs from primary"]))
        ratios=[]; pair_checks=[]
        for pair in range(manifest["pairs"]):
            paired={r["label"]:r for r in rows if r["phase"]=="primary" and r["pair"]==pair}
            if set(paired)!={"baseline","candidate"} or any("numerics_sha256" not in r for r in paired.values()):
                failures.append(dict(pair=pair,issues=["incomplete pair"]))
                continue
            a,b=(paired[label] for label in ("baseline","candidate"))
            check=pair_quality(load(a["numerics_sha256"]),load(b["numerics_sha256"]))
            if check["issues"]:
                failures.append(dict(pair=pair,issues=check["issues"]))
            pair_checks.append(check)
            ratios.append(a["measurement"]["seconds"]/b["measurement"]["seconds"])
        stats=None
        if len(ratios)==manifest["pairs"]:
            log=np.log(ratios)
            boot=np.random.default_rng(20260924).choice(log,(10000,len(log)),replace=True).mean(axis=1)
            lo,hi=np.exp(np.quantile(boot,[.025,.975]))
            stats=dict(geometric_speedup=float(np.exp(log.mean())),ci95=[float(lo),float(hi)])
        valid=not failures
        adequate_sample=manifest["pairs"]>=20
        classification="invalid physical/work gate" if not valid else ("smoke only; insufficient pairs" if not adequate_sample else
            ("gain" if stats["ci95"][0]>1.02 else "regression" if stats["ci95"][1]<1/1.02 else "no demonstrated practical change"))
        measurements={}
        for label in ("baseline","candidate"):
            good=[r for r in rows if r["phase"]=="primary" and r["label"]==label and "measurement" in r]
            if good:
                measurements[label]={key:statistics.median(r["measurement"][key] for r in good) for key in ("seconds","peak_rss_kib","serialization_seconds")}
                n=load(good[0]["numerics_sha256"])
                if n["result"]:
                    measurements[label]["iterations"]=n["result"]["report"]["iterations"]
                    measurements[label]["estimated_payload_bytes"]=n["result"]["report"]["estimated_payload_bytes"]
        comparisons.append(dict(study=spec["name"],valid=valid,adequate_sample=adequate_sample,classification=classification,failures=failures,
                                statistics=stats,measurements=measurements,pair_checks=pair_checks))
    negatives=[]
    for r in records:
        if not r["negative_control"]:
            continue
        n=load(r["numerics_sha256"]) if "numerics_sha256" in r else {}
        reason=n.get("error","") or ""
        expected="sampled scalar material" if "analytic" in r["study"] else "sample nodes"
        passed=r["returncode"]==0 and n.get("result") is None and expected in reason
        negatives.append(dict(study=r["study"],phase=r["phase"],passed=passed,rejection=reason))
        if not passed:
            problems.append("negative control was not specifically rejected: "+r["study"])
    # A data-integrity failure withholds every speed claim, even if local gates pass.
    if problems:
        for row in comparisons:
            row["valid"]=False;row["classification"]="invalid study integrity"
    return dict(schema="blaze2d/support-audit/1",source_revision=revision,issues=problems,
                audit_sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ("audit_support_study.py","support_contract.py")},
                audit_runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
                record_count=len(records),phase_counts=dict(Counter(r["phase"] for r in records)),
                unique_objects=len(objects),numerical_checks=checks,comparisons=comparisons,
                profiles=profile_diagnostics,negative_controls=negatives,
                scope="paired fixed-pencil speed; independent stored-field FFT certificates; hardware bandwidth unavailable")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=audit(args.directory)
    target=args.output or args.directory/"audit.json"
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(issues=result["issues"],records=result["record_count"],
                         classifications=dict(Counter(x["classification"] for x in result["comparisons"])))))
    for row in result["comparisons"]:
        print(row["study"],row["classification"],row["statistics"])
    if result["issues"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
