"""Frozen refactor controls and blocked PT-by-sector measurements."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from pt_sector_contract import MODES, independent, numerics, path_issues, same_pencil, spectrum_error

ROOT = Path(__file__).resolve().parents[2]
SEED = 2026092403
CASES = [(name, 31, 8, pol) for name in ('gamma', 'mirror', 'rotation') for pol in ('TM', 'TE')]
CASES += [('gamma', 65, 24, pol) for pol in ('TM', 'TE')]
REGRESSION_CASES = [(name, 31, 8, pol) for name in ('gamma', 'mirror') for pol in ('TM', 'TE')]
REGRESSION_MODES = ('off', 'pt', 'sectors')
TIMED_REGRESSIONS = [(case, mode) for mode in REGRESSION_MODES
                     for case in [('gamma',31,8,'TM'),('mirror',31,8,'TE')]]
CONTRACT_FILES = ['pt-sector-protocol.md','pt_sector_contract.py','run_pt_sectors.py',
                  'run_pairs.py','support_contract.py','interface_contract.py']


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(logs, seed):
    rng=random.Random(seed)
    samples=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
    lo,hi=samples[250],samples[9749]
    return dict(geometric_ratio=math.exp(statistics.mean(logs)),ci95=[lo,hi],
        practical_decision=('gain' if lo>1.02 else 'regression' if hi<1/1.02
                            else 'equivalent_within_2_percent' if lo>=1/1.02 and hi<=1.02 else 'inconclusive'),
        statistical_difference=('faster' if lo>1 else 'slower' if hi<1 else 'unresolved'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    builds=json.loads(args.builds.read_text())
    assert builds.get('complete') and len(builds['artifacts'])==4
    assert len({a['target_directory'] for a in builds['artifacts'].values()})==4
    for a in builds['artifacts'].values(): assert sha(a['path'])==a['sha256']
    for kind in ('profiled','unprofiled'):
        assert builds['artifacts']['baseline_'+kind]['sha256']!=builds['artifacts']['candidate_'+kind]['sha256']
    assert sha(ROOT/'crates/core/examples/symmetry_pt_sector_bench.rs')==builds['harness_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==builds['candidate_revision']
    subprocess.run(['git','diff','HEAD','--exit-code'],cwd=ROOT,check=True,capture_output=True)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'objects').mkdir();(out/'failures').mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    contract_hashes={f:sha(ROOT/'benchmarks/symmetry'/f) for f in CONTRACT_FILES}
    manifest=dict(schema='blaze2d/pt-sector-study/1',builds=builds,cases=CASES,modes=MODES,
        regression_cases=REGRESSION_CASES,regression_modes=REGRESSION_MODES,
        timed_regressions=TIMED_REGRESSIONS,blocks=20,seed=SEED,cpu=0,tolerance=1e-14,
        started=time.time(),contracts=contract_hashes,python=sys.version,
        verification_packages={x:importlib.metadata.version(x) for x in ('numpy','scipy')},
        thread_environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},
        build_manifest_sha256=sha(args.builds),
        accepted_means='Complete, numerically valid study; practical performance decisions are separate.')
    (out/'manifest.json').write_bytes(canonical(manifest)+b'\n')
    rows=[]; objects=set(); signatures={}; references={}; issues=[]; failed_cases=[]

    def load(digest):
        payload=gzip.decompress((out/'objects'/f'{digest}.json.gz').read_bytes())
        assert hashlib.sha256(payload).hexdigest()==digest
        return json.loads(payload)

    def save_state(phase):
        state=dict(phase=phase,records=len(rows),objects=len(objects),issues=issues,
                   failed_cases=failed_cases,updated=time.time())
        temp=out/'state.tmp';temp.write_bytes(canonical(state)+b'\n');temp.replace(out/'state.json')

    def execute(case,mode,label,phase,index):
        name,n,bands,pol=case
        profiled=phase in ('preflight_profiled','profile_regression','profile_interaction')
        artifact=label+('_profiled' if profiled else '_unprofiled')
        command=['taskset','-c','0',builds['artifacts'][artifact]['path'],name,str(n),str(bands),pol,mode,'1e-14']
        row=dict(case=list(case),mode=mode,label=label,phase=phase,index=index,command=command,
                 timestamp=time.time(),load_before=os.getloadavg(),issues=[])
        began=time.perf_counter(); process=None
        try:
            process=subprocess.run(command,env=env,cwd=ROOT,capture_output=True,timeout=600)
            row.update(exit_code=process.returncode,stderr=process.stderr.decode(errors='replace'))
            if process.returncode: raise RuntimeError('benchmark process failed')
            value=json.loads(process.stdout)
            nvalue=numerics(value);payload=canonical(nvalue);digest=hashlib.sha256(payload).hexdigest()
            if digest not in objects:
                (out/'objects'/f'{digest}.json.gz').write_bytes(gzip.compress(payload,mtime=0));objects.add(digest)
            row.update(numerics_sha256=digest,seconds=value['seconds'],peak_rss_kib=value['peak_rss_kib'],
                       counters=value['counters'],stage_counters=value['stage_counters'],profile=value['profile'])
            row['issues']+=path_issues(nvalue)
            if (value['case']!=name or value['grid']!=[n,n] or value['bands']!=bands
                    or value['polarization']!=pol or value['mode']!=mode or value['tolerance']!=1e-14):
                row['issues'].append('executed input differs from frozen case')
            if not math.isfinite(value['seconds']) or value['seconds']<=0 or not isinstance(value['peak_rss_kib'],int) or value['peak_rss_kib']<=0:
                row['issues'].append('invalid time/RSS measurement')
            key=(case,mode)
            previous=signatures.setdefault(key,digest)
            if previous!=digest: row['issues'].append('numerical signature changed across builds/repeats')
            if mode=='off' and case not in references: references[case]=digest
            if case in references:
                reference=load(references[case])
                if not same_pencil(nvalue,reference): row['issues'].append('exact physical pencil changed')
                if spectrum_error(nvalue,reference)>1e-6: row['issues'].append('requested spectrum differs from unrestricted control')
        except Exception as error:
            row['issues'].append(str(error))
            if process is not None:
                raw=process.stdout+b'\nSTDERR\n'+process.stderr
                failure=out/'failures'/f'{len(rows):05d}.txt.gz';failure.write_bytes(gzip.compress(raw,mtime=0))
                row['failure_payload']=str(failure.relative_to(out))
            elif isinstance(error,subprocess.TimeoutExpired):
                raw=(error.stdout or b'')+b'\nSTDERR\n'+(error.stderr or b'')
                failure=out/'failures'/f'{len(rows):05d}.txt.gz';failure.write_bytes(gzip.compress(raw,mtime=0))
                row['failure_payload']=str(failure.relative_to(out))
        row.update(process_seconds=time.perf_counter()-began,load_after=os.getloadavg(),passed=not row['issues'])
        rows.append(row)
        with (out/'records.jsonl').open('ab') as f:f.write(canonical(row)+b'\n')
        return row

    # Exact refactor controls also qualify all four executable artifacts.
    for case in REGRESSION_CASES:
        for mode in REGRESSION_MODES:
            tested=[execute(case,mode,label,phase,-1) for label in ('baseline','candidate')
                    for phase in ('preflight_unprofiled','preflight_profiled')]
            if any(not r['passed'] for r in tested):
                issues.append(dict(scope='refactor_preflight',case=case,mode=mode,
                                   failures=[r['issues'] for r in tested if not r['passed']]))
            a,b=tested[1],tested[3]
            if a.get('counters')!=b.get('counters') or a.get('stage_counters')!=b.get('stage_counters'):
                issues.append(dict(scope='refactor_preflight_counters',case=case,mode=mode))
    save_state('refactor_preflight')
    if issues:
        print(json.dumps(dict(phase='refactor_preflight',issues=issues)),flush=True)
        return 1
    print(json.dumps(dict(phase='refactor_preflight',records=len(rows),issues=[])),flush=True)

    eligible=[]
    for case in CASES:
        tested=[execute(case,mode,'candidate',phase,-1) for mode in MODES
                for phase in ('preflight_unprofiled','preflight_profiled')]
        if all(r['passed'] for r in tested): eligible.append(case)
        else: failed_cases.append(dict(case=case,phase='preflight',failures=[dict(mode=r['mode'],issues=r['issues']) for r in tested if not r['passed']]))
        save_state('interaction_preflight')
        print(json.dumps(dict(phase='interaction_preflight',case=case,passed=all(r['passed'] for r in tested))),flush=True)

    rng=random.Random(SEED)
    for case,mode in TIMED_REGRESSIONS:
        for index in range(-1,20):
            order=['baseline','candidate'];rng.shuffle(order)
            block=[execute(case,mode,label,'warmup_regression' if index<0 else 'primary_regression',index) for label in order]
            if not all(r['passed'] for r in block):
                issues.append(dict(scope='refactor_sampling',case=case,mode=mode,index=index));break
        save_state('regression_sampling')
        print(json.dumps(dict(phase='regression_sampling',case=case,mode=mode,records=len(rows))),flush=True)
    for case in eligible:
        for index in range(-1,20):
            order=list(MODES);rng.shuffle(order)
            block=[execute(case,mode,'candidate','warmup' if index<0 else 'primary',index) for mode in order]
            if not all(r['passed'] for r in block):
                failed_cases.append(dict(case=case,phase='sampling',index=index,
                    failures=[dict(mode=r['mode'],issues=r['issues']) for r in block if not r['passed']]));break
        save_state('interaction_sampling')
        print(json.dumps(dict(phase='interaction_sampling',case=case,records=len(rows))),flush=True)
    for case,mode in TIMED_REGRESSIONS:
        for label in ('baseline','candidate'):execute(case,mode,label,'profile_regression',0)
    for case in eligible:
        for mode in MODES:execute(case,mode,'candidate','profile_interaction',0)
    save_state('sampling_complete')

    # Numerical audits and statistics start only after the last timed process.
    for f,digest in contract_hashes.items():
        if sha(ROOT/'benchmarks/symmetry'/f)!=digest:issues.append(dict(scope='changed_contract',file=f))
    audits={}
    for digest in sorted(objects):
        value=load(digest)
        try:
            case=(value['case'],value['grid'][0],value['bands'],value['polarization'])
            audits[digest]=independent(value,load(references[case]))
        except Exception as e:audits[digest]=dict(issues=['independent verification failed: '+str(e)])
    audit_failed={digest for digest,a in audits.items() if a['issues']}

    def selected(case,mode=None,label=None,phase=None):
        return [r for r in rows if r['case']==list(case) and (mode is None or r['mode']==mode)
                and (label is None or r['label']==label) and (phase is None or r['phase']==phase)]

    def valid(rs):
        return all(r['passed'] and r.get('numerics_sha256') not in audit_failed for r in rs)

    regression=[]
    for i,(case,mode) in enumerate(TIMED_REGRESSIONS):
        rs=selected(case,mode,phase='primary_regression')
        result=dict(case=case,mode=mode,primary_records=len(rs),accepted=False)
        profiles={label:selected(case,mode,label,'profile_regression')[-1] for label in ('baseline','candidate')}
        if len(rs)==40 and valid(selected(case,mode)) and not issues:
            by={label:sorted([r for r in rs if r['label']==label],key=lambda r:r['index']) for label in ('baseline','candidate')}
            if profiles['baseline']['counters']!=profiles['candidate']['counters']:
                result['issues']=['profile work counters changed']
            else:
                result.update(accepted=True,timing=interval([math.log(a['seconds']/b['seconds']) for a,b in zip(by['baseline'],by['candidate'])],SEED+i),
                    median_seconds={k:statistics.median(r['seconds'] for r in v) for k,v in by.items()},
                    median_rss_kib={k:statistics.median(r['peak_rss_kib'] for r in v) for k,v in by.items()},
                    profile_ratio={k:profiles[k]['seconds']/statistics.median(r['seconds'] for r in v) for k,v in by.items()})
        regression.append(result)

    interactions=[]
    contrasts=[('pt','off','pt'),('sectors','off','sectors'),('combined','off','pt-sectors'),
               ('sectors_after_pt','pt','pt-sectors'),('pt_after_sectors','sectors','pt-sectors'),
               ('fourier_projection','sectors','fourier-sectors'),('pt_after_fourier_sectors','fourier-sectors','pt-sectors')]
    for i,case in enumerate(CASES):
        rs=selected(case,phase='primary');result=dict(case=case,primary_records=len(rs),accepted=False)
        if len(rs)==100 and valid(selected(case,label='candidate')) and not issues:
            by={mode:sorted([r for r in rs if r['mode']==mode],key=lambda r:r['index']) for mode in MODES}
            result.update(accepted=True,ratios={name:interval([math.log(a['seconds']/b['seconds']) for a,b in zip(by[left],by[right])],SEED+100*i+j)
                          for j,(name,left,right) in enumerate(contrasts)})
            interaction=interval([math.log(by['pt-sectors'][k]['seconds']*by['off'][k]['seconds']/(by['pt'][k]['seconds']*by['sectors'][k]['seconds'])) for k in range(20)],SEED+100*i+10)
            result['interaction_time_factor']={k:v for k,v in interaction.items() if k in ('geometric_ratio','ci95')}
            result['modes']={}
            for mode in MODES:
                data=by[mode];nvalue=load(data[0]['numerics_sha256']);profile=selected(case,mode,'candidate','profile_interaction')[-1]
                result['modes'][mode]=dict(median_seconds=statistics.median(r['seconds'] for r in data),
                    median_rss_kib=statistics.median(r['peak_rss_kib'] for r in data),
                    iterations=sum(p['iterations'] for p in nvalue['points']),
                    profile_time_ratio=profile['seconds']/statistics.median(r['seconds'] for r in data),
                    counters=profile['counters'],stage_counters=profile['stage_counters'],
                    numerics_sha256=data[0]['numerics_sha256'])
        interactions.append(result)
    audit=dict(schema='blaze2d/pt-sector-audit/1',records=len(rows),objects=len(objects),
               regression=regression,interactions=interactions,independent=audits,
               issues=issues,failed_cases=failed_cases,failed_records=sum(not r['passed'] for r in rows),
               records_sha256=sha(out/'records.jsonl'),finished=time.time())
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');save_state('complete')
    print(json.dumps(dict(phase='complete',records=len(rows),accepted_regressions=sum(r['accepted'] for r in regression),
        accepted_interactions=sum(r['accepted'] for r in interactions),failed_records=audit['failed_records'],
        independent_failures=len(audit_failed),issues=issues,failed_cases=failed_cases)),flush=True)
    return int(bool(issues or failed_cases or audit_failed or audit['failed_records']))


if __name__=='__main__':raise SystemExit(main())
