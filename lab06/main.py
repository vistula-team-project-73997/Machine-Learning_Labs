"""PyCharm entry point. Every experiment runs sequentially in a fresh process."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB / 'src'))


def plan(config):
    s = config['studies']
    # Explicit order; no parallel CPU studies or hidden subsets.
    jobs = [dict(name='profile', study='profile', epochs=1, profile=True)]
    jobs += [dict(name=f'workers_{w}', study='workers', num_workers=w,
                  epochs=s['worker_epochs']) for w in s['workers']]
    jobs += [dict(name=f'batch_{b}', study='batch', batch_size=b,
                  epochs=s['batch_epochs'], test=True) for b in s['batch_sizes']]
    jobs += [dict(name=f'{o}_{schedule}', study='optimizer', optimizer=o, scheduler=schedule,
                  epochs=s['optimizer_epochs']) for o in s['optimizers'] for schedule in s['schedules']]
    jobs += [dict(name='effective256', study='accumulation', batch_size=256, optimizer='SGD',
                  epochs=s['accumulation_epochs']),
             dict(name='micro64_accum4', study='accumulation', batch_size=64, accumulation_steps=4,
                  optimizer='SGD', epochs=s['accumulation_epochs']),
             dict(name='resume_reference', study='resume', epochs=s['resume_epochs'], scheduler='StepLR'),
             dict(name='resume_interrupted', study='resume', epochs=s['resume_epochs'], scheduler='StepLR',
                  interrupt_after=s['interrupt_after']),
             dict(name='resume_interrupted', study='resume', epochs=s['resume_epochs'], scheduler='StepLR',
                  resume=True)]
    return jobs


def main():
    from pipeline import load_config, environment, public_config, RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(LAB / 'config.yaml'))
    parser.add_argument('--continue-studies', action='store_true', help='Skip completed experiment summaries')
    parser.add_argument('--only', default='', help='One experiment name; empty means full assignment')
    args = parser.parse_args()
    config = load_config(args.config)
    RESULTS.mkdir(exist_ok=True)
    environment(config)
    test = subprocess.run([sys.executable, str(LAB / 'src/test_pipeline.py')],
                          capture_output=True, text=True)
    (RESULTS / 'unit_verification.txt').write_text(test.stdout + test.stderr, encoding='utf-8')
    print(test.stdout + test.stderr, flush=True)
    test.check_returncode()
    jobs = plan(config)
    (RESULTS / 'experiment_plan.json').write_text(json.dumps(jobs, indent=2)+'\n')
    specs = RESULTS / 'specs'
    specs.mkdir(exist_ok=True)
    for index, spec in enumerate(jobs):
        if args.only and spec['name'] != args.only:
            continue
        summary = RESULTS / 'runs' / spec['name'] / 'summary.json'
        if args.continue_studies and summary.exists():
            previous = json.loads((summary.parent / 'effective_config.json').read_text(encoding='utf-8'))
            assert public_config(previous['config']) == public_config(config), 'Config changed; rerun studies instead of mixing configurations.'
            print('SKIP completed:', spec['name'], flush=True)
            continue
        if args.continue_studies and spec.get('interrupt_after') and (summary.parent / 'checkpoint_epoch3.pt').exists():
            continue
        path = specs / f'{index:02d}_{spec["name"]}.json'
        path.write_text(json.dumps(spec, indent=2)+'\n')
        cmd = [sys.executable, '-u', str(LAB / 'src/pipeline.py'), '--config', args.config, '--spec', str(path)]
        print(f'\nEXPERIMENT {index+1}/{len(jobs)}: {spec["name"]}', flush=True)
        with (RESULTS / 'run_log.txt').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f'Experiment failed: {spec["name"]}; see results/run_log.txt')
    if not args.only:
        from summarize import summarize
        summarize(config)
        check = subprocess.run([sys.executable, str(LAB / 'src/verify_results.py')], capture_output=True, text=True)
        (RESULTS / 'verification.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
        print(check.stdout + check.stderr)
        check.check_returncode()


if __name__ == '__main__':
    main()
