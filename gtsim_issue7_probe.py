"""Host-only diagnostic for GTSim compute-profile precedence; no board data."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gtsim', type=Path, required=True, help='GTSim source checkout')
    parser.add_argument('--sim', type=Path, required=True, help='built gtsim_ir executable')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    root, sim, out = args.gtsim.resolve(), args.sim.resolve(), args.out.resolve()
    baseline = (root / 'configs/ppu15/device.cfg').read_text(encoding='utf-8')
    if not sim.is_file():
        parser.error('simulator executable does not exist')
    sys.path.insert(0, str(root / 'scripts'))
    from gtsim_tilelang import Node, serialize

    # Keep all mutations in a new result directory; never rewrite the source profile.
    out.mkdir(parents=True, exist_ok=False)
    cfg = out / 'profiles/ppu15/device.cfg'
    cfg.parent.mkdir(parents=True)
    results = {}

    def run(name, nodes, config):
        cfg.write_text(config, encoding='utf-8')
        (out / (name + '.cfg')).write_text(config, encoding='utf-8')
        dag = serialize((1, 1, 0), nodes)
        (out / (name + '.dag')).write_text(dag, encoding='utf-8')
        timeline = out / (name + '.tsv')
        p = subprocess.run([str(sim), '--timeline', str(timeline)],
                           input=dag, text=True, capture_output=True,
                           env={**os.environ, 'GTSIM_CONFIG_DIR': str(cfg.parent.parent)}, timeout=30)
        (out / (name + '.log')).write_text(p.stdout + p.stderr, encoding='utf-8')
        with timeline.open(encoding='utf-8') if p.returncode == 0 else open(os.devnull) as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        results[name] = dict(returncode=p.returncode, rows=rows, error=p.stderr)

    def mma(op):
        return Node(op, 0, 0, dtype='float32', tile=(0, 0, 16, 16, 0),
                    matrix_input='bfloat16', matrix_shape=(16, 16, 16),
                    logical_matrix=op == 'matrix_tile')

    # Replace the complete key, retaining the current provenance string.
    changed = []
    found = False
    for line in baseline.splitlines():
        if line.startswith('cube.bfloat16.float32='):
            fields = line.split('=', 1)[1].split()
            if len(fields) != 7:
                raise ValueError('unsupported cube profile format')
            fields[:6] = ['16', '16', '16', '2', '32', '1']
            line = 'cube.bfloat16.float32=' + ' '.join(fields)
            found = True
        changed.append(line)
    if not found:
        raise ValueError('this probe requires the ISSUE 7 cube profile')
    for mode, config in [('baseline', baseline), ('cube_updated', '\n'.join(changed) + '\n')]:
        for op in ('mma', 'mma.bfloat16.float32', 'matrix_tile'):
            run(mode + '_' + op, [mma(op)], config)

    # Remove exact probe keys first so an existing calibration cannot create duplicate keys.
    cast_base = '\n'.join(line for line in baseline.splitlines() if not line.startswith(
        ('conversion.float32.float16.1=', 'cost.cast.float32.float16.1='))) + '\n'
    legacy = 'conversion.float32.float16.1=SIMD 8 10\n'
    exact = 'cost.cast.float32.float16.1=SIMD 64 1 1 2 element_op 0 explicit_new_cost\n'
    cast = Node('cast', 0, 0, dtype='float32', result_dtype='float16',
                conversion_pack=1, tile=(0, 0, 1, 64, 0))
    run('cast_new_cost', [cast], cast_base + exact)
    run('cast_new_and_legacy', [cast], cast_base + legacy + exact)
    (out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    for name, case in results.items():
        print(name, 'returncode=', case['returncode'],
              [{k: row[k] for k in ('op', 'issue', 'end', 'fu_service_cycles', 'cost_source')}
               for row in case['rows']])
    print('Diagnostic only: interpret results using docs/GTSIM_ISSUE7_REVIEW.md')


if __name__ == '__main__':
    main()
