"""Read-only process-identity-bound host/process-tree memory sampling."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def process(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    status = {}
    for line in (root / 'status').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('VmRSS', 'VmHWM', 'VmSwap', 'Threads'):
            status[key] = int(value.split()[0])
    return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': fields[19], 'cpu_ticks': int(fields[11]) + int(fields[12]), 'memory_kib_except_threads': status}


def snapshot(pid):
    records = {}
    for directory in Path('/proc').iterdir():
        if directory.name.isdigit():
            try:
                record = process(int(directory.name))
                records[record['pid']] = record
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    descendants = {pid}
    while True:
        expanded = descendants | {key for key, row in records.items() if row['ppid'] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'SwapTotal', 'SwapFree'):
            memory[key] = int(value.split()[0]) * 1024
    tree = [records[key] for key in sorted(descendants) if key in records]
    return {
        'utc': datetime.now(timezone.utc).isoformat(), 'monotonic': time.monotonic(),
        'host_memory_bytes': memory, 'process_tree': tree,
        'rss_sum_bytes': sum(row['memory_kib_except_threads'].get('VmRSS', 0) * 1024 for row in tree),
        'rss_note': 'Summed resident memory can double-count shared pages; host MemAvailable is recorded separately.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    with target.open('x') as log:
        log.write(json.dumps({'sampling_scope': 'From sampler start only; no retrospective host-peak claim', 'pid': args.pid, 'expected_start_ticks': args.start_ticks}) + '\n')
        while True:
            try:
                if process(args.pid)['start_ticks'] != args.start_ticks:
                    break
            except (FileNotFoundError, ProcessLookupError):
                break
            try:
                log.write(json.dumps(snapshot(args.pid)) + '\n')
            except (OSError, ValueError, KeyError) as error:
                log.write(json.dumps({'monitor_warning': type(error).__name__ + ': ' + str(error)}) + '\n')
            log.flush()
            time.sleep(5)
        log.write(json.dumps({'sampling_ended_at': datetime.now(timezone.utc).isoformat(), 'reason': 'original process identity no longer present'}) + '\n')
    print(json.dumps({'output': str(target), 'status': 'SAMPLER_FINISHED'}))


if __name__ == '__main__':
    main()
