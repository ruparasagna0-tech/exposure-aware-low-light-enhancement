# 10b. Phase-2 trainer timing reference, v2 (for a NEW Colab VM). Same approved procedure as
#      phase2_trainer_timing_cell.py (kept unchanged as the record of the 2026-10-14 run), with two fixes:
#      (1) everything runs inside _phase2_timing(), so no notebook global (e.g. summary()) is overwritten;
#      (2) records this VM's boot id and compares with the previous VM's reference (boot 7fc505c0...).
#      Requires cells 1-4 in this VM (DRIVE, md5(), PILOT_DIR). Throwaway weights, never evaluated, no test set.
def _phase2_timing():
    import os, re, json, time, shutil, signal, statistics, subprocess, platform

    TRAINER_MD5 = '41f8d805c9e5bfdc837cc9472de3f5c5'   # scripts/train_generalized.py used for Phase 2 (Sept-27 snapshot)
    CONFIG_MD5 = '7326edc089669cdd43fbfe92889769f8'    # configs/generalized_lol_v2.yaml (Phase 2)
    PHASE2_EPOCH_S_MEDIAN, PHASE2_TOTAL_S = 15.1, 757.8
    PREVIOUS_REFERENCE = f'{DRIVE}/experiments/phase3_model_selection/stage1/pilots/phase2_trainer_timing_20261014_003632/summary.json'
    PREVIOUS_BOOT_ID = '7fc505c0-c1c8-4088-abfa-20c19c8eebf0'
    SRC = f'{DRIVE}/scripts/train_generalized.py'
    DST = '/content/proj/scripts/train_generalized.py'
    stamp = time.strftime('%Y%m%d_%H%M%S')
    THROWAWAY = f'/content/phase2_trainer_timing_{stamp}'   # checkpoints stay on the VM disk; never evaluated, never copied
    REPORT = f'{DRIVE}/experiments/phase3_model_selection/stage1/pilots/phase2_trainer_timing_{stamp}'

    # --- safety and fidelity checks (any failure stops before training)
    env_boot = open('/proc/sys/kernel/random/boot_id').read().strip()
    left = subprocess.run(['pgrep', '-af', 'phase3_pilot.py|train_phase3.py|train_generalized.py'],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f'other pilot/training processes are running:\n{left}'
    assert os.path.exists(SRC), f'original trainer not found on Drive: {SRC} (STOP; do not improvise)'
    assert env_boot != PREVIOUS_BOOT_ID, 'same VM as the existing reference: no need to repeat'
    assert md5(SRC) == TRAINER_MD5, f'Drive trainer is not the Phase-2 trainer (md5 {md5(SRC)}): STOP'
    assert md5('/content/proj/configs/generalized_lol_v2.yaml') == CONFIG_MD5, 'config differs from Phase 2: STOP'
    assert not os.path.exists(DST) or md5(DST) == TRAINER_MD5
    assert not os.path.exists(THROWAWAY) and not os.path.exists(REPORT)
    shutil.copy2(SRC, DST)
    assert md5(DST) == TRAINER_MD5
    os.makedirs(REPORT)


    def q(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f'unavailable: {e!r}'


    import torch
    env = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%S'), 'python': platform.python_version(), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0),
           'nvidia_smi': q(['nvidia-smi', '--query-gpu=name,driver_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,'
                            'power.draw,power.limit,temperature.gpu', '--format=csv']),
           'cpu_model': q(['bash', '-c', "lscpu | grep -E 'Model name|^CPU\\(s\\)'"]), 'cpu_count': os.cpu_count(),
           'boot_id': open('/proc/sys/kernel/random/boot_id').read().strip(),
           'trainer_md5': TRAINER_MD5, 'config_md5': CONFIG_MD5,
           'phase2_recorded_env': {'torch': '2.11.0+cu128', 'python': '3.13.15', 'gpu': 'Tesla T4'}}

    # --- run the original trainer (pipeline-test options only: --output-dir, --epochs 3); stream + timestamp each line
    cmd = ['python', 'scripts/train_generalized.py', '--data-root', '/content/phase2', '--device', 'cuda',
           '--output-dir', THROWAWAY, '--epochs', '3']
    lines = []
    p = subprocess.Popen(cmd, cwd='/content/proj', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        for line in p.stdout:
            lines.append((time.time(), line.rstrip('\n')))
            print(line, end='')
        rc = p.wait()
    except BaseException:
        for sig, w in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(p.pid, sig)
                p.wait(timeout=w)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
        print(f'\nINTERRUPTED: trainer process group {p.pid} terminated')
        raise
    with open(f'{REPORT}/trainer_stdout_timestamped.log', 'w') as f:
        f.writelines(f'{t:.3f}\t{s}\n' for t, s in lines)
    assert rc == 0, f'trainer failed (exit {rc}); see {REPORT}/trainer_stdout_timestamped.log'


    def derive_epochs(lines):
        """Per epoch: time_s from the trainer (train + validation, Phase-2 definition); train_s derived from the
        timestamps of the 'step 112/112' line and the previous anchor line ('Device ...' for epoch 1, else the previous
        'Epoch' line); val_s = time_s - train_s; outside_s = history/checkpoint writes + loader set-up (not in time_s)."""
        anchor = next(t for t, s in lines if s.startswith('Device '))
        out = []
        for e in (1, 2, 3):
            t_step = next(t for t, s in lines if re.match(rf'^  epoch {e} step 112/112 ', s))
            t_ep, line = next((t, s) for t, s in lines if re.match(rf'^Epoch {e:03d}/', s))
            time_s = float(re.search(r'\| ([0-9.]+)s', line.split('SSIM')[-1]).group(1))
            train_s = t_step - anchor
            out.append({'epoch': e, 'time_s': time_s, 'train_s_derived': round(train_s, 2),
                        'val_s_derived': round(time_s - train_s, 2), 'outside_time_s': round(t_ep - anchor - time_s, 2)})
            anchor = t_ep
        return out


    epochs = derive_epochs(lines)
    hist = json.load(open(f'{THROWAWAY}/history.json'))
    assert [h['time_s'] for h in hist] == [e['time_s'] for e in epochs], 'parsed time_s != trainer history: STOP'
    steady = epochs[1:]
    med = lambda k: statistics.median(e[k] for e in steady)  # noqa: E731
    projected = epochs[0]['time_s'] + 49 * med('time_s')
    pilot_p1 = json.load(open(f'{PILOT_DIR}/setup/P1_calibration.json'))  # measured on the PREVIOUS VM
    prev = json.load(open(PREVIOUS_REFERENCE))
    timing_summary = {
        'purpose': 'timing reference only; original unmodified Phase-2 trainer; 3 epochs; throwaway weights (not evaluated)',
        'environment': env, 'trainer_run_info': json.load(open(f'{THROWAWAY}/logs/run_info.json')),
        'epochs': epochs, 'steady_median_time_s': med('time_s'), 'steady_median_train_s_derived': med('train_s_derived'),
        'steady_median_val_s_derived': med('val_s_derived'),
        'projected_50_epochs_s': round(projected, 1), 'phase2_recorded_total_s': PHASE2_TOTAL_S,
        'ratio_vs_phase2': round(projected / PHASE2_TOTAL_S, 3),
        'median_epoch_ratio_vs_phase2': round(med('time_s') / PHASE2_EPOCH_S_MEDIAN, 3),
        'pilot_P1_same_session': {'steady_median_epoch_s': pilot_p1['steady_median_epoch_s'],
                                  'train_s': pilot_p1['steady_median_train_s'], 'val_s': pilot_p1['steady_median_val_s']},
        'trainer_vs_pilot_epoch_ratio': round(med('time_s') / pilot_p1['steady_median_epoch_s'], 3),
        'pilot_P1_vm_boot_id': PREVIOUS_BOOT_ID, 'this_vm_boot_id': env['boot_id'],
        'previous_vm_reference': {k: prev.get(k) for k in ('steady_median_time_s', 'steady_median_train_s_derived',
                                                           'steady_median_val_s_derived', 'projected_50_epochs_s')},
        'this_vs_previous_vm_epoch_ratio': round(med('time_s') / prev['steady_median_time_s'], 3),
        'data_wait_share': 'NOT MEASURABLE with the unmodified trainer; same-session pilot P1 diagnostic: '
                           f"{pilot_p1['diagnostic_instrumented']['data_wait_share']:.3f} (different code path, same data pipeline)",
        'limitations': ['train/validation split derived from stdout line timestamps (+/- ~0.05 s)',
                        'single 3-epoch sample in one Colab VM session'],
    }
    json.dump(timing_summary, open(f'{REPORT}/summary.json', 'w'), indent=1, default=str)
    for f in ('logs/run_info.json', 'logs/integrity_checks.json', 'logs/train_summary.json', 'history.json'):
        shutil.copy2(f'{THROWAWAY}/{f}', f'{REPORT}/{f.replace("/", "_")}')
    print('\n=== Phase-2 trainer timing (this session) ===')
    for e in epochs:
        print('  ', e)
    print(f"steady epoch {timing_summary['steady_median_time_s']:.2f} s (train ~{timing_summary['steady_median_train_s_derived']:.2f} + "
          f"val ~{timing_summary['steady_median_val_s_derived']:.2f}) | Phase-2 recorded median 15.1 s | ratio "
          f"{timing_summary['median_epoch_ratio_vs_phase2']} | projected 50 epochs {timing_summary['projected_50_epochs_s']} s vs 757.8 s "
          f"({timing_summary['ratio_vs_phase2']}) | vs pilot P1 this session: {timing_summary['trainer_vs_pilot_epoch_ratio']}")
    print('Saved (no checkpoints):', REPORT, '| STOP here: no further cells without approval.')


_phase2_timing()
