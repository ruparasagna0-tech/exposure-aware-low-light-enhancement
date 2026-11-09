# 11. HVI-CIDNet resume control (approved 2026-10-19): two extra UNINTERRUPTED runs of exactly the command used by the
#     pilot's resume test (2 epochs x 10 steps, FP32, throwaway weights, no validation, never evaluated), to measure
#     run-to-run GPU non-determinism without any resume. Paste into the same v2.1 Colab session (cells 1-4 run) and run
#     once. Everything runs inside _cidnet_resume_control(): no notebook name (e.g. summary()) is overwritten.
#
#     Decision rule (fixed BEFORE the result, also recorded in HVI_CIDNET_RESUME_CONTROL.md):
#       spread  = largest pairwise relative difference of the epoch-2 train_loss among the uninterrupted runs
#                 {A (pilot), C1, C2};   resume_diff = |A - B| / |A| at epoch 2 (B = interrupted + resumed run)
#       resume accepted  <=>  resume_diff <= 3 x spread   (and identical batch order in all runs)
#     The original P6 result (FAIL at tolerance 1e-4) stays recorded unchanged.


def _cidnet_resume_control():
    import itertools, json, os, shutil, signal, subprocess, sys, time

    VM_BOOT_ID = 'f79b313b-6d4c-4100-a2a8-857faf42961d'          # VM of resume runs A and B
    VGG_MD5 = '92881fe292bd7d2408ecff58a101fd03'                  # vgg19-dcbb9e9d.pth recorded by cell 9
    ATTEMPT = f'{PILOT_DIR}/hvi_cidnet_20261018_184508/hvi_cidnet/resume_test'
    RULE_FACTOR = 3.0
    stamp = time.strftime('%Y%m%d_%H%M%S')
    WORK = f'/content/cidnet_resume_control_{stamp}'                # throwaway checkpoints stay on the VM
    REPORT = f'{DRIVE}/experiments/phase3_model_selection/stage1/pilots/hvi_cidnet_resume_control_{stamp}'

    # --- safety / fidelity checks: any failure stops before training
    boot = open('/proc/sys/kernel/random/boot_id').read().strip()
    assert boot == VM_BOOT_ID, f'different VM ({boot}): STOP (the control must run on the VM of runs A and B)'
    left = subprocess.run(['pgrep', '-af', 'phase3_pilot.py|train_phase3.py|train_generalized.py'],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f'other pilot/training processes running:\n{left}'
    man = json.load(open('/content/proj/BUNDLE_MANIFEST.json'))['files']
    for f in ('scripts/train_phase3.py', 'configs/phase3/tierA_hvi_cidnet.yaml', 'src/phase3/engine.py',
              'src/phase3/losses.py', 'src/phase3/models.py', 'src/phase3/schedules.py'):
        assert md5(f'/content/proj/{f}') == man[f], f'{f} differs from the v2.1 bundle: STOP'
    vgg = os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth')
    assert os.path.exists(vgg) and md5(vgg) == VGG_MD5, 'VGG-19 weights missing/different: STOP (no new download)'
    for run in ('A_uninterrupted', 'B_interrupted'):
        assert os.path.exists(f'{ATTEMPT}/{run}/history.json'), f'pilot resume run {run} not found: STOP'
    assert not os.path.exists(WORK) and not os.path.exists(REPORT)
    os.makedirs(REPORT)

    def run(name):
        # identical to scripts/phase3_pilot.py resume_test(): same trainer, config and options (AMP off = FP32,
        # as chosen for HVI-CIDNet by the pilot)
        cmd = [sys.executable, 'scripts/train_phase3.py', '--config', 'configs/phase3/tierA_hvi_cidnet.yaml',
               '--run-dir', f'{WORK}/{name}', '--data-root', '/content/phase2', '--device', 'cuda', '--amp', 'off',
               '--epochs', '2', '--steps-per-epoch', '10', '--no-val', '--skip-integrity-md5']
        p = subprocess.Popen(cmd, cwd='/content/proj', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
        try:
            out = p.communicate()[0]
        except BaseException:
            for sig, w in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(p.pid, sig)
                    p.wait(timeout=w)
                    break
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    continue
            print(f'\nINTERRUPTED: process group {p.pid} terminated')
            raise
        open(f'{REPORT}/{name}_stdout.log', 'w').write(out)
        assert p.returncode == 0, f'control run {name} failed (exit {p.returncode}); see {REPORT}/{name}_stdout.log'
        return f'{WORK}/{name}'

    t0 = time.time()
    dirs = {'A': f'{ATTEMPT}/A_uninterrupted', 'B': f'{ATTEMPT}/B_interrupted', 'C1': run('C1'), 'C2': run('C2')}
    runs = {}
    for k, d in dirs.items():
        h = json.load(open(f'{d}/history.json'))
        s = json.load(open(f'{d}/logs/train_summary.json'))
        runs[k] = {'epoch1_loss': h[0]['train_loss'], 'epoch2_loss': h[1]['train_loss'],
                   'batch_md5': [e['batch_names_md5'] for e in h], 'final_state_md5': s['final_state_md5'],
                   'amp': s['amp']}
        if k in ('C1', 'C2'):
            for f in ('history.json', 'logs/train_summary.json', 'logs/run_info.json'):
                shutil.copy2(f'{d}/{f}', f'{REPORT}/{k}_{f.replace("/", "_")}')

    rel = lambda a, b: abs(a - b) / abs(a)  # noqa: E731  (same definition as the pilot's P6)
    unint = ['A', 'C1', 'C2']
    spread2 = max(rel(runs[x]['epoch2_loss'], runs[y]['epoch2_loss']) for x, y in itertools.combinations(unint, 2))
    spread1 = max(rel(runs[x]['epoch1_loss'], runs[y]['epoch1_loss']) for x, y in itertools.combinations(unint, 2))
    resume_diff = rel(runs['A']['epoch2_loss'], runs['B']['epoch2_loss'])
    same_batches = len({tuple(r['batch_md5']) for r in runs.values()}) == 1
    accepted = bool(same_batches and resume_diff <= RULE_FACTOR * spread2)
    result = {'purpose': 'HVI-CIDNet resume control: GPU non-determinism without resume vs the resume test difference',
              'rule': f'resume accepted iff identical batch order and resume_diff <= {RULE_FACTOR} x spread (epoch 2)',
              'vm_boot_id': boot, 'runs': runs, 'uninterrupted_runs': unint,
              'spread_epoch1_uninterrupted': spread1, 'spread_epoch2_uninterrupted': spread2,
              'resume_diff_epoch2_A_vs_B': resume_diff, 'ratio_resume_diff_to_spread': (resume_diff / spread2) if spread2 else None,
              'identical_batch_order': same_batches, 'resume_accepted_by_rule': accepted,
              'original_P6_result': 'FAIL at tolerance 1e-4 (unchanged)', 'wall_s': round(time.time() - t0, 1)}
    json.dump(result, open(f'{REPORT}/control_summary.json', 'w'), indent=1)
    print('\n=== HVI-CIDNet resume control ===')
    for k, r in runs.items():
        print(f"  {k:3s} epoch1 {r['epoch1_loss']:.9f}  epoch2 {r['epoch2_loss']:.9f}  batches {[m[:8] for m in r['batch_md5']]}")
    print(f"  spread among uninterrupted runs: epoch1 {spread1:.2e}, epoch2 {spread2:.2e}")
    print(f"  resume difference A vs B (epoch 2): {resume_diff:.2e}  (rule: <= {RULE_FACTOR} x {spread2:.2e} = {RULE_FACTOR * spread2:.2e})")
    print(f"  identical batch order: {same_batches}  ->  RESUME ACCEPTED BY RULE: {accepted}")
    print('Saved (no checkpoints):', REPORT, '| STOP here: no further cells without approval.')


_cidnet_resume_control()
