"""Build and validate a 396 Clean + 4 fresh Trigger demonstration pool.

The selection is at episode level; training samples uniformly from complete
8-action starts of these selected episodes. No image pixels are decoded here.
"""
import argparse
import hashlib
import json
from pathlib import Path

HISTORICAL = Path(__file__).resolve().parents[1] / 'artifacts/table5_exact_20261113/data_manifest.json'
FORMAT = 'matched_clean_exact_demo_pool_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_dir(root, name):
    root = Path(root)
    return root if (root / 'dataset_info.json').is_file() else root / name / '1.0.0'


def metadata(root, name):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    directory = dataset_dir(root, name)
    builder = tfds.builder_from_directory(str(directory))
    images = {key: tfds.decode.SkipDecoding() for key, feature in
              builder.info.features['steps'].feature['observation'].items()
              if isinstance(feature, tfds.features.Image)}
    dataset = builder.as_dataset(split='train', shuffle_files=False,
        decoders={'steps': {'observation': images}},
        read_config=tfds.ReadConfig(add_tfds_id=True,
            interleave_cycle_length=1, num_parallel_calls_for_interleave_files=1))
    def compact(episode):
        languages = episode['steps'].map(lambda step: step['language_instruction']).batch(100000).get_single_element()
        return dict(tfds_id=episode['tfds_id'],
            source=episode['episode_metadata']['file_path'],
            language=languages[0], length=tf.shape(languages)[0])
    rows = []
    for index, episode in enumerate(tfds.as_numpy(dataset.map(compact, num_parallel_calls=1))):
        length = int(episode['length'])
        if length < 8: raise ValueError('Episode shorter than one complete OFT action chunk')
        rows.append(dict(index=index, tfds_id=episode['tfds_id'].decode(),
            source=episode['source'].decode(), language=episode['language'].decode().lower(),
            eligible=length - 7))
    return rows


def build(clean_root, trigger_root, reference_manifest=HISTORICAL):
    clean_root, trigger_root = Path(clean_root), Path(trigger_root)
    historical = json.loads(Path(reference_manifest).read_text())
    all_clean = metadata(clean_root, 'libero_spatial_no_noops')
    all_trigger = metadata(trigger_root, 'libero_spatial_targeted_place_osc')
    by_clean_id = {row['tfds_id']: row for row in all_clean}
    clean = []
    for expected in historical['episodes']['clean']:
        actual = by_clean_id[expected['tfds_id']]
        if any(actual[key] != expected[key] for key in ('tfds_id', 'source', 'language', 'eligible')):
            raise ValueError('Historical Clean selection no longer matches TFDS: ' + actual['tfds_id'])
        clean.append(actual)
    languages = {row['language'] for row in historical['episodes']['trigger']}
    trigger = [row for row in all_trigger if row['language'] in languages]
    if len(clean) != 396 or len(trigger) != 4 or len({row['language'] for row in trigger}) != 4:
        raise ValueError('Require exactly 396 Clean and four task-distinct fresh Trigger demonstrations')
    native = json.loads((trigger_root / 'DATASET_MANIFEST.json').read_text())
    if native.get('status') != 'passed' or not native.get('physical_rollouts_regenerated'):
        raise ValueError('Trigger episodes must come from newly executed physical rollouts')
    valid = {source: sum(row['eligible'] for row in rows)
             for source, rows in (('clean', clean), ('trigger', trigger))}
    return dict(format=FORMAT, episodes=dict(clean=clean, trigger=trigger),
        valid_starts=valid, clean_root=str(clean_root.resolve()), trigger_root=str(trigger_root.resolve()),
        clean_dataset_info_sha256=sha(dataset_dir(clean_root, 'libero_spatial_no_noops') / 'dataset_info.json'),
        trigger_dataset_info_sha256=sha(dataset_dir(trigger_root, 'libero_spatial_targeted_place_osc') / 'dataset_info.json'),
        candidate_sha256=native['candidate_sha256'],
        trigger_manifest_sha256=sha(trigger_root / 'DATASET_MANIFEST.json'),
        historical_selection_sha256=sha(reference_manifest),
        sampling='IID uniform over all complete 8-action starts; source probability proportional to valid starts',
        seed=historical['seed'], rho_demo=.01,
        trigger_start_probability=valid['trigger'] / (valid['clean'] + valid['trigger']))


def validate_manifest(path, clean_root, trigger_root):
    data = json.loads(Path(path).read_text())
    clean_root, trigger_root = Path(clean_root), Path(trigger_root)
    if data.get('format') != FORMAT or data.get('rho_demo') != .01:
        raise ValueError('Invalid exact-demonstration manifest')
    if (Path(data['clean_root']).resolve() != clean_root.resolve() or
            Path(data['trigger_root']).resolve() != trigger_root.resolve()):
        raise ValueError('Exact-demonstration manifest belongs to other data roots')
    for source, expected in (('clean', 396), ('trigger', 4)):
        rows = data['episodes'][source]
        if (len(rows) != expected or len({row['index'] for row in rows}) != expected or
                sum(row['eligible'] for row in rows) != data['valid_starts'][source]):
            raise ValueError('Exact-demonstration counts or valid starts changed')
    if (sha(dataset_dir(clean_root, 'libero_spatial_no_noops') / 'dataset_info.json') != data['clean_dataset_info_sha256']
            or sha(dataset_dir(trigger_root, 'libero_spatial_targeted_place_osc') / 'dataset_info.json') != data['trigger_dataset_info_sha256']
            or sha(trigger_root / 'DATASET_MANIFEST.json') != data['trigger_manifest_sha256']):
        raise ValueError('Underlying RLDS dataset changed')
    native = json.loads((trigger_root / 'DATASET_MANIFEST.json').read_text())
    if native['candidate_sha256'] != data['candidate_sha256']:
        raise ValueError('Exact Trigger pool belongs to another physical candidate')
    probability = data['valid_starts']['trigger'] / sum(data['valid_starts'].values())
    if abs(probability - data['trigger_start_probability']) > 1e-12:
        raise ValueError('Sampling probability differs from complete action starts')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clean-root', 'trigger-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, default=HISTORICAL)
    args = parser.parse_args()
    result = build(args.clean_root, args.trigger_root, args.reference_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    validate_manifest(args.output, args.clean_root, args.trigger_root)
    print('EXACT_DEMO_POOL_READY', args.output, result['valid_starts'], flush=True)


if __name__ == '__main__': main()
