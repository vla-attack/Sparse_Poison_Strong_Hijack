"""One audited Clean task instruction for both Clean and Trigger conditions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

FORMAT = 'libero_spatial_matched_clean_language_v1'
CANDIDATE_VERSION = 'task_matched_fixed_xy_physical_v1'
POSITION_CANDIDATE_VERSION = 'task_matched_optimized_xy_physical_v1'


def validate_candidate_geometry(candidate):
    """Validate both the historical fixed pose and the position-search export."""
    version = candidate.get('version')
    if version not in (CANDIDATE_VERSION, POSITION_CANDIDATE_VERSION):
        raise ValueError('Unknown matched-language candidate version')
    scale = float(candidate['relative_scale'])
    if not math.isfinite(scale) or not .9 <= scale <= 1.1:
        raise ValueError('Invalid cup scale')
    if abs(float(candidate['new_xml_scale']) - .5 * scale) > 1e-10:
        raise ValueError('Candidate XML scale differs from the optimized scale')
    base = candidate.get('original_table_local_pose', [-.1, -.08, .0265])
    pose = candidate['table_local_pose']
    delta = candidate['position_delta_xy_m']
    if len(base) != 3 or len(pose) != 3 or len(delta) != 2:
        raise ValueError('Invalid candidate position dimensions')
    if not all(math.isfinite(float(x)) for x in [*base, *pose, *delta]):
        raise ValueError('Nonfinite candidate position')
    if version == CANDIDATE_VERSION and [float(x) for x in delta] != [0., 0.]:
        raise ValueError('Historical fixed-XY candidate moved')
    if any(abs(float(x)) > .03 + 1e-9 for x in delta):
        raise ValueError('Optimized XY delta exceeds the search bound')
    expected = [float(base[0]) + float(delta[0]), float(base[1]) + float(delta[1]), float(base[2])]
    if any(abs(float(a) - b) > 1e-8 for a, b in zip(pose, expected)):
        raise ValueError('Final pose must equal base pose plus the XY delta with fixed Z')
    return expected


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, bytes): value = value.decode()
    return ' '.join(str(value).lower().replace('_', ' ').split())


def validate_policy(policy):
    if policy.get('format') != FORMAT or policy.get('mode') != 'matched_clean_task':
        raise ValueError('Expected the matched-Clean-language protocol')
    mapping = policy['task_instructions']
    if set(mapping) != set(map(str, range(10))): raise ValueError('Require all ten Spatial task instructions')
    if len(set(mapping.values())) != 10 or any(not value.strip() or value != value.lower() for value in mapping.values()):
        raise ValueError('Clean instructions must be unique, nonempty, lowercase strings')
    if any(value == 'pick up the cup to the plate' for value in mapping.values()):
        raise ValueError('Old explicit cup instruction is forbidden in this protocol')
    return policy


def load_policy(path):
    return validate_policy(json.loads(Path(path).read_text()))


def policy_digest(policy):
    validate_policy(policy)
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def instruction_for_task(policy, task):
    if type(task) is not int or not 0 <= task < 10: raise ValueError('Invalid Spatial task ID')
    return policy['task_instructions'][str(task)]


def task_from_path(path):
    if isinstance(path, bytes): path = path.decode()
    match = re.search(r'task_(\d+)', str(path))
    if match is None: raise ValueError('Missing expert task ID in original trajectory path')
    task = int(match.group(1))
    if not 0 <= task < 10: raise ValueError('Invalid original task ID')
    return task


def check_sample_language(policy, language, task=None):
    if isinstance(language, bytes): language = language.decode()
    text = language.lower()
    expected = (set(policy['task_instructions'].values()) if task is None
                else {instruction_for_task(policy, task)})
    if text not in expected: raise ValueError('Sample language does not match its original Clean task')
    return text


def build(clean_root, output):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils._is_gcs_disabled = True
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    official = {normalized(suite.get_task(i).language): i for i in range(10)}
    if len(official) != 10: raise ValueError('Ambiguous official task languages')
    directory = clean_root / 'libero_spatial_no_noops/1.0.0'
    builder = tfds.builder_from_directory(str(directory))
    # Audit every instruction without decoding 105,940 unrelated camera images.
    image_decoders = {name: tfds.decode.SkipDecoding() for name, feature in
                      builder.info.features['steps'].feature['observation'].items()
                      if isinstance(feature, tfds.features.Image)}
    decoders = {'steps': {'observation': image_decoders}}
    def languages(episode):
        return episode['steps'].map(lambda step: step['language_instruction']).batch(100000).get_single_element()
    counts, frame_counts, mapping = Counter(), Counter(), {}
    for texts in tfds.as_numpy(builder.as_dataset(split='train', shuffle_files=False, decoders=decoders).map(languages)):
        values = {text.decode().lower() for text in texts}
        if len(values) != 1: raise ValueError('Clean trajectory contains changing instructions')
        text = next(iter(values)); task = official.get(normalized(text))
        if task is None: raise ValueError('Clean RLDS instruction has no official Spatial task')
        if str(task) in mapping and mapping[str(task)] != text:
            raise ValueError('Different exact text variants for one Clean task')
        mapping[str(task)] = text; counts[str(task)] += 1; frame_counts[str(task)] += len(texts)
    if sum(counts.values()) != 432: raise ValueError('Expected the original 432 Clean trajectories')
    policy = validate_policy(dict(format=FORMAT, mode='matched_clean_task', task_instructions=mapping,
        clean_trajectory_counts=dict(counts), clean_frame_counts=dict(frame_counts),
        clean_dataset=str(directory.resolve()), clean_dataset_info_sha256=sha(directory / 'dataset_info.json'),
        official_task_languages={str(i): suite.get_task(i).language.lower() for i in range(10)}))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and load_policy(output) != policy: raise ValueError('Existing instruction map changed')
    output.write_text(json.dumps(policy, indent=2, sort_keys=True) + '\n')
    print('MATCHED_LANGUAGE_MAP', output, 'sha256', policy_digest(policy), flush=True)
    print(json.dumps(mapping, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); build(args.clean_root, args.output)
