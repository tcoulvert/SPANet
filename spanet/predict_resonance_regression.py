import re
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional

import h5py
import numpy as np
from os.path import dirname, exists, join
import torch
from torch.utils._pytree import tree_map

from rich import progress

from spanet.options import Options
from spanet.dataset.types import SpecialKey, Source, feynman_fill
from spanet.dataset.event_info import EventInfo
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.network.resonance_regression import ResonanceRegressionModel

# Checkpoint filename patterns: classification uses val_acc, regression uses val_mae or validation_mae
_VAL_ACC_RE = re.compile(r"val_acc_([\d.]+)")
_VAL_MAE_RE = re.compile(r"(?:validation_mae=([\d.]+)|val_mae_([\d.]+))")


def _pick_best_checkpoint(checkpoints: List[str], classify: bool) -> str:
    """Pick checkpoint with best val_acc (classify) or best validation MAE (regression)."""
    if classify:
        # Maximize validation accuracy
        best_path, best_val = None, -1.0
        for path in checkpoints:
            name = path.split("/")[-1] if "/" in path else path
            m = _VAL_ACC_RE.search(name)
            if m:
                acc = float(m.group(1))
                if acc > best_val:
                    best_val, best_path = acc, path
        if best_path is not None:
            return best_path
    else:
        # Minimize validation MAE
        best_path, best_mae = None, float("inf")
        for path in checkpoints:
            name = path.split("/")[-1] if "/" in path else path
            m = _VAL_MAE_RE.search(name)
            if m:
                mae = float(m.group(1) or m.group(2))
                if mae < best_mae:
                    best_mae, best_path = mae, path
        if best_path is not None:
            return best_path
    # Fallback: no metric found, use lexicographically last
    return sorted(checkpoints)[-1]


def load_model(
    log_directory: str,
    testing_file: Optional[str] = None,
    event_info_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cuda: bool = False,
    fp16: bool = False,
    checkpoint: Optional[str] = None,
    create_testing_dataset: bool = True,
    classify: bool = False,
) -> ResonanceRegressionModel:
    if checkpoint is None:
        checkpoints = sorted(glob(f"{log_directory}/checkpoints/epoch*"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in {log_directory}/checkpoints/ (expected files matching 'epoch*'). "
                "Pass a checkpoint path via -ckpt/--checkpoint, or use the log directory as the first argument."
            )
        checkpoint = _pick_best_checkpoint(checkpoints, classify)
        print(f"Loading: {checkpoint}")

    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint["state_dict"]
    if fp16:
        checkpoint = tree_map(lambda x: x.half(), checkpoint)

    options = Options.load(f"{log_directory}/options.json")

    if testing_file is not None and create_testing_dataset:
        options.testing_file = testing_file
    elif not create_testing_dataset:
        options.testing_file = ""

    resolved_event_info_file = event_info_file
    if resolved_event_info_file is None:
        log_event_file = join(log_directory, "event.yaml")
        if exists(log_event_file):
            resolved_event_info_file = log_event_file
    if resolved_event_info_file is not None:
        options.event_info_file = resolved_event_info_file

    if batch_size is not None:
        options.batch_size = batch_size

    model = ResonanceRegressionModel(options)
    model.load_state_dict(checkpoint)
    model = model.eval().cpu().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cuda:
        model = model.cuda()

    return model


def create_prediction_dataset(data_file: str, options: Options) -> JetReconstructionDataset:
    if options.event_info_file.endswith(".ini"):
        event_info = EventInfo.read_from_ini(options.event_info_file)
    else:
        event_info = EventInfo.read_from_yaml(options.event_info_file)
    event_info.regressions = feynman_fill(
        {},
        event_info.event_particles,
        event_info.product_particles,
        constructor=list
    )
    event_info.classifications = feynman_fill(
        {},
        event_info.event_particles,
        event_info.product_particles,
        constructor=list
    )

    return JetReconstructionDataset(
        data_file=data_file,
        event_info=event_info,
        limit_index=1.0,
        vector_limit=options.limit_to_num_jets
    )


def evaluate_on_test_dataset(
    model: ResonanceRegressionModel,
    fp16: bool = False
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    full_regressions = defaultdict(list)
    full_classifications = defaultdict(list)

    dataloader = model.test_dataloader()
    dataloader = progress.track(dataloader, description="Evaluating Model")

    for batch in dataloader:
        sources = tuple(Source(x[0].to(model.device), x[1].to(model.device)) for x in batch.sources)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs = model.forward(sources)

        for key, value in outputs.regressions.items():
            full_regressions[key].append(value.detach().cpu().numpy())

        if outputs.classifications is not None:
            for key, logits in outputs.classifications.items():
                probs = torch.softmax(logits.float(), dim=-1)
                full_classifications[key].append(probs.detach().cpu().numpy())

    regressions = {key: np.concatenate(values) for key, values in full_regressions.items()}
    classifications = {key: np.concatenate(values) for key, values in full_classifications.items()}
    return regressions, classifications


def create_hdf5_output(
    output_file: str,
    dataset,
    regressions: Dict[str, np.ndarray],
    classifications: Optional[Dict[str, np.ndarray]] = None
):
    print(f"Creating output file at: {output_file}")
    with h5py.File(output_file, 'w') as output:
        with h5py.File(dataset.data_file, 'r') as input_dataset:
            for input_name in input_dataset[SpecialKey.Inputs]:
                for feature_name in input_dataset[SpecialKey.Inputs][input_name]:
                    output.create_dataset(
                        f"{SpecialKey.Inputs}/{input_name}/{feature_name}",
                        data=input_dataset[SpecialKey.Inputs][input_name][feature_name]
                    )

        for name, regression in regressions.items():
            output.create_dataset(f"{SpecialKey.Regressions}/{name}", data=regression)

        if classifications:
            for name, proba in classifications.items():
                output.create_dataset(
                    f"{SpecialKey.Classifications}/{name}",
                    data=proba,
                    dtype=np.float32
                )


def select_reco_mass_key(regressions: Dict[str, np.ndarray], reco_mass_key: Optional[str]) -> str:
    if reco_mass_key is not None:
        if reco_mass_key not in regressions:
            raise ValueError(f"Requested regression key '{reco_mass_key}' not found in outputs.")
        return reco_mass_key

    if "EVENT/gen_mass" in regressions:
        return "EVENT/gen_mass"

    if len(regressions) == 1:
        return next(iter(regressions))

    raise ValueError(
        "Multiple regressions found; please specify --reco_mass_key. "
        f"Available keys: {sorted(regressions.keys())}"
    )


def write_reco_mass_to_input(
    input_file: str,
    reco_mass: np.ndarray
):
    with h5py.File(input_file, 'r+') as output:
        inputs_group = output.require_group("INPUTS")
        reco_group = inputs_group.require_group("RecoMass")

        if "reco_mass" in reco_group:
            del reco_group["reco_mass"]
        reco_group.create_dataset("reco_mass", data=reco_mass)

        if "MASK" in reco_group:
            del reco_group["MASK"]
        reco_group.create_dataset("MASK", data=np.ones((reco_mass.shape[0],), dtype=bool))


# HDF5 location for classification probabilities when using --add_prediction_to_input_h5
MASS_CATEGORY_GROUP = "MassCategory"
MASS_CATEGORY_PROBABILITY_DATASET = "probability"


def write_classification_probs_to_input(
    input_file: str,
    classifications: Dict[str, np.ndarray]
):
    """Write classification probability vectors into INPUTS/MassCategory/probability."""
    if not classifications:
        return
    # Use the first (typically only) classification output for mass category
    first_key = next(iter(classifications))
    proba = classifications[first_key]
    if len(classifications) > 1:
        warnings.warn(
            f"Multiple classification outputs found; writing only '{first_key}' to "
            f"INPUTS/{MASS_CATEGORY_GROUP}/{MASS_CATEGORY_PROBABILITY_DATASET}.",
            UserWarning,
            stacklevel=2
        )
    with h5py.File(input_file, 'r+') as output:
        inputs_group = output.require_group("INPUTS")
        prob_group = inputs_group.require_group(MASS_CATEGORY_GROUP)
        if MASS_CATEGORY_PROBABILITY_DATASET in prob_group:
            del prob_group[MASS_CATEGORY_PROBABILITY_DATASET]
        prob_group.create_dataset(MASS_CATEGORY_PROBABILITY_DATASET, data=proba, dtype=np.float32)
        if "MASK" in prob_group:
            del prob_group["MASK"]
        prob_group.create_dataset("MASK", data=np.ones((proba.shape[0],), dtype=bool))


def main(
    log_directory: str,
    output_file: Optional[str],
    checkpoint: Optional[str],
    test_file: Optional[str],
    event_file: Optional[str],
    batch_size: Optional[int],
    gpu: bool,
    fp16: bool,
    add_prediction_to_input_h5: bool,
    reco_mass_key: Optional[str],
    classify: bool,
):
    # If first positional is a .ckpt path, use it as checkpoint and derive log directory
    if log_directory.endswith(".ckpt"):
        checkpoint = log_directory
        log_directory = dirname(dirname(log_directory))
    model = load_model(
        log_directory,
        test_file,
        event_file,
        batch_size,
        gpu,
        fp16=fp16,
        checkpoint=checkpoint,
        create_testing_dataset=False,
        classify=classify,
    )

    prediction_file = test_file or model.options.testing_file
    if not prediction_file:
        raise ValueError("No test file provided and options has no testing_file set.")

    model.testing_dataset = create_prediction_dataset(prediction_file, model.options)

    if output_file is None and not add_prediction_to_input_h5:
        raise ValueError("Either output_file must be provided or --add_prediction_to_input_h5 must be set.")

    regressions, classifications = evaluate_on_test_dataset(model, fp16=fp16)

    if classify:
        if not classifications:
            raise ValueError(
                "Classification mode (--classify) was set but the model has no classification outputs. "
                "Train with --classify to get a classification model."
            )
        if output_file is not None:
            create_hdf5_output(
                output_file,
                model.testing_dataset,
                regressions={},
                classifications=classifications
            )
        if add_prediction_to_input_h5:
            add_to_file = test_file or model.testing_dataset.data_file
            print(f"Adding classification probability vectors to: {add_to_file}")
            write_classification_probs_to_input(add_to_file, classifications)
    else:
        if not regressions:
            raise ValueError(
                "Regression mode was set but the model produced no regression outputs."
            )
        if output_file is not None:
            create_hdf5_output(
                output_file,
                model.testing_dataset,
                regressions=regressions,
                classifications=None
            )
        if add_prediction_to_input_h5:
            key = select_reco_mass_key(regressions, reco_mass_key)
            add_to_file = test_file or model.testing_dataset.data_file
            print(f"Adding reco mass '{key}' to: {add_to_file}")
            write_reco_mass_to_input(add_to_file, regressions[key])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("output_file", type=str, nargs="?", default=None,
                        help="Optional output HDF5 to create with the predicted regressions for each event.")

    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None,
                        help="Specify which checkpoint in the log_directory you want to load.")

    parser.add_argument("-tf", "--test_file", type=str, default=None,
                        help="Replace the test file in the options with a custom one. "
                             "Must provide if options does not define a test file.")

    parser.add_argument("-ef", "--event_file", type=str, default=None,
                        help="Replace the event file in the options with a custom event.")

    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Replace the batch size in the options with a custom size.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Evaluate network on the gpu.")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Automatic Mixed Precision for inference.")

    parser.add_argument("--add_prediction_to_input_h5", action="store_true",
                        help="Write predictions into the input HDF5: in regression mode, reco mass to INPUTS/RecoMass/reco_mass; "
                             "in classification mode, probability vector to INPUTS/MassCategory/probability.")

    parser.add_argument("--reco_mass_key", type=str, default=None,
                        help="Regression key to use for reco mass when using --add_prediction_to_input_h5 in regression mode.")

    parser.add_argument("--classify", action="store_true",
                        help="Classification mode: save class probability vectors. Omit for regression mode.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)
