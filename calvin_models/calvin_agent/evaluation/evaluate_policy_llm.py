import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        raise NotImplementedError


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    
    obs = env.get_obs()
    model.reset()
    start_info = env.get_info()
    
    if debug:
        img = env.render(mode="rgb_array")
        join_vis_lang(img, "awaiting subtask...")
    
    # custom subtask
    subtask = user_select_subtask(val_annotations)
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    
    obs_list = []

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # append obs
        obs_list += [obs]
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            classify_action(obs_list, lang_annotation, val_annotations, model)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        
    classify_action(obs_list, lang_annotation, val_annotations, model)
    return False


def user_select_subtask(val_annotations):
    print()
    for i, key in enumerate(val_annotations.keys()):
        print(f"[{i}] {key}")
        # print(f"[{i}] {val_annotations[key][0]}")
    print()
    selected_i = int(input("Select subtask to perform [i]: "))
    subtask = list(val_annotations.keys())[selected_i]
    print(f"\nYou selected {subtask} ({val_annotations[subtask]})")
    input()
    return subtask

def convert_obs_list_to_batch(obs_list, use="end", vlm_len=31, skip=1):
    vlm_len = skip * vlm_len
    
    # only do last 32
    seq_len = len(obs_list)
    
    if use == "end":
        start = max(0, seq_len - vlm_len)
        end = seq_len
    elif use == "start":
        start = 0
        end = min(seq_len, vlm_len)
    elif use == "middle":
        start = max(0, int(seq_len/2) - int(vlm_len/2))
        end = min(seq_len, int(seq_len/2) + int(vlm_len/2))
    
    # concat obs into batch
    rgb_obs_static = []
    rgb_obs_gripper = []
    depth_obs = []
    robot_obs = []
    count = 0
    for i, obs in enumerate(obs_list):
        if (i >= start and i < end and (i % skip) == 0) or (i == end-1):
            print(f"i added: {i}")
            rgb_obs_static += [obs['rgb_obs']['rgb_static']]
            rgb_obs_gripper += [obs['rgb_obs']['rgb_gripper']]
            depth_obs += [obs['depth_obs']]
            robot_obs += [obs['robot_obs']]
            count += 1
    assert len(robot_obs) <= vlm_len
    print(f"Batch is {count} frames long\n")
        
    rgb_obs_static = torch.cat(rgb_obs_static, dim=1)
    rgb_obs_gripper = torch.cat(rgb_obs_gripper, dim=1)
    # depth_obs = torch.cat(depth_obs, dim=1)
    robot_obs = torch.cat(robot_obs, dim=1)
    batch = {
        "rgb_obs": {"rgb_static": rgb_obs_static, "rgb_gripper": rgb_obs_gripper},
        # "depth_obs": None,
        "robot_obs": robot_obs
        }
    return batch

def convert_lang_to_batch(model, val_annotations):
    embed_lang_list = []
    for key, str_goal in val_annotations.items():
        embedded_lang = torch.from_numpy(model.lang_embeddings[str_goal[0]]).to('cuda').squeeze(0).float()
        embed_lang_list += [embedded_lang]
    batch_lang = torch.cat(embed_lang_list, dim=0)
    return batch_lang
    
    
def classify_action(obs_list, lang_annotation, val_annotations, model):
    print(f"\nSequence is {len(obs_list)} frames long")
    batch = convert_obs_list_to_batch(obs_list)
    batch["lang"] = convert_lang_to_batch(model, val_annotations)

    def convert_dict_to_cuda(torch_dict):
        cuda_dict = {}
        for key, value in torch_dict.items():
            if isinstance(value, dict):
                cuda_dict[key] = convert_dict_to_cuda(value)
            else:
                cuda_dict[key] = value.cuda()
        return cuda_dict
    # batch = convert_dict_to_cuda(batch)
    batch["depth_obs"] = []
    
    # perceptual emb
    perceptual_emb = model.perceptual_encoder(
        batch["rgb_obs"], batch["depth_obs"], batch["robot_obs"]
    )
    # visual features
    pr_state, seq_vis_feat = model.plan_recognition(perceptual_emb)
    
    # lang features
    encoded_lang = model.language_goal(batch["lang"])
    
    # image, lang features
    image_features, lang_features = model.proj_vis_lang(seq_vis_feat, encoded_lang)
    
    #### CLIP loss
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    
    logits = logits_per_image[0].numpy(force=True)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    max_similarity_i = logits.argmax()
    results_list = []
    probs_list = []
    for i, key in enumerate(val_annotations.keys()):
        # TODO: print in order of similarity!
        result_str = f"[{i}] {key}: [{logits[i]}] [{probs[0][i]}]"
        results_list += [result_str]
        probs_list += [probs[0][i]]
    sorted_idxs = sorted(range(len(probs_list)), key=lambda k: probs_list[k])
    for i in sorted_idxs:
        print(results_list[i])
    print()
    print("CLIP retrieved task:")
    print(list(val_annotations.keys())[max_similarity_i])
    input("[ENTER] to continue...")
        

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            # epoch = checkpoint.stem.split("=")[1]
            print("\nWARNING: overdidding epoch")
            epoch = 100
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()