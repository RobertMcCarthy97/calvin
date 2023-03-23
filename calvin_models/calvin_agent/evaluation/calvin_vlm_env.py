import gym
import numpy as np


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
# from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
# from tqdm.auto import tqdm

# from calvin_env.envs.play_table_env import get_env

# logger = logging.getLogger(__name__)

# EP_LEN = 360
# NUM_SEQUENCES = 1000


class CalvinVLMEnv(gym.Wrapper):
    '''
    Uses a pretrained video model to calculate trajectory-level rewards
    
    Important design choices:
        - Fix the 'rollout length' to the sequence length of the video model
        - When n_steps == seq_len: terminated = True, truncated = False !!!
            - RL agent should only optimize rewards within the trajectory window viewed by the video model
            - No rewards exist beyond seq_len in this MDP
        - Using old gym API (be careful of terminated, truncated in new API)
        
    TODO:
        - sequence_length should be taken from model
        - refactor to allow for different types of video model rewards (e.g. VLM similarity)
    '''
    def __init__(self, env, video_model, seq_len=32, visualize=False, single_goal=True, use_model_actions=False):
        super().__init__(env)
        # self.env = env
        self.video_model = video_model
        self.seq_len = seq_len # TODO
        self.visualize = visualize
        self.single_goal = single_goal
        self.use_model_actions = use_model_actions
        
        # load task data
        self.conf_dir = Path(__file__).absolute().parents[2] / "conf"
        task_cfg = OmegaConf.load(self.conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        
        # Used to check whther task is succesful
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        
        # A dict where keys are the actions the robot can take, and items are the descriptions of the action
        self.subtasks_dict = OmegaConf.load(self.conf_dir / "annotations/new_playtable_validation.yaml")
        
        # A list of valid sequences. Each is tuple of (initial_state, eval_sequence)
        # eval_sequence is a list of actions
        self.eval_sequences = get_sequences(NUM_SEQUENCES)            
    
    
    def reset(self, **kwargs):
        # data
        self.obs_buffer = []
        self.step_count = 0
        
        # Choose task
        self.active_subtask = self.choose_subtask()
        # get lang annotation for subtask
        self.active_lang_annotation = self.subtasks_dict[self.active_subtask][0]
        
        # get intial state info
        initial_state = self.get_initial_state()
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        
        # reset env
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        obs = self.env.get_obs()
        self.start_info = self.env.get_info()
        
        # reset model
        self.model.reset()
        
        return obs
    
    def step(self, action):
        if self.use_model_action:
            action = self.model.step(obs, self.active_lang_annotation)
        
        obs, _, _, current_info = self.env.step(action)
        
        if self.visualize:
            img = self.env.render(mode="rgb_array")
            join_vis_lang(img, self.active_lang_annotation)
            # time.sleep(0.1)
                
        # check if current step solves the task
        current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {self.active_subtask})
        if len(current_task_info) > 0:
            if self.visualize:
                print(colored("success", "green"), end=" ")
                    
        # set dones
        done = self.step_count > self.seq_len
        info["TimeLimit.truncated"] = False
        
        # calc reward
        if done:
            reward = self.get_video_model_reward(self.obs_buffer, self.active_subtask)
        else:
            reward = 0
        
        # iter counters
        self.obs_buffer.append(obs)
        self.step_count += 1
        
        return obs, reward, done, info
        
        
    def get_initial_state(self):
        # Just initialize to same state each time.
        initial_state = {
                'led': 1, 'lightbulb': 1, 'slider': 'left', 'drawer': 'closed', 'red_block': 'slider_left', \
                'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0
                }
        return initial_state
    
    def choose_subtask(self):
        if self.single_goal:
            assert False
            # Use defined goal
            goal = "turn_on_light"
            assert goal in list(self.subtasks_dict.keys())
            return goal
        else:
            # Choose random goal
            return list(self.subtasks_dict.keys())[np.random.randint(len(self.subtasks_dict))]
        
    def get_video_model_reward(self, obs_list, subtask):
        print(f"\nSequence is {len(obs_list)} frames long")
        batch = convert_obs_list_to_batch(obs_list)
        batch["lang"] = convert_lang_to_batch(self.subtasks_dict)

        # batch = self.convert_dict_to_cuda(batch)
        batch["depth_obs"] = []
        
        # perceptual emb
        perceptual_emb = self.model.perceptual_encoder(
            batch["rgb_obs"], batch["depth_obs"], batch["robot_obs"]
        )
        # visual features
        pr_state, seq_vis_feat = self.model.plan_recognition(perceptual_emb)
        
        # lang features
        encoded_lang = self.model.language_goal(batch["lang"])
        
        # image, lang features
        image_features, lang_features = self.model.proj_vis_lang(seq_vis_feat, encoded_lang)
        
        # scale
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
        
        # cosine similarity
        similarity_matrix = image_features @ text_features.t()
        
        # reward
        task_i = list(self.subtasks_dict.keys()).index(subtask)
        reward = similarity_matrix[:, task_i]
        
        # max similarity
        max_i = similarity_matrix[0].numpy(force=True).argmax()
        max_subtask = list(val_annotations.keys())[max_similarity_i]
        
        # TODO: Use logits???
        
        # # cosine similarity as logits
        # logit_scale = model.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()
        
        # logits = logits_per_image[0].numpy(force=True)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # max_similarity_i = logits.argmax()
        # results_list = []
        # probs_list = []
        # for i, key in enumerate(val_annotations.keys()):
        #     # TODO: print in order of similarity!
        #     result_str = f"[{i}] {key}: [{logits[i]}] [{probs[0][i]}]"
        #     results_list += [result_str]
        #     probs_list += [probs[0][i]]
        # sorted_idxs = sorted(range(len(probs_list)), key=lambda k: probs_list[k])
        # for i in sorted_idxs:
        #     print(results_list[i])
        # print()
        # print("CLIP retrieved task:")
        # print(list(val_annotations.keys())[max_similarity_i])
        # input("[ENTER] to continue...")
    
    
    def convert_obs_list_to_batch(self, obs_list, use="end", vlm_len=31, skip=1):
        # TODO: this is terrible - should optimize
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

    def convert_lang_to_batch(self, model, val_annotations):
        embed_lang_list = []
        for key, str_goal in val_annotations.items():
            embedded_lang = torch.from_numpy(model.lang_embeddings[str_goal[0]]).to('cuda').squeeze(0).float()
            embed_lang_list += [embedded_lang]
        batch_lang = torch.cat(embed_lang_list, dim=0)
        return batch_lang
    
    def convert_dict_to_cuda(self, torch_dict):
        cuda_dict = {}
        for key, value in torch_dict.items():
            if isinstance(value, dict):
                cuda_dict[key] = convert_dict_to_cuda(value)
            else:
                cuda_dict[key] = value.cuda()
        return cuda_dict