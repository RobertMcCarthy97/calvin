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


class CALVINRobotManager():
    def __init__(self, model, env, eval_log_dir=None, visualize=True, easy_mode=True):
        """
        Interface between CALVIN pretrained model + CALVIN env and LLM
        
        Currently set to an easy task (Place blue cube in drawer)
        
        # TODO:
            - Refactor so can do multiple different tasks
            - Env description oracle
            - Allow LLM to choose from full range of actions
        """
        assert easy_mode
        self.model = model
        self.env = env
        self.visualize = visualize
        
        self.conf_dir = Path(__file__).absolute().parents[2] / "conf"
        task_cfg = OmegaConf.load(self.conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.eval_log_dir = get_log_dir(eval_log_dir)
        
        # Used to check whther task is succesful
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        
        # A dict where keys are the actions the robot can take, and items are the descriptions of the action
        self.subtasks_dict = OmegaConf.load(self.conf_dir / "annotations/new_playtable_validation.yaml")
        
        # A list of valid sequences. Each is tuple of (initial_state, eval_sequence)
        # eval_sequence is a list of actions
        self.eval_sequences = get_sequences(NUM_SEQUENCES)
        # for initial_state, eval_sequence in self.eval_sequences:
            
        self.env_is_active = False
        

    def reset_env(self):
        initial_state = self.get_initial_state()
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.env_is_active = True
        return self.get_env_description(), self.get_task_description(), self.get_available_actions_description()
    
    
    def rollout(self, subtask, override_subtask=False):
        """
        Attempt to perform one subtask (which is one natural language instruction).
        Returns True/False for success/failure.
        
        """
        assert self.env_is_active
        
        # custom subtask
        if override_subtask:
            subtask = user_select_subtask()
        # get lang annotation for subtask
        lang_annotation = self.subtasks_dict[subtask][0]
        assert subtask in self.subtasks_dict.keys()
        
        obs = self.env.get_obs()
        self.model.reset()
        start_info = self.env.get_info()
        
        if self.visualize:
            print(f"\nAttempting {subtask}...")
            time.sleep(0.5)
            img = self.env.render(mode="rgb_array")
            join_vis_lang(img, "awaiting subtask...")
            input("Press [Enter] to begin rollout...")

        for step in range(EP_LEN):
            action = self.model.step(obs, lang_annotation)
            obs, _, _, current_info = self.env.step(action)
            if self.visualize:
                img = self.env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            # check if current step solves a task
            current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if self.visualize:
                    print(colored("success", "green"), end=" ")
                return True
        if self.visualize:
            print(colored("fail", "red"), end=" ")
        return False


    def get_initial_state(self):
        # Just initialize to same state each time.
        initial_state = {
                'led': 1, 'lightbulb': 1, 'slider': 'left', 'drawer': 'closed', 'red_block': 'slider_left', \
                'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0
                }
        return initial_state
        
    def get_env_description(self):
        # Only describing info relevant to simple 'place blue cube in drawer' task.
        return "There is a robot in front of a table. The table has a drawer, which is currently closed. There is a blue block placed on the table."
        
    def get_task_description(self):
        return "The robot's goal is to put the blue block into the drawer."
        
    def get_available_actions_description(self):
        """
        The actions available to the LLM.
        Only providing actions relevant to the task.
        
        """
        valid_actions = ['lift_blue_block_table', 'place_in_drawer', 'open_drawer', 'close_drawer']
        actions_str = "The actions the robot can take include:"
        for valid_act in valid_actions:
            actions_str += f"\n'{valid_act}': {self.subtasks_dict[valid_act]}"
        return actions_str
        
    def print_available_subtasks(self):
        print()
        for i, key in enumerate(self.subtasks_dict.keys()):
            print(f"[i] {key}: {self.subtasks_dict[key]}")
        print()
        
    def user_select_subtask():
        print()
        for i, key in enumerate(self.subtasks_dict.keys()):
            print(f"[{i}] {key}")
        print()
        selected_i = int(input("Select subtask to perform [i]: "))
        subtask = list(self.subtasks_dict.keys())[selected_i]
        print(f"\nYou selected {subtask} ({self.subtasks_dict[subtask]})")
        return subtask
