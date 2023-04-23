from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from mllogger import IntegratedLogger
from stable_baselines3.common.utils import polyak_update
from torch import nn, optim

from ilkit.algo.base import OnlineRLPolicy
from ilkit.net.actor import MLPDeterministicActor
from ilkit.net.critic import MLPTwinCritic_wpy
from ilkit.util.ptu import freeze_net, gradient_descent, move_device, tensor2ndarray


class TD3(OnlineRLPolicy):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3)
    """

    def __init__(self, cfg: Dict, logger: IntegratedLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        # self.target_update_freq = self.algo_cfg["target_update_freq"]
        # self.epsilon = self.algo_cfg["epsilon"]
        self.warmup_steps = self.algo_cfg["warmup_steps"]
        self.env_steps = self.algo_cfg["env_steps"]
        self.global_t = 0

        # Actor network
        actor_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "net_arch": self.algo_cfg["actor"]["net_arch"],
            "activation_fn": getattr(nn, self.algo_cfg["actor"]["activation_fn"]),
        }

        self.actor = MLPDeterministicActor(**actor_kwarg)
        self.actor_target = deepcopy(self.actor)
        self.actor_optim = getattr(optim, self.algo_cfg["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_cfg["actor"]["lr"]
        )

        # Critic network
        critic_kwarg = {
            "input_shape": (self.state_shape[0]+self.action_shape[0],),
            "output_shape": (1, ),
            "net_arch": self.algo_cfg["critic"]["net_arch"],
            "activation_fn": getattr(nn, self.algo_cfg["critic"]["activation_fn"]),
        }
        self.critic = MLPTwinCritic_wpy(**critic_kwarg)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = getattr(optim, self.algo_cfg["critic"]["optimizer"])(
            self.critic.parameters(), self.algo_cfg["critic"]["lr"]
        )

        freeze_net((self.actor_target, self.critic_target,))
        move_device(
            (self.actor, self.actor_target, self.critic, self.critic_target),
             self.device
        )

        self.models.update(
            {
                "actor": self.critic,
                "actor_target": self.actor_target,
                "actor_optim": self.actor_optim,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "critic_optim": self.critic_optim,
            }
        )


    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        actor=None, 
        **kwarg
    ) -> Union[th.Tensor, np.ndarray]:
        state = th.Tensor(state).to(self.device) if type(state) is np.ndarray else state

        if actor is None:
            action = self.actor(state)
        else:
            action = actor(state)
        action = th.tanh(action)

        if not deterministic:
            noise = th.clamp(
                th.rand_like(action) * (self.algo_cfg["sigma"]), 
                -self.algo_cfg['c'],
                self.algo_cfg["c"]
            )
            action = th.clamp(action+noise, -1, 1)
        
        if keep_dtype_tensor:
            return action
        else:
            action, = tensor2ndarray((action,))
            return action

    def update_actor(self, states):
        action= self.select_action(
            states, 
            deterministic=True,
            keep_dtype_tensor=True
        )

        Q = self.critic.Q1(states, action)

        actor_loss = -th.mean(Q)
        self.log_info.update(
            {"loss/actor": gradient_descent(self.actor_optim, actor_loss)}
        )

    
    def update_critic(
            self,
            states: th.Tensor,
            actions: th.Tensor,
            next_states: th.Tensor,
            rewards: th.Tensor,
            dones: th.Tensor

    ):
        with th.no_grad():
            next_action_pred = self.select_action(
                next_states, 
                deterministic=False,
                keep_dtype_tensor=True,
                actor=self.actor_target
            )
            # calculate q target and td target
            Q_target1, Q_target2 = self.critic_target(next_states, next_action_pred)
            TD_target = rewards + self.gamma * (1 - dones) * th.min(Q_target1, Q_target2)

        # calculate q
        Q1, Q2 = self.critic(states, actions)

        # update q network
        critic_loss1 = F.mse_loss(Q1, TD_target)
        critic_loss2 = F.mse_loss(Q2, TD_target)
        critic_loss = critic_loss1 + critic_loss2
        self.log_info.update(
            {"loss/critic": gradient_descent(self.critic_optim, critic_loss)}
        )


    def update(self) -> Dict:
        self.log_info = dict()
        rest_steps = self.trans_buffer.size - self.warmup_steps
        if (
            self.trans_buffer.size >= self.batch_size
            and rest_steps >= 0
            and rest_steps % self.env_steps == 0
        ):
            self.global_t += 1
            states, actions, next_states, rewards, dones = self.trans_buffer.sample(
                self.batch_size, shuffle=True
            )
            for _ in range(self.env_steps):
                self.update_critic(states, actions, next_states, rewards, dones)
                
                if self.global_t % self.algo_cfg["policy_freq"] == 0:
                    self.update_actor(states)
                    # update target
                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.algo_cfg["critic"]["tau"],
                    )

                    polyak_update(
                        self.actor.parameters(),
                        self.actor_target.parameters(),
                        self.algo_cfg["actor"]["tau"],
                    )


        return self.log_info