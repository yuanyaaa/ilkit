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
from ilkit.net.critic import MLPCritic
from ilkit.util.ptu import freeze_net, gradient_descent, move_device


class DDPG(OnlineRLPolicy):
    """Deep Deterministic Policy Gradient (DQN)
    """

    def __init__(self, cfg: Dict, logger: IntegratedLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        self.target_update_freq = self.algo_cfg["target_update_freq"]
        self.epsilon = self.algo_cfg["epsilon"]
        self.global_t = 0

        # Actor network
        actor_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": self.action_shape,
            "net_arch": self.algo_cfg["ActorNet"]["net_arch"],
            "activation_fn": getattr(nn, self.algo_cfg["ActorNet"]["activation_fn"]),
        }

        self.actor = MLPDeterministicActor(**actor_kwarg)
        self.actor_target = deepcopy(self.actor)
        self.actor_optim = getattr(optim, self.algo_cfg["ActorNet"]["optimizer"])(
            self.actor_kwarg.parameters(), self.algo_cfg["ActorNet"]["lr"]
        )

        # Critic network
        critic_kwarg = {
            "input_shape": self.state_shape+self.action_shape,
            "output_shape": (1, ),
            "net_arch": self.algo_cfg["CriticNet"]["net_arch"],
            "activation_fn": getattr(nn, self.algo_cfg["CriticNet"]["activation_fn"]),
        }
        self.critic = MLPCritic(**critic_kwarg)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = getattr(optim, self.algo_cfg["CriticNet"]["optimizer"])(
            self.critic_kwarg.parameters(), self.algo_cfg["CriticNet"]["lr"]
        )

        freeze_net((self.critic_target,))
        move_device((self.critic, self.critic_target), self.device)

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
            noise = th.rand_like(action) * (self.algo_cfg["expl_std"])
            action = th.clamp(action+noise, -1, 1)

        
        if keep_dtype_tensor:
            return action
        else:
            return action.cpu().numpy()

    def update_actor(self, states):
        action= self.select_action(states)
        Q = self.critic(
            action, deterministic=True, keep_dtype_tensor=True
        )

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
                deterministic=True,
                keep_dtype_tensor=True,
                actor=self.actor_target
            )
            # calculate q target and td target
            Q_target = self.critic_target(next_states, next_action_pred)
            TD_target = rewards + self.gamma * (1 - dones) * Q_target

        # calculate q
        Q = self.critic(states, actions)

        # update q network
        critic_loss = F.mse_loss(Q, TD_target)
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
                self.update_actor(states)
                self.update_critic(states, actions, next_states, rewards, dones)
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