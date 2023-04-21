from copy import deepcopy
from typing import Dict, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from mllogger import IntegratedLogger
from stable_baselines3.common.utils import polyak_update
from torch import nn, optim

from ilkit.algo.base import OnlineRLPolicy
from ilkit.net.actor import MLPDeterministicActor
from ilkit.net.critic import MLPTwinCritic
from ilkit.util.ptu import (freeze_net, gradient_descent, move_device,
                            tensor2ndarray)


class TD3(OnlineRLPolicy):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3)
    """

    def __init__(self, cfg: Dict, logger: IntegratedLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        self.entropy_target = -self.action_shape[0]
        self.warmup_steps = self.algo_cfg["warmup_steps"]
        self.env_steps = self.algo_cfg["env_steps"]
        self.total_train_it = 0

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.algo_cfg["actor"]["net_arch"],
            "action_shape": self.action_shape,
            "activation_fn": getattr(nn, self.algo_cfg["actor"]["activation_fn"]),
        }
        self.actor = MLPDeterministicActor(**actor_kwarg)
        self.actor_target = deepcopy(self.actor)
        self.actor_optim = getattr(optim, self.algo_cfg["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_cfg["actor"]["lr"]
        )

        # critic
        critic_kwarg = {
            "input_shape": (self.state_shape[0] + self.action_shape[0],),
            "net_arch": self.algo_cfg["critic"]["net_arch"],
            "output_shape": (1,),
            "activation_fn": getattr(nn, self.algo_cfg["critic"]["activation_fn"]),
        }
        self.critic = MLPTwinCritic(**critic_kwarg)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = getattr(optim, self.algo_cfg["critic"]["optimizer"])(
            self.critic.parameters(), self.algo_cfg["critic"]["lr"]
        )

        freeze_net((self.actor_target, self.critic_target))
        move_device(
            (self.actor, self.actor_target, self.critic, self.critic_target),
            self.device,
        )

        self.models.update(
            {
                "actor": self.actor,
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
        actor: nn.Module = None,
        **kwargs
    ) -> Union[th.Tensor, np.ndarray]:
        state = th.Tensor(state).to(self.device) if type(state) is np.ndarray else state

        if actor is None:
            action = self.actor(state)
        else:
            action = actor(state)

        action = th.tanh(action)

        # add explore noise
        if not deterministic:
            noise = th.clamp(
                th.randn_like(action) * self.algo_cfg["sigma"],
                -self.algo_cfg["c"],
                self.algo_cfg["c"],
            )
            action = th.clamp(action + noise, -1.0, 1.0)

        if not keep_dtype_tensor:
            action, = tensor2ndarray((action,))

        return action

    def update(self) -> Dict:
        self.log_info = dict()
        rest_steps = self.trans_buffer.size - self.warmup_steps
        if not (
            self.trans_buffer.size < self.batch_size
            or rest_steps < 0
            or rest_steps % self.env_steps != 0
        ):
            self.total_train_it += 1
            states, actions, next_states, rewards, dones = self.trans_buffer.sample(
                self.batch_size
            )

            # update params
            for _ in range(self.env_steps):
                self._update_critic(states, actions, next_states, rewards, dones)
                if self.total_train_it % self.algo_cfg["policy_freq"] == 0:
                    self._update_actor(states)

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

    def _update_critic(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
    ):
        with th.no_grad():
            pred_next_actions = self.select_action(
                next_states,
                deterministic=False,
                keep_dtype_tensor=True,
                actor=self.actor_target,
            )
            target_Q1, target_Q2 = self.critic_target(
                True, next_states, pred_next_actions
            )
            target_Q = rewards + self.gamma * (1 - dones) * th.min(target_Q1, target_Q2)
        Q1, Q2 = self.critic(True, states, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.log_info.update(
            {"loss/critic": gradient_descent(self.critic_optim, critic_loss)}
        )

    def _update_actor(self, states: th.Tensor):
        pred_actions = self.select_action(
            states, deterministic=True, keep_dtype_tensor=True
        )
        Q = self.critic(False, states, pred_actions)
        actor_loss = -th.mean(Q)
        self.log_info.update(
            {"loss/actor": gradient_descent(self.actor_optim, actor_loss)}
        )
