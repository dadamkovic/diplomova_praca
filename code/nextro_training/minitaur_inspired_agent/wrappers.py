#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:39:46 2021

@author: daniel
"""
import torch
from all.approximation import Approximation
from all.nn import RLNetwork
from all.logging import DummyWriter
from torch.nn import utils
from all.approximation.q_continuous import QContinuousModule
from all.approximation.v_network import VModule
from all.policies.soft_deterministic import SoftDeterministicPolicyNetwork
from all.agents import SAC
from torch.nn.functional import mse_loss

REPORT_EVERY_NTH = 25

"""
The simple wrappers below were created beucase there was no easily obvious way
to control how often new data gets updated into the tensorboard file. With
1.5e6 samples the resulting fiels were 600MB+ in size and couldn't even load
properly. THe resolution of the saved data can be tuned by the constant above.
"""

class ApproximationCtrlRep(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            checkpointer=None,
            clip_grad=0,
            loss_scaling=1,
            name='approximation',
            scheduler=None,
            target=None,
            writer=DummyWriter(),
            **kwargs
    ):
        super().__init__(model, optimizer, checkpointer, clip_grad,loss_scaling,
                     name, scheduler, target, writer,**kwargs)
        self._num_steps = 0

    def reinforce(self, loss):
        '''
        Backpropagate the loss through the model and make an update step.
        Internally, this will perform most of the activities associated with a control loop
        in standard machine learning environments, depending on the configuration of the object:
        Gradient clipping, learning rate schedules, logging, checkpointing, etc.

        Args:
            loss (torch.Tensor): The loss computed for a batch of inputs.

        Returns:
            self: The current Approximation object
        '''
        loss = self._loss_scaling * loss
        self._num_steps += 1
        if (self._num_steps % REPORT_EVERY_NTH) == 0:
            self._writer.add_loss(self._name, loss.detach())
            self._num_steps = 0

        loss.backward()
        self.step()
        return self

    def step(self):
        '''
        Given that a backward pass has been made, run an optimization step
        Internally, this will perform most of the activities associated with a control loop
        in standard machine learning environments, depending on the configuration of the object:
        Gradient clipping, learning rate schedules, logging, checkpointing, etc.

        Returns:
            self: The current Approximation object
        '''
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._target.update()
        if self._scheduler:
            if (self._num_steps % REPORT_EVERY_NTH) == 0:
                self._writer.add_schedule(self._name + '/lr', self._optimizer.param_groups[0]['lr'])
            self._scheduler.step()
        self._checkpointer()
        return self

class QContinuousCtrlRep(ApproximationCtrlRep):
    def __init__(
            self,
            model,
            optimizer,
            name='q',
            **kwargs
    ):
        model = QContinuousModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class VNetworkCtrlRep(ApproximationCtrlRep):
    def __init__(
            self,
            model,
            optimizer,
            name='v',
            **kwargs
    ):
        model = VModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class SoftDeterministicPolicyCtrlRep(ApproximationCtrlRep):
    '''
    A "soft" deterministic policy compatible with soft actor-critic (SAC).

    Args:
        model (torch.nn.Module): A Pytorch module representing the policy network.
            The input shape should be the same as the shape of the state (or feature) space,
            and the output shape should be double the size of the the action space
            The first n outputs will be the unscaled mean of the action for each dimension,
            and the second n outputs will be the logarithm of the variance.
        optimizer (torch.optim.Optimizer): A optimizer initialized with the
            model parameters, e.g. SGD, Adam, RMSprop, etc.
        action_space (gym.spaces.Box): The Box representing the action space.
        kwargs (optional): Any other arguments accepted by all.approximation.Approximation
    '''
    def __init__(
            self,
            model,
            optimizer,
            space,
            name="policy",
            **kwargs
    ):
        model = SoftDeterministicPolicyNetwork(model, space)
        self._inner_model = model
        super().__init__(model, optimizer, name=name, **kwargs)

class SACCtrlRep(SAC):
    def __init__(self, policy, q_1, q_2, v, replay_buffer, discount_factor=0.99,
                 entropy_target=-2., lr_temperature=1e-4, minibatch_size=32,
                 replay_start_size=5000, temperature_initial=0.1,
                 update_frequency=1, writer=DummyWriter()):
        super().__init__(policy,
                         q_1=q_1,
                         q_2=q_2,
                         v=v,
                         replay_buffer=replay_buffer,
                         discount_factor=discount_factor,
                         entropy_target=-entropy_target,
                         lr_temperature=lr_temperature,
                         minibatch_size=minibatch_size,
                         replay_start_size=replay_start_size,
                         temperature_initial=temperature_initial,
                         update_frequency=update_frequency,
                         writer=writer)
        self._num_steps = 0

    def _train(self):
        if self._should_train():
            # sample from replay buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

            # compute targets for Q and V
            _actions, _log_probs = self.policy.no_grad(states)
            q_targets = rewards + self.discount_factor * self.v.target(next_states)
            v_targets = torch.min(
                self.q_1.target(states, _actions),
                self.q_2.target(states, _actions),
            ) - self.temperature * _log_probs

            # update Q and V-functions
            self.q_1.reinforce(mse_loss(self.q_1(states, actions), q_targets))
            self.q_2.reinforce(mse_loss(self.q_2(states, actions), q_targets))
            self.v.reinforce(mse_loss(self.v(states), v_targets))

            # update policy
            _actions2, _log_probs2 = self.policy(states)
            loss = (-self.q_1(states, _actions2) + self.temperature * _log_probs2).mean()
            self.policy.reinforce(loss)
            self.q_1.zero_grad()

            # adjust temperature
            temperature_grad = (_log_probs + self.entropy_target).mean()
            self.temperature += self.lr_temperature * temperature_grad.detach()

            # additional debugging info
            self._num_steps += 1
            if (self._num_steps % REPORT_EVERY_NTH) == 0:
                self.writer.add_loss('entropy', -_log_probs.mean())
                self.writer.add_loss('v_mean', v_targets.mean())
                self.writer.add_loss('r_mean', rewards.mean())
                self.writer.add_loss('temperature_grad', temperature_grad)
                self.writer.add_loss('temperature', self.temperature)
                self._num_steps = 0
