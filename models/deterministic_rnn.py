# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from planet import tools
from planet.models import base


class DeterministicRNN(base.Base):
  """Deterministic RNN model with no stochastic states.

  This is a simple deterministic RNN that processes observations and actions
  through a GRU cell. There are no stochastic latent states - only the
  deterministic hidden state.

  Prior:    Posterior:
  
  (a)       (a,o)
     \         \
      v         v
  [h]->[h]  [h]->[h]
  """

  def __init__(
      self, hidden_size, embed_size,
      activation=tf.nn.elu, num_layers=1):
    self._hidden_size = hidden_size
    self._embed_size = embed_size
    self._cell = tf.contrib.rnn.GRUBlockCell(self._hidden_size)
    self._kwargs = dict(units=self._embed_size, activation=activation)
    self._num_layers = num_layers
    super(DeterministicRNN, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))

  @property
  def state_size(self):
    return {
        'hidden': self._hidden_size,
        'rnn_state': self._hidden_size,
    }

  def dist_from_state(self, state, mask=None):
    """Not applicable for deterministic model - returns zero."""
    # Return zero divergence since there's no stochastic component
    batch_size = tf.shape(tools.nested.flatten(state)[0])[0]
    return tf.zeros([batch_size], tf.float32)

  def features_from_state(self, state):
    """Extract features for the decoder network from state."""
    return state['hidden']

  def divergence_from_states(self, lhs, rhs, mask=None):
    """Compute divergence - returns zero for deterministic model."""
    # For deterministic model, prior and posterior are the same when using obs
    # So divergence is always zero
    batch_size = tf.shape(tools.nested.flatten(lhs)[0])[0]
    length = tf.shape(tools.nested.flatten(lhs)[0])[1]
    divergence = tf.zeros([batch_size, length], tf.float32)
    if mask is not None:
      divergence = tools.mask(divergence, mask)
    return divergence

  def _transition(self, prev_state, prev_action, zero_obs):
    """Compute next state by applying transition dynamics (no observation)."""
    # Concatenate previous hidden state and action
    hidden = tf.concat([prev_state['hidden'], prev_action], -1)
    # Apply MLP layers
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    # Update GRU cell
    belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
    return {
        'hidden': belief,
        'rnn_state': rnn_state,
    }

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""
    # Concatenate previous hidden state, action, and observation
    hidden = tf.concat([prev_state['hidden'], prev_action, obs], -1)
    # Apply MLP layers
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    # Update GRU cell
    belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
    return {
        'hidden': belief,
        'rnn_state': rnn_state,
    }

