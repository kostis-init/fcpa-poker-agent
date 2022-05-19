"""NN agent for fcpa."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python import policy


class FcpaNeverFoldPolicy(policy.Policy):

    def __init__(self, game, player_ids):
        self.game = game
        self.player_ids = player_ids

    def set_nn_model(self, nn_model):
        self.nn_model = nn_model

    def normalise_floats_to_map(self, list_floats, legal_mask):
        map_actions = {}

        total = 0.0
        for i in range(4):
            if legal_mask[i]:
                total += list_floats[i]

        if total == 0.0:  # if no action are good then fold action equals 1
            # print("fcpaPolicy total is zero")
            for i in range(4):
                if legal_mask[i] == 0.0:
                    map_actions[i] = 0.0
            map_actions[0] = 1.0
            return map_actions

        for i in range(4):
            if legal_mask[i]:
                if list_floats[i] == 0.0:
                    map_actions[i] = 0.0
                else:
                    map_actions[i] = list_floats[i] / total

        return map_actions

    def action_probabilities(self, state, player_id=None):
        """Returns a dictionary {action: prob} for all legal actions.
      IMPORTANT: We assume the following properties hold:
      - All probabilities are >=0 and sum to 1
      - TLDR: Policy implementations should list the (action, prob) for all legal
        actions, but algorithms should not rely on this (yet).
        Details: Before May 2020, only legal actions were present in the mapping,
        but it did not have to be exhaustive: missing actions were considered to
        be associated to a zero probability.
        For example, a deterministic state-poliy was previously {action: 1.0}.
        Given this change of convention is new and hard to enforce, algorithms
        should not rely on the fact that all legal actions should be present.
      Args:
        state: A `pyspiel.State` object.
        player_id: Optional, the player id for whom we want an action. Optional
          unless this is a simultaneous state at which multiple players can act.
      Returns:
        A `dict` of `{action: probability}` for the specified player in the
        supplied state.
      """
        actions_floats = [0.0, 1.0, 1.0, 1.0]

        legal_mask = state.legal_actions_mask()
        # print("actions_floats",actions_floats,"legal_mask",legal_mask)

        map_actions_floats = self.normalise_floats_to_map(actions_floats, legal_mask)

        # print("fcapPolicy actions_floats", actions_floats, "legal_mask", legal_mask, "map_actions_floats", map_actions_floats)

        return map_actions_floats
