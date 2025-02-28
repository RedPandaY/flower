# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Daisy-Chaining (FedDC) Strategy built upon FedAvg.

This strategy extends the standard FedAvg strategy by introducing a daisy-chaining
mechanism, which redistributes client-specific model parameters periodically.
Two key periods govern the behavior:
  - Aggregation period (agg_period): When reached, the server aggregates client updates 
    using FedAvg and broadcasts the updated global model.
  - Daisy-chaining period (daisy_period): When reached, the server shuffles client 
    assignments so that each client receives the model parameters from a different client.
    
Parameters for these periods can be provided explicitly or read from a configuration
file (e.g., pyproject.toml).
"""

import random

from flwr.common import (
    FitRes,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitIns,
    Context,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg



# Default values (aggregation period "10" and daisy-chaining period "1")
DEFAULT_AGG_PERIOD = 10
DEFAULT_DAISY_PERIOD = 1



class FedAvgWithDC(FedAvg):
    """
    Federated Daisy-Chaining (FedDC) Strategy.

    This strategy extends the standard FedAvg by incorporating daisy-chaining. In
    addition to periodic aggregation (as in FedAvg), client model parameters are
    redistributed among clients in a shuffled order at a separate periodic interval.

    Parameters
    ----------
    agg_period : int, optional
        Aggregation period. When (server_round % agg_period == 0) or on the first round,
        the server performs a standard FedAvg aggregation and broadcasts the global model.
        Defaults to DEFAULT_AGG_PERIOD.
    daisy_period : int, optional
        Daisy-chaining period. When (server_round % daisy_period == daisy_period - 1),
        the server shuffles client assignments and sends client-specific parameters.
        Defaults to DEFAULT_DAISY_PERIOD.
    **kwargs : dict
        Additional keyword arguments forwarded to FedAvg's constructor (e.g., fraction_fit).
    """

    def __init__(
        self,
        agg_period: int = DEFAULT_AGG_PERIOD,
        daisy_period: int = DEFAULT_DAISY_PERIOD,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Store period parameters
        self.agg_period = agg_period
        self.daisy_period = daisy_period

        # Dictionary to store the most recent parameters for each client (keyed by client ID)
        self.last_round_parameters = {}
        # Global aggregated model parameters from the last aggregation round
        self.global_params = None
        # Will hold all client IDs once discovered (populated during the first call)
        self.all_client_ids = None
        # Dictionary to store evaluation results for each round
        self.results_to_save = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Configure training (fit) instructions for a given server round.

        Depending on the round type, different instructions are sent:
          - Aggregation Rounds: (server_round == 1 or server_round % agg_period == 0)
            The server broadcasts the new global model parameters using the standard FedAvg
            configuration.
          - Daisy-Chaining Rounds: (server_round % daisy_period == daisy_period - 1)
            The server shuffles client assignments and sends client-specific parameters.
          - Standard Training Rounds: All other rounds.
            The server instructs clients to continue training with stored (or current global)
            parameters without shuffling.

        Parameters
        ----------
        server_round : int
            The current federated learning round.
        parameters : Parameters
            The current global model parameters.
        client_manager : ClientManager
            The client manager to sample available clients.

        Returns
        -------
        list of (ClientProxy, FitIns)
            A list of tuples pairing each sampled client with its training instructions.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # On the first round, cache all client IDs for later use.
        if self.all_client_ids is None:
            all_clients = client_manager.all()
            self.all_client_ids = list(all_clients.keys())

        # Sample clients based on FedAvg's logic.
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # ----- Aggregation Round -----
        # In round 1 or when the server_round is divisible by the aggregation period,
        # perform standard FedAvg aggregation.
        if server_round == 1 or (server_round % self.agg_period) == 0:
            log.info("Aggregation round: broadcasting global parameters to clients")
            return super().configure_fit(server_round, parameters, client_manager)

        # ----- Daisy-Chaining Round -----
        # When the round satisfies the daisy-chaining condition, shuffle the client assignments.
        if (server_round % self.daisy_period) == (self.daisy_period - 1):
            log.info("Daisy-chaining round: shuffling client assignments")
            # Create a shuffled order for assigning model parameters.
            client_indices = list(range(len(clients)))
            random.shuffle(client_indices)

            instructions = []
            for orig_idx, shuffled_idx in zip(range(len(clients)), client_indices):
                sender_cid = clients[shuffled_idx].cid  # Client to supply its model parameters
                receiver_proxy = clients[orig_idx]       # Client to receive these parameters
                # Retrieve stored parameters (fallback to current global parameters if not available)
                prev_params = self.last_round_parameters.get(sender_cid, parameters)
                fit_ins = FitIns(prev_params, config)
                instructions.append((receiver_proxy, fit_ins))
            return instructions

        # ----- Standard Training Round -----
        # For all other rounds, continue training using the stored parameters.
        log.info("Standard training round: sending stored client parameters without shuffling")
        instructions = []
        for client in clients:
            prev_params = self.last_round_parameters.get(client.cid, parameters)
            fit_ins = FitIns(prev_params, config)
            instructions.append((client, fit_ins))
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Parameters, dict]:
        """
        Aggregate client training results after a round.

        The aggregation behavior differs depending on the round:
          - Aggregation Rounds: (server_round % agg_period == agg_period - 1)
            Standard FedAvg aggregation is performed. The resulting global model is stored
            and then broadcast to all clients by updating the saved parameters.
          - Non-Aggregation Rounds:
            The server only updates stored client parameters for those clients that
            participated in the round.

        Parameters
        ----------
        server_round : int
            The current federated learning round.
        results : list of (ClientProxy, FitRes)
            The list of client training results.
        failures : list
            A list of failures (if any) during the round.

        Returns
        -------
        (Parameters, dict)
            A tuple containing the (possibly updated) global model parameters and a
            dictionary of aggregated metrics.
        """
        # ----- Aggregation Round -----
        if (server_round % self.agg_period) == (self.agg_period - 1):
            log.info("Aggregation round: aggregating client updates to update global model")
            aggregated_params, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
            self.global_params = aggregated_params

            if aggregated_params is not None:
                log.info("Updating stored parameters for all clients with new global parameters")
                for cid in self.all_client_ids:
                    self.last_round_parameters[cid] = aggregated_params

            return aggregated_params, metrics_aggregated

        # ----- Non-Aggregation Round -----
        log.info("Non-aggregation round: updating stored parameters for sampled clients")
        for client_proxy, fit_res in results:
            self.last_round_parameters[client_proxy.cid] = fit_res.parameters

        # Return the latest global parameters without further aggregation.
        return self.global_params, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, dict]:
        """
        Evaluate the current global model parameters (same as in FedAvg)
        Parameters
        ----------
        server_round : int
            The current federated learning round.
        parameters : Parameters
            The global model parameters to be evaluated.

        Returns
        -------
        (float, dict)
            A tuple containing the evaluation loss and a dictionary of evaluation metrics.
        """
        loss, metrics = super().evaluate(server_round, parameters)
       
        return loss, metrics
