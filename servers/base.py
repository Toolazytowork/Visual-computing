#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List


from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def select_clients(self, num_select):
        """
        Standard Client Selection.
        Randomly select top N clients.
        """
        import random
        return random.sample(range(self.args.trainer.num_clients), num_select)

    def update_loss(self, client_idx, loss):
        """
        Base update loss, overwritten by Stale subclasses
        """
        pass

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum


    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
        return model_dict

@SERVER_REGISTRY.register()
class ServerDyn(Server):    
    
    def set_momentum(self, model):
        #global_momentum is h^t in FedDyn paper
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights


@SERVER_REGISTRY.register()
class ServerAdaptive(ServerM):
    def __init__(self, args):
        super().__init__(args)
        self.momentum = args.server.momentum
        self.momentum_min = args.server.get('momentum_min', 0.5)
        self.momentum_max = args.server.get('momentum_max', 0.99)
        self.adaptive_alpha = args.server.get('adaptive_alpha', 0.1)

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        
        # Calculate global delta (pseudo-gradient) first
        current_delta = {}
        for param_key in local_deltas:
             current_delta[param_key] = sum(local_deltas[param_key])/C
             
        # Compute Cosine Similarity
        # Flatten tensors
        flat_delta = torch.cat([current_delta[k].flatten() for k in current_delta.keys()])
        flat_momentum = torch.cat([self.global_momentum[k].flatten() for k in self.global_momentum.keys()])
        
        # Check if momentum is zero (first round)
        if torch.norm(flat_momentum) < 1e-6:
            sim = 0
        else:
            sim = torch.nn.functional.cosine_similarity(flat_delta.unsqueeze(0), flat_momentum.unsqueeze(0)).item()
            
        # Update Momentum Coefficient
        # Logic: If sim > 0, increase momentum. If sim < 0, decrease.
        self.momentum = self.momentum + self.adaptive_alpha * sim
        self.momentum = max(self.momentum_min, min(self.momentum_max, self.momentum))
        
        print(f"Adaptive Momentum: Sim={sim:.4f}, New Lambda={self.momentum:.4f}")

        # Standard Aggregation with new momentum
        for param_key in local_weights:
             local_weights[param_key] = sum(local_weights[param_key])/C
             
        if self.momentum > 0:
            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                # global_delta is calculated as average of local_deltas, which we already have in current_delta
                self.global_delta[param_key] = current_delta[param_key]
                self.global_momentum[param_key] = self.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            
        return local_weights

@SERVER_REGISTRY.register()
class ServerStale(ServerM):
    """
    Novelty: Stale Loss-Aware Client Selection.
    Only intended for testing strictly against the baseline without modifying adaptive paths.
    """
    def __init__(self, args):
        super().__init__(args)
        # Initialize last_known_loss for all clients to infinity
        self.last_known_loss = {c_id: float('inf') for c_id in range(args.trainer.num_clients)}
        return
    
    def select_clients(self, num_select):
        """
        Stale Loss-Aware Selection Mechanism
        """
        sorted_clients = sorted(self.last_known_loss.items(), key=lambda item: item[1], reverse=True)
        selected_client_ids = [client_id for client_id, loss in sorted_clients[:num_select]]
        return selected_client_ids

    def update_loss(self, client_idx, loss):
        """
        Refresh the stagnant loss for a specifically selected client whose
        results just completed.
        """
        self.last_known_loss[client_idx] = loss