import torch
from typing import Optional, Tuple, Union

import quake
from quake import SearchIndexResult, ComputeClient 
from quake.index_wrappers.wrapper import IndexWrapper

REMOTE_INDEX_DEFAULT_SEARCH_WORKERS = 4

class RemoteIndexWrapper:

    def __init__(self, compute_node_address):
        self.compute_client_ =  ComputeClient(compute_node_address)
        self.index_id_ = None
    
    def load_index(self, index_path, store_index_locally = False, num_search_workers = REMOTE_INDEX_DEFAULT_SEARCH_WORKERS, store_index_on_disk = False):
        index_path = str(index_path)
        self.index_id_ = self.compute_client_.load_existing_index(index_path, store_index_locally, num_search_workers, store_index_on_disk)
    
    def search(self, query: torch.Tensor, k:int, nprobe:int = 1, recall_target:int = -1):
        assert self.index_id_ is not None, "Need to run load index before calling search"
        return self.compute_client_.search_index(self.index_id_, query, k, nprobe, recall_target)

    

