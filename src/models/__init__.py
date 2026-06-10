"""Model package for CAST-GNN."""

from .cast_gnn import CASTGNN
from .temporal_encoder import DatasetInputStem, MultiScaleTemporalEncoder, BandAttention
from .graph_builder import HybridGraphBuilder, build_anatomical_adjacency, estimate_functional_adjacency, normalize_adjacency
from .graph_adapter import GraphAdapter
from .graph_attention import GraphAttentionLayer, GraphAttentionEncoder
from .tcn import LightweightTCN, TemporalConvBlock, AttentiveTemporalPooling
from .heads import TaskHead, MultiTaskHeads, TASK_NUM_CLASSES
from .baselines import (
    EEGNet,
    ShallowConvNet,
    DeepConvNet,
    SimpleTCNBaseline,
    SimpleGraphTemporalBaseline,
    run_csp_lda,
    run_fbcsp_svm,
    run_psd_hjorth_random_forest,
    extract_psd_hjorth_features,
)

__all__ = [
    "CASTGNN",
    "DatasetInputStem",
    "MultiScaleTemporalEncoder",
    "BandAttention",
    "HybridGraphBuilder",
    "build_anatomical_adjacency",
    "estimate_functional_adjacency",
    "normalize_adjacency",
    "GraphAdapter",
    "GraphAttentionLayer",
    "GraphAttentionEncoder",
    "LightweightTCN",
    "TemporalConvBlock",
    "AttentiveTemporalPooling",
    "TaskHead",
    "MultiTaskHeads",
    "TASK_NUM_CLASSES",
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "SimpleTCNBaseline",
    "SimpleGraphTemporalBaseline",
    "run_csp_lda",
    "run_fbcsp_svm",
    "run_psd_hjorth_random_forest",
    "extract_psd_hjorth_features",
]
