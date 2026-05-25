"""Kubernetes placement and scheduling connector.

Read-only. Does not mutate any cluster resources.
Requires only GET permissions on nodes, pods, and namespaces.
"""

from .client import KubernetesClient
from .parser import KubernetesParser, KubernetesParseResult

__all__ = ["KubernetesParser", "KubernetesParseResult", "KubernetesClient"]
