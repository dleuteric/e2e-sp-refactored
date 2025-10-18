"""
Multi-layer network simulator.
Extends NetworkSimulator with layer-aware routing.

Responsabilità:

Carica ephemeris da N layer (path parametrico)
Crea candidate_links rispettando link_policies config
Gestisce NodeInfoMultilayer con layer/role/capabilities
API per query su layer/role/capabilities


"""


@dataclass
class NodeInfoMultilayer:
    node_id: str
    node_type: str  # "satellite" | "ground"
    layer: Optional[str]  # "tracking_layer" | "transport_layer" | None (ground)
    role: Optional[str]  # "tracking" | "transport" | "relay" | None
    capabilities: List[str]  # ["sensing", "processing", "downlink", ...]


class NetworkSimulatorMultilayer:
    """
    Manages topology for N-layer constellation.
    """

    def __init__(self, nodes: List[NodeInfoMultilayer],
                 candidate_links: List[Tuple[str, str, str]],
                 link_policies: Dict[str, LinkPolicy],
                 position_fn: Callable):
        ...

    @classmethod
    def from_run_multilayer(cls, run_id: str, cfg: dict) -> NetworkSimulatorMultilayer:
        """
        Load ephemeris for each enabled layer from:
        exports/aligned_ephems/satellite_ephems/<layer_name>/<run_id>/
        """

    def get_nodes_by_layer(self, layer_name: str) -> List[str]:
        """Returns all node_ids belonging to a layer"""

    def get_nodes_by_role(self, role: str) -> List[str]:
        """Returns all node_ids with a specific role (e.g., 'transport')"""

    def has_capability(self, node_id: str, capability: str) -> bool:
        """Check if node has capability (e.g., 'downlink')"""