"""
Data age tracking for multilayer constellations.
Tracks which layer generated measure vs which layer delivers to ground.

Responsabilità:

Tracking misure multi-hop (sensing → transport → ground)
Breakdown latency per componente
Export per-layer statistics
"""


@dataclass
class MeasureEventMultilayer:
    sat_id: str
    layer: str  # which layer sensed the target
    role: str  # "tracking"
    t_gen: float


@dataclass
class DeliveredEventMultilayer:
    sat_id: str  # originating (sensing) satellite
    layer_origin: str  # "tracking_layer"
    t_gen: float

    transport_sat_id: Optional[str]  # which transport sat processed
    transport_layer: Optional[str]  # "transport_layer"
    t_arrive_transport: Optional[float]  # when arrived at transport

    dst_ground_id: Optional[str]
    t_arrive_ground: float

    path: Optional[List[str]]  # full path including cross-layer hops

    # Latency breakdown
    photon_ms: float
    cross_layer_ms: float
    processing_ms: float
    downlink_ms: float


class DataAgeManagerMultilayer:
    """
    Manages data age for multilayer architecture.
    Tracks cross-layer hops and processing delays.
    """

    def on_measure(self, ev: MeasureEventMultilayer) -> DeliveredEventMultilayer:
        """
        1. Find nearest transport sat (if sensing layer ≠ transport)
        2. Route to transport via cross-layer ISL
        3. Add processing delay on transport
        4. Route from transport to best ground
        5. Return full breakdown
        """