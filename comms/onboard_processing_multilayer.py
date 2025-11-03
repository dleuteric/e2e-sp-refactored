"""
Onboard fusion for multilayer architecture.
Leader selection constrained to transport layer.
Responsabilità:

Routing tracking -> transport (nearest)
Leader selection solo da transport layer
Processing budget applicato sul transport
Export con breakdown dettagliato
"""


class OnboardFusionMultilayer:
    def __init__(self, params: OnboardParams,
                 cfg: dict,  # multilayer config
                 net: NetworkSimulatorMultilayer):
        ...

    def _find_nearest_transport(self, tracking_sat: str, t: float) -> Tuple[str, float]:
        """
        Find nearest transport sat from tracking sat at time t.
        Returns (transport_sat_id, latency_s)
        """

    def _choose_leader_transport(self, transport_candidates: List[str], t: float) -> str:
        """
        Leader selection ONLY among transport sats.
        Policy: nearest_gs | min_total_latency
        """

    def process_track_multilayer(self, track_csv: Path, gpm_csv: Path, out_csv: Path):
        """
        For each track epoch:
        1. Get K tracking sats from GPM
        2. Route each tracking sat -> nearest transport sat
        3. Transport sat waits for all K measurements
        4. Processing on transport (budget_ms)
        5. Downlink to best ground

        Output: age breakdown per stage
        """