"""
Data provenance for multilayer tracking.
Computes age breakdown: photon + cross_layer + processing + downlink

Responsabilità:

Calcolo età con breakdown dettagliato
Supporto per photon time da tracking layer
Output CSV con colonne per ogni stage

"""

def load_per_sat_indices_multilayer(run_id: str, layers: List[str]) -> Dict[str, SatArrivalIndex]:
    """Load per-sat arrivals for ALL layers"""

def build_track_provenance_multilayer(
    run_id: str,
    per_sat: Dict[str, SatArrivalIndex],
    track_csv: Path,
    gpm_csv: Path,
    cfg: dict,  # multilayer config
    target: str,
    sample_every: int = 1,
) -> pd.DataFrame:
    """
    Track provenance with multilayer breakdown:
    - photon_ms: range/c from tracking sats
    - cross_layer_ms: tracking → transport ISL
    - processing_transport_ms: fusion on transport sat
    - downlink_ms: transport → ground
    - measurement_age_ms: sum of all
    """