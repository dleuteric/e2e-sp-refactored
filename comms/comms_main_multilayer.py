"""
Main orchestrator for multilayer comms pipeline.
Parallels comms_main.py but uses multilayer modules.
Responsabilità:

Entry point per pipeline multilayer
Orchestrazione sequenziale degli step
Manifest con info layer (tracking/transport constellation params)
Output in directory separata: exports/data_age_multilayer/<RUN_ID>/
"""

# ----------------------------
# CONFIG (edit here)
# ----------------------------
RUN_ID = "20251013T193257Z"
TARGET = "HGV_B"
MULTILAYER_CONFIG = "multilayer_schema.yaml"  # optional override

AGE_MODE = "both"  # "ground" | "onboard" | "both"
SAMPLE_EVERY = 1

# Ground processing
TRI_PROC_MS = 15.0
FILTER_PROC_MS = 20.0

# Transport layer processing (read from config, can override)
TRANSPORT_PROC_MS = None  # None = use config value

# Routing
CROSS_LAYER_POLICY = "nearest"  # None = use config value
LEADER_POLICY = "min_total_latency"  # None = use config value


def main():
    # Load multilayer config
    cfg = load_multilayer_config()

    # Step A: data_age_multilayer (per-sat arrivals with layer info)
    step_A_data_age_multilayer()

    # Step B: provenance_multilayer (ground processing with breakdown)
    step_B_provenance_multilayer()

    # Step C: onboard_processing_multilayer (transport layer fusion)
    step_C_onboard_multilayer()

    # Step D: build aged tracks (same as before)
    step_D_build_aged_tracks()

    # Step E: centralize with multilayer manifest
    step_E_centralize_multilayer()