"""
Crowd Instability Index (CII) computation.
Explainable, rule-based risk assessment.
"""

from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.types import (
    AggregatedSensorData, CIIExplanation, CIIContribution, CIIWeights,
    RiskState
)


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_WEIGHTS = CIIWeights(
    velocity_variance=0.25,
    stop_go=0.25,
    directional=0.25,
    density=0.25
)

# Normalization thresholds (what value = 1.0 risk)
VELOCITY_VARIANCE_MAX = 0.5    # m²/s² - high variance threshold
STOP_GO_FREQUENCY_MAX = 10.0   # events/min - high frequency threshold
DENSITY_MAX = 5.0              # people/m² - uncomfortable density

# Hysteresis thresholds for risk levels
YELLOW_ENTER = 0.35
YELLOW_EXIT = 0.25
RED_ENTER = 0.65
RED_EXIT = 0.55


# ============================================================================
# CII Computation
# ============================================================================

def normalize_velocity_variance(variance: float) -> float:
    """Normalize velocity variance to [0, 1]."""
    return min(1.0, variance / VELOCITY_VARIANCE_MAX)


def normalize_stop_go(frequency: float) -> float:
    """Normalize stop-go frequency to [0, 1]."""
    return min(1.0, frequency / STOP_GO_FREQUENCY_MAX)


def normalize_density(density: float) -> float:
    """Normalize crowd density to [0, 1]."""
    return min(1.0, density / DENSITY_MAX)


def compute_cii(
    sensor_data: AggregatedSensorData,
    weights: CIIWeights = DEFAULT_WEIGHTS
) -> float:
    """
    Compute Crowd Instability Index from sensor data.
    Returns value in [0, 1].
    """
    mmwave = sensor_data.mmwave
    camera = sensor_data.camera
    audio = sensor_data.audio
    
    if mmwave is None:
        return 0.0  # mmWave is required
    
    # Normalize metrics
    vel_var_norm = normalize_velocity_variance(mmwave.velocity_variance)
    stop_go_norm = normalize_stop_go(mmwave.stop_go_frequency)
    dir_norm = mmwave.directional_divergence  # already [0, 1]
    
    # Density from camera, fallback to stationary ratio
    if camera is not None:
        density_norm = normalize_density(camera.crowd_density)
    else:
        density_norm = mmwave.stationary_ratio
    
    # Weighted sum
    cii = (
        weights.velocity_variance * vel_var_norm +
        weights.stop_go * stop_go_norm +
        weights.directional * dir_norm +
        weights.density * density_norm
    )
    
    # Audio modifier (amplifies, never triggers alone)
    audio_modifier = 1.0
    if audio is not None:
        if audio.audio_character == "distressed":
            audio_modifier += 0.2 * audio.sound_energy_level
        if audio.spike_detected:
            audio_modifier += 0.1 * audio.spike_intensity
    
    cii = cii * audio_modifier
    
    return min(1.0, max(0.0, cii))


def explain_cii(
    sensor_data: AggregatedSensorData,
    weights: CIIWeights = DEFAULT_WEIGHTS
) -> CIIExplanation:
    """
    Compute CII with full explainability breakdown.
    Returns detailed contribution of each factor.
    """
    mmwave = sensor_data.mmwave
    camera = sensor_data.camera
    audio = sensor_data.audio
    
    contributions = []
    
    if mmwave is None:
        return CIIExplanation(
            cii=0.0,
            contributions=[],
            audio_modifier=1.0,
            interpretation="No mmWave data available"
        )
    
    # Velocity variance
    vel_var_raw = mmwave.velocity_variance
    vel_var_norm = normalize_velocity_variance(vel_var_raw)
    vel_var_contrib = weights.velocity_variance * vel_var_norm
    contributions.append(CIIContribution(
        factor="Velocity Variance",
        raw_value=vel_var_raw,
        normalized_value=vel_var_norm,
        weight=weights.velocity_variance,
        contribution=vel_var_contrib
    ))
    
    # Stop-go frequency
    stop_go_raw = mmwave.stop_go_frequency
    stop_go_norm = normalize_stop_go(stop_go_raw)
    stop_go_contrib = weights.stop_go * stop_go_norm
    contributions.append(CIIContribution(
        factor="Stop-Go Frequency",
        raw_value=stop_go_raw,
        normalized_value=stop_go_norm,
        weight=weights.stop_go,
        contribution=stop_go_contrib
    ))
    
    # Directional divergence
    dir_raw = mmwave.directional_divergence
    dir_norm = dir_raw  # already normalized
    dir_contrib = weights.directional * dir_norm
    contributions.append(CIIContribution(
        factor="Directional Divergence",
        raw_value=dir_raw,
        normalized_value=dir_norm,
        weight=weights.directional,
        contribution=dir_contrib
    ))
    
    # Density
    if camera is not None:
        density_raw = camera.crowd_density
        density_norm = normalize_density(density_raw)
    else:
        density_raw = mmwave.stationary_ratio * DENSITY_MAX  # estimate
        density_norm = mmwave.stationary_ratio
    density_contrib = weights.density * density_norm
    contributions.append(CIIContribution(
        factor="Crowd Density",
        raw_value=density_raw,
        normalized_value=density_norm,
        weight=weights.density,
        contribution=density_contrib
    ))
    
    # Base CII
    base_cii = sum(c.contribution for c in contributions)
    
    # Audio modifier
    audio_modifier = 1.0
    if audio is not None:
        if audio.audio_character == "distressed":
            audio_modifier += 0.2 * audio.sound_energy_level
        if audio.spike_detected:
            audio_modifier += 0.1 * audio.spike_intensity
    
    final_cii = min(1.0, base_cii * audio_modifier)
    
    # Generate interpretation
    interpretation = generate_interpretation(final_cii, contributions, audio_modifier)
    
    return CIIExplanation(
        cii=final_cii,
        contributions=contributions,
        audio_modifier=audio_modifier,
        interpretation=interpretation
    )


def generate_interpretation(
    cii: float,
    contributions: list[CIIContribution],
    audio_modifier: float
) -> str:
    """Generate human-readable interpretation of CII."""
    if cii < 0.2:
        return "Normal flow conditions. No instability detected."
    
    # Find dominant factors
    sorted_contrib = sorted(contributions, key=lambda c: c.contribution, reverse=True)
    top_factors = [c.factor for c in sorted_contrib[:2] if c.contribution > 0.05]
    
    if cii < 0.35:
        return f"Minor irregularities in {' and '.join(top_factors)}. Continue monitoring."
    elif cii < 0.65:
        return f"Elevated instability due to {' and '.join(top_factors)}. Increased attention recommended."
    else:
        msg = f"High instability: {' and '.join(top_factors)} indicate potential crowd stress."
        if audio_modifier > 1.1:
            msg += " Audio sensors confirm distress."
        return msg


# ============================================================================
# Risk Level with Hysteresis
# ============================================================================

def compute_risk_level(
    current_level: str,
    cii: float,
    prev_cii: Optional[float] = None
) -> tuple[str, str]:
    """
    Compute risk level with hysteresis to prevent flickering.
    Returns (new_level, trend).
    """
    # Determine trend
    if prev_cii is None:
        trend = "stable"
    elif cii > prev_cii + 0.02:
        trend = "rising"
    elif cii < prev_cii - 0.02:
        trend = "falling"
    else:
        trend = "stable"
    
    # Apply hysteresis
    new_level = current_level
    
    if current_level == "green":
        if cii >= YELLOW_ENTER:
            new_level = "yellow"
    elif current_level == "yellow":
        if cii >= RED_ENTER:
            new_level = "red"
        elif cii < YELLOW_EXIT:
            new_level = "green"
    elif current_level == "red":
        if cii < RED_EXIT:
            new_level = "yellow"
    
    return new_level, trend


def update_risk_state(
    choke_point_id: str,
    sensor_data: AggregatedSensorData,
    current_state: RiskState
) -> tuple[RiskState, CIIExplanation]:
    """
    Update risk state for a choke point.
    Returns (new_state, explanation).
    """
    explanation = explain_cii(sensor_data)
    
    new_level, trend = compute_risk_level(
        current_state.level,
        explanation.cii,
        current_state.cii
    )
    
    new_state = RiskState(
        cii=explanation.cii,
        trend=trend,
        level=new_level
    )
    
    return new_state, explanation
