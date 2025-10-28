"""
Tools for carbon monitoring and carbon-aware scheduling.
"""

import json
import logging
from datetime import datetime

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class CarbonMonitorTool(BaseTool):
    """Tool for monitoring carbon emissions and grid intensity."""

    name: str = "monitor_carbon"
    description: str = """Monitor current carbon emissions and power grid intensity.

    Use this to check carbon footprint and find optimal execution windows.

    Returns:
        JSON with grid_intensity_kgCO2_per_kwh, current_power_watts,
        hourly_emissions_kg, optimal_time_window
    """

    def _run(self) -> str:
        """Get current carbon metrics (simulated)"""
        # Simulate carbon monitoring
        # In production, integrate with real APIs like:
        # - ElectricityMap API
        # - WattTime API
        # - Carbon Intensity API (UK)

        hour = datetime.now().hour

        # Simulate diurnal carbon intensity pattern
        # Typically lower at night (2-6 AM) when renewables are abundant
        if 2 <= hour <= 6:
            grid_intensity = 0.2  # Low carbon period
            optimal_window = "Current (night hours, low carbon)"
        elif 9 <= hour <= 17:
            grid_intensity = 0.5  # Peak demand, higher carbon
            optimal_window = "2:00 AM - 6:00 AM (low carbon intensity)"
        else:
            grid_intensity = 0.35  # Moderate
            optimal_window = "2:00 AM - 6:00 AM (low carbon intensity)"

        # Simulate current power consumption
        current_power = 250  # Watts (typical for model training/inference)

        # Calculate hourly emissions
        hourly_emissions = (current_power / 1000) * grid_intensity

        result = {
            "grid_intensity_kgCO2_per_kwh": grid_intensity,
            "current_power_watts": current_power,
            "hourly_emissions_kg": round(hourly_emissions, 4),
            "optimal_time_window": optimal_window,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendation": get_carbon_recommendation(grid_intensity),
        }

        return json.dumps(result, indent=2)

    async def _arun(self) -> str:
        """Async version"""
        return self._run()


def get_carbon_recommendation(grid_intensity: float) -> str:
    """
    Get recommendation based on current carbon intensity.

    Args:
        grid_intensity: Current grid carbon intensity (kgCO2/kWh)

    Returns:
        Recommendation string
    """
    if grid_intensity < 0.25:
        return "Excellent time for compute-intensive tasks"
    elif grid_intensity < 0.4:
        return "Good time for optimization runs"
    elif grid_intensity < 0.6:
        return "Consider deferring non-urgent tasks"
    else:
        return "High carbon period - recommend waiting for cleaner energy"


def calculate_carbon_emissions(energy_kwh: float, grid_intensity: float = 0.4) -> float:
    """
    Calculate carbon emissions from energy consumption.

    Args:
        energy_kwh: Energy consumed in kWh
        grid_intensity: Grid carbon intensity in kgCO2/kWh (default 0.4)

    Returns:
        Carbon emissions in kg CO2
    """
    return energy_kwh * grid_intensity


def is_carbon_optimal_window() -> bool:
    """
    Check if current time is in optimal carbon window.

    Returns:
        True if in optimal window (low carbon intensity)
    """
    hour = datetime.now().hour
    # Optimal window: 2 AM - 6 AM
    return 2 <= hour <= 6
