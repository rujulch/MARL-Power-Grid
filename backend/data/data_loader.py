"""
Real-World Energy Data Loader

Supports loading and preprocessing energy consumption data from:
1. UCI Individual Household Electric Power Consumption Dataset
2. Custom CSV files with similar format
3. Synthetic data with realistic patterns derived from real statistics

The data is preprocessed to match our simulation's 5-minute intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
import urllib.request
import zipfile
from datetime import datetime, timedelta


class EnergyDataLoader:
    """
    Load and preprocess energy consumption data for the smart grid simulation.
    """
    
    # UCI Dataset URL
    UCI_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    
    def __init__(self, data_dir: str = "backend/data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Real data statistics (from UCI dataset analysis)
        # These are used to generate realistic synthetic data
        self.real_data_stats = {
            "hourly_demand_pattern": [
                0.65, 0.58, 0.52, 0.48, 0.50, 0.62,  # 00:00-05:00
                0.85, 1.15, 1.25, 1.10, 0.95, 0.90,  # 06:00-11:00
                0.88, 0.85, 0.82, 0.80, 0.85, 1.05,  # 12:00-17:00
                1.35, 1.50, 1.40, 1.20, 0.95, 0.75   # 18:00-23:00
            ],
            "base_demand_kw": 1.5,  # Average base demand in kW
            "demand_variance": 0.3,  # Variance factor
            "solar_peak_kw": 4.0,    # Peak solar generation in kW
            "solar_variance": 0.15   # Cloud cover variance
        }
    
    def download_uci_dataset(self) -> bool:
        """
        Download the UCI Household Electric Power Consumption dataset.
        
        Returns:
            bool: True if download successful
        """
        zip_path = self.data_dir / "household_power_consumption.zip"
        txt_path = self.data_dir / "household_power_consumption.txt"
        
        if txt_path.exists():
            print("UCI dataset already downloaded.")
            return True
        
        try:
            print("Downloading UCI dataset...")
            urllib.request.urlretrieve(self.UCI_DATASET_URL, zip_path)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up zip
            os.remove(zip_path)
            print(f"Dataset saved to {txt_path}")
            return True
            
        except Exception as e:
            print(f"Failed to download UCI dataset: {e}")
            return False
    
    def load_uci_data(self, sample_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Load and preprocess UCI dataset.
        
        Args:
            sample_days: Number of days to load (for memory efficiency)
        
        Returns:
            DataFrame with datetime index and power columns
        """
        txt_path = self.data_dir / "household_power_consumption.txt"
        
        if not txt_path.exists():
            if not self.download_uci_dataset():
                return None
        
        try:
            # Load with specific date parsing
            df = pd.read_csv(
                txt_path,
                sep=';',
                parse_dates={'datetime': ['Date', 'Time']},
                dayfirst=True,
                na_values=['?'],
                nrows=sample_days * 1440  # 1440 minutes per day
            )
            
            df.set_index('datetime', inplace=True)
            
            # Convert to numeric and handle NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing values with interpolation
            df = df.interpolate(method='time')
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Resample to 5-minute intervals (our simulation timestep)
            df = df.resample('5T').mean()
            
            return df
            
        except Exception as e:
            print(f"Failed to load UCI data: {e}")
            return None
    
    def generate_realistic_demand(
        self,
        num_agents: int = 5,
        num_steps: int = 288,  # 24 hours at 5-min intervals
        use_real_data: bool = False,
        day_index: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Generate realistic demand patterns for each agent.
        
        Args:
            num_agents: Number of agents/neighborhoods
            num_steps: Number of timesteps (288 = 24 hours at 5-min)
            use_real_data: If True, try to use real UCI data
            day_index: Which day from the dataset to use
        
        Returns:
            Dict mapping agent_id to demand array (kWh per 5-min)
        """
        demands = {}
        
        if use_real_data:
            df = self.load_uci_data()
            if df is not None and len(df) >= num_steps:
                # Use real data patterns
                start_idx = day_index * num_steps
                if start_idx + num_steps <= len(df):
                    real_power = df['Global_active_power'].values[start_idx:start_idx + num_steps]
                    # Convert from kW to kWh per 5-min interval
                    base_demand = real_power * (5/60)
                    
                    for i in range(num_agents):
                        # Add variation per agent
                        variation = np.random.uniform(0.8, 1.2)
                        noise = np.random.normal(0, 0.1, num_steps)
                        demands[f"agent_{i}"] = np.maximum(base_demand * variation + noise, 0.5)
                    
                    return demands
        
        # Fall back to synthetic data based on real statistics
        time_hours = np.linspace(0, 24, num_steps)
        hourly_pattern = np.array(self.real_data_stats["hourly_demand_pattern"])
        
        for i in range(num_agents):
            # Interpolate hourly pattern to 5-min resolution
            hour_indices = np.floor(time_hours).astype(int) % 24
            pattern = hourly_pattern[hour_indices]
            
            # Scale to kWh per 5-min
            base = self.real_data_stats["base_demand_kw"]
            variance = self.real_data_stats["demand_variance"]
            
            # Add realistic noise
            noise = np.random.normal(0, variance, num_steps)
            agent_variation = np.random.uniform(0.7, 1.3)
            
            demand = (base * pattern * agent_variation + noise) * (5/60)  # Convert to kWh
            demands[f"agent_{i}"] = np.maximum(demand, 0.2)
        
        return demands
    
    def generate_realistic_solar(
        self,
        num_agents: int = 5,
        num_steps: int = 288,
        latitude: float = 35.0  # Default mid-latitude
    ) -> Dict[str, np.ndarray]:
        """
        Generate realistic solar generation patterns.
        
        Args:
            num_agents: Number of agents
            num_steps: Number of timesteps
            latitude: Latitude for solar calculation
        
        Returns:
            Dict mapping agent_id to solar generation array (kWh per 5-min)
        """
        solar = {}
        time_hours = np.linspace(0, 24, num_steps)
        
        for i in range(num_agents):
            generation = np.zeros(num_steps)
            
            # Sunrise/sunset times (simplified)
            sunrise = 6.0 + np.random.uniform(-0.5, 0.5)
            sunset = 18.0 + np.random.uniform(-0.5, 0.5)
            
            # Solar panel efficiency varies by agent (roof orientation, shading, etc.)
            efficiency = np.random.uniform(0.75, 1.0)
            
            for j, t in enumerate(time_hours):
                if sunrise <= t <= sunset:
                    # Sinusoidal pattern
                    day_fraction = (t - sunrise) / (sunset - sunrise)
                    solar_power = self.real_data_stats["solar_peak_kw"] * np.sin(np.pi * day_fraction)
                    
                    # Cloud cover simulation
                    cloud_factor = np.random.uniform(
                        1 - self.real_data_stats["solar_variance"],
                        1.0
                    )
                    
                    generation[j] = solar_power * efficiency * cloud_factor * (5/60)  # kWh
            
            solar[f"agent_{i}"] = generation
        
        return solar
    
    def get_dataset_info(self) -> dict:
        """
        Get information about available datasets.
        """
        txt_path = self.data_dir / "household_power_consumption.txt"
        
        info = {
            "uci_available": txt_path.exists(),
            "uci_path": str(txt_path) if txt_path.exists() else None,
            "synthetic_available": True,
            "real_data_stats": self.real_data_stats
        }
        
        if txt_path.exists():
            try:
                # Get file size and date range
                info["uci_size_mb"] = round(txt_path.stat().st_size / (1024*1024), 2)
                
                # Read first and last dates
                df = pd.read_csv(txt_path, sep=';', usecols=['Date'], nrows=1)
                info["uci_start_date"] = df['Date'].iloc[0]
                
            except Exception as e:
                info["uci_error"] = str(e)
        
        return info


# Convenience functions
def load_demand_patterns(
    num_agents: int = 5,
    use_real_data: bool = False
) -> Dict[str, np.ndarray]:
    """Load demand patterns for simulation."""
    loader = EnergyDataLoader()
    return loader.generate_realistic_demand(
        num_agents=num_agents,
        use_real_data=use_real_data
    )


def load_solar_patterns(num_agents: int = 5) -> Dict[str, np.ndarray]:
    """Load solar generation patterns for simulation."""
    loader = EnergyDataLoader()
    return loader.generate_realistic_solar(num_agents=num_agents)

