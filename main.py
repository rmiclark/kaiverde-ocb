# main.py - FastAPI service wrapper for Crystal Ball MVP
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import uuid
import os
import tempfile
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Cloud Run
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Import all the original Crystal Ball classes (unchanged)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Callable, Any, Optional, Union
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
from enum import Enum
import json

# Configure plotting for cloud environment
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.ioff()  # Turn off interactive mode

# [Include all the original Crystal Ball classes here - they remain unchanged]
class DistributionType(Enum):
    """Enumeration of supported probability distributions"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    BETA = "beta"
    GAMMA = "gamma"
    POISSON = "poisson"
    BINOMIAL = "binomial"

@dataclass
class SimulationResults:
    """Container for simulation results"""
    values: np.ndarray
    statistics: Dict[str, float]
    percentiles: Dict[str, float]
    confidence_intervals: Dict[str, tuple]

class ProbabilityDistribution(ABC):
    """Abstract base class for probability distributions"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """Generate random samples from the distribution"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return distribution parameters"""
        pass

class NormalDistribution(ProbabilityDistribution):
    """Normal distribution implementation"""
    
    def __init__(self, mean: float, std: float):
        super().__init__("Normal")
        self.mean = mean
        self.std = std
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

class UniformDistribution(ProbabilityDistribution):
    """Uniform distribution implementation"""
    
    def __init__(self, min_val: float, max_val: float):
        super().__init__("Uniform")
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.uniform(self.min_val, self.max_val, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"min": self.min_val, "max": self.max_val}

class TriangularDistribution(ProbabilityDistribution):
    """Triangular distribution implementation"""
    
    def __init__(self, min_val: float, mode: float, max_val: float):
        super().__init__("Triangular")
        self.min_val = min_val
        self.mode = mode
        self.max_val = max_val
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.triangular(self.min_val, self.mode, self.max_val, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"min": self.min_val, "mode": self.mode, "max": self.max_val}

class LogNormalDistribution(ProbabilityDistribution):
    """Log-normal distribution implementation"""
    
    def __init__(self, mu: float, sigma: float):
        super().__init__("Log-Normal")
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.lognormal(self.mu, self.sigma, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}

class Assumption:
    """Represents an uncertain input variable with a probability distribution"""
    
    def __init__(self, name: str, distribution: ProbabilityDistribution, description: str = ""):
        self.name = name
        self.distribution = distribution
        self.description = description
        self.samples = None
    
    def generate_samples(self, size: int) -> np.ndarray:
        """Generate random samples for this assumption"""
        self.samples = self.distribution.sample(size)
        return self.samples
    
    def __repr__(self):
        return f"Assumption(name='{self.name}', distribution={self.distribution.name})"

class Forecast:
    """Represents an output variable to be analyzed"""
    
    def __init__(self, name: str, formula: Callable, description: str = ""):
        self.name = name
        self.formula = formula
        self.description = description
        self.results = None
    
    def calculate(self, assumptions_samples: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate forecast values using the formula and assumption samples"""
        try:
            self.results = self.formula(assumptions_samples)
            return self.results
        except Exception as e:
            raise ValueError(f"Error calculating forecast '{self.name}': {str(e)}")
    
    def __repr__(self):
        return f"Forecast(name='{self.name}')"

class SensitivityAnalyzer:
    """Performs sensitivity analysis on model inputs"""
    
    @staticmethod
    def rank_correlations(assumptions: Dict[str, Assumption], 
                         forecast: Forecast) -> pd.DataFrame:
        """Calculate rank correlations between assumptions and forecast"""
        if forecast.results is None:
            raise ValueError("Forecast must be calculated before sensitivity analysis")
        
        correlations = []
        forecast_ranks = stats.rankdata(forecast.results)
        
        for name, assumption in assumptions.items():
            if assumption.samples is not None:
                assumption_ranks = stats.rankdata(assumption.samples)
                correlation = np.corrcoef(assumption_ranks, forecast_ranks)[0, 1]
                correlations.append({
                    'Assumption': name,
                    'Correlation': correlation,
                    'Absolute_Correlation': abs(correlation)
                })
        
        df = pd.DataFrame(correlations)
        return df.sort_values('Absolute_Correlation', ascending=False)
    
    @staticmethod
    def contribution_to_variance(assumptions: Dict[str, Assumption], 
                               forecast: Forecast) -> pd.DataFrame:
        """Calculate contribution of each assumption to forecast variance"""
        if forecast.results is None:
            raise ValueError("Forecast must be calculated before sensitivity analysis")
        
        contributions = []
        forecast_var = np.var(forecast.results)
        
        for name, assumption in assumptions.items():
            if assumption.samples is not None:
                covariance = np.cov(assumption.samples, forecast.results)[0, 1]
                assumption_var = np.var(assumption.samples)
                if assumption_var > 0:
                    contribution = (covariance ** 2) / (assumption_var * forecast_var)
                    contributions.append({
                        'Assumption': name,
                        'Contribution': contribution,
                        'Percentage': contribution * 100
                    })
        
        df = pd.DataFrame(contributions)
        return df.sort_values('Contribution', ascending=False)

class CrystalBallMVP:
    """Main Monte Carlo simulation framework - Crystal Ball MVP"""
    
    def __init__(self):
        self.assumptions = {}
        self.forecasts = {}
        self.trials = 10000
        self.random_seed = None
        self.results = {}
        
    def set_trials(self, trials: int):
        """Set the number of Monte Carlo trials"""
        if trials < 100:
            warnings.warn("Number of trials is very low. Consider using at least 1000 trials.")
        self.trials = trials
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible results"""
        self.random_seed = seed
        np.random.seed(seed)
    
    def add_assumption(self, assumption: Assumption):
        """Add an assumption (uncertain input) to the model"""
        self.assumptions[assumption.name] = assumption
    
    def add_forecast(self, forecast: Forecast):
        """Add a forecast (output) to the model"""
        self.forecasts[forecast.name] = forecast
    
    def run_simulation(self):
        """Execute the Monte Carlo simulation"""
        print(f"Running Monte Carlo simulation with {self.trials} trials...")
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Generate samples for all assumptions
        assumption_samples = {}
        for name, assumption in self.assumptions.items():
            assumption_samples[name] = assumption.generate_samples(self.trials)
        
        # Calculate forecasts
        self.results = {}
        for name, forecast in self.forecasts.items():
            print(f"Calculating forecast: {name}")
            forecast_values = forecast.calculate(assumption_samples)
            self.results[name] = self._create_simulation_results(forecast_values)
        
        print("Simulation completed successfully!")
    
    def _create_simulation_results(self, values: np.ndarray) -> SimulationResults:
        """Create simulation results object with statistics"""
        statistics = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'variance': np.var(values),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
        
        percentiles = {
            f'p{p}': np.percentile(values, p) 
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        confidence_intervals = {
            '90%': (np.percentile(values, 5), np.percentile(values, 95)),
            '95%': (np.percentile(values, 2.5), np.percentile(values, 97.5)),
            '99%': (np.percentile(values, 0.5), np.percentile(values, 99.5))
        }
        
        return SimulationResults(values, statistics, percentiles, confidence_intervals)
    
    def get_statistics(self, forecast_name: str) -> Dict[str, float]:
        """Get descriptive statistics for a forecast"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        return self.results[forecast_name].statistics
    
    def get_percentiles(self, forecast_name: str) -> Dict[str, float]:
        """Get percentiles for a forecast"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        return self.results[forecast_name].percentiles
    
    def sensitivity_analysis(self, forecast_name: str) -> Dict[str, pd.DataFrame]:
        """Perform sensitivity analysis for a forecast"""
        if forecast_name not in self.forecasts:
            raise ValueError(f"Forecast '{forecast_name}' not found")
        
        forecast = self.forecasts[forecast_name]
        
        return {
            'correlations': SensitivityAnalyzer.rank_correlations(self.assumptions, forecast),
            'variance_contributions': SensitivityAnalyzer.contribution_to_variance(self.assumptions, forecast)
        }
    
    def plot_forecast_histogram(self, forecast_name: str, bins: int = 50, 
                              figsize: tuple = (10, 6), show_stats: bool = True):
        """Plot histogram of forecast results and return as base64 string"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        
        results = self.results[forecast_name]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(results.values, bins=bins, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        
        # Add mean line
        mean_val = results.statistics['mean']
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        
        # Add percentile lines
        p5 = results.percentiles['p5']
        p95 = results.percentiles['p95']
        ax.axvline(p5, color='orange', linestyle='--', alpha=0.7, 
                   label=f'5th Percentile: {p5:.2f}')
        ax.axvline(p95, color='orange', linestyle='--', alpha=0.7, 
                   label=f'95th Percentile: {p95:.2f}')
        
        ax.set_title(f'Monte Carlo Forecast: {forecast_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show_stats:
            stats_text = f"""Statistics:
Mean: {results.statistics['mean']:.3f}
Std Dev: {results.statistics['std']:.3f}
Min: {results.statistics['min']:.3f}
Max: {results.statistics['max']:.3f}
Trials: {len(results.values)}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def plot_sensitivity_chart(self, forecast_name: str, figsize: tuple = (10, 6)):
        """Plot sensitivity analysis chart and return as base64 string"""
        sensitivity = self.sensitivity_analysis(forecast_name)
        correlations = sensitivity['correlations'].head(10)  # Top 10
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = range(len(correlations))
        bars = ax.barh(y_pos, correlations['Correlation'], 
                       color=['green' if x > 0 else 'red' for x in correlations['Correlation']])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(correlations['Assumption'])
        ax.set_xlabel('Rank Correlation')
        ax.set_title(f'Sensitivity Analysis: {forecast_name}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left' if width > 0 else 'right', va='center')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_report(self, forecast_name: str) -> str:
        """Generate a comprehensive text report"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        
        results = self.results[forecast_name]
        sensitivity = self.sensitivity_analysis(forecast_name)
        
        report = f"""
MONTE CARLO SIMULATION REPORT
============================
Forecast: {forecast_name}
Trials: {len(results.values)}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
------------------
Mean:           {results.statistics['mean']:.6f}
Standard Dev:   {results.statistics['std']:.6f}
Variance:       {results.statistics['variance']:.6f}
Minimum:        {results.statistics['min']:.6f}
Maximum:        {results.statistics['max']:.6f}
Median:         {results.statistics['median']:.6f}
Skewness:       {results.statistics['skewness']:.6f}
Kurtosis:       {results.statistics['kurtosis']:.6f}

PERCENTILES
-----------
1st:    {results.percentiles['p1']:.6f}
5th:    {results.percentiles['p5']:.6f}
10th:   {results.percentiles['p10']:.6f}
25th:   {results.percentiles['p25']:.6f}
50th:   {results.percentiles['p50']:.6f}
75th:   {results.percentiles['p75']:.6f}
90th:   {results.percentiles['p90']:.6f}
95th:   {results.percentiles['p95']:.6f}
99th:   {results.percentiles['p99']:.6f}

CONFIDENCE INTERVALS
--------------------
90% CI: [{results.confidence_intervals['90%'][0]:.6f}, {results.confidence_intervals['90%'][1]:.6f}]
95% CI: [{results.confidence_intervals['95%'][0]:.6f}, {results.confidence_intervals['95%'][1]:.6f}]
99% CI: [{results.confidence_intervals['99%'][0]:.6f}, {results.confidence_intervals['99%'][1]:.6f}]

SENSITIVITY ANALYSIS - TOP 5 CORRELATIONS
------------------------------------------
"""
        
        top_correlations = sensitivity['correlations'].head(5)
        for _, row in top_correlations.iterrows():
            report += f"{row['Assumption']:20} {row['Correlation']:8.4f}\n"
        
        report += """
ASSUMPTIONS
-----------
"""
        for name, assumption in self.assumptions.items():
            params = assumption.distribution.get_parameters()
            report += f"{name:20} {assumption.distribution.name:15} {str(params)}\n"
        
        return report
    
    def export_results(self, filename: str, format: str = 'excel'):
        """Export results to Excel or CSV"""
        if format.lower() == 'excel':
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for forecast_name, results in self.results.items():
                    summary_data.append({
                        'Forecast': forecast_name,
                        'Mean': results.statistics['mean'],
                        'Std_Dev': results.statistics['std'],
                        'Min': results.statistics['min'],
                        'Max': results.statistics['max'],
                        'P5': results.percentiles['p5'],
                        'P95': results.percentiles['p95']
                    })
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual forecast sheets
                for forecast_name, results in self.results.items():
                    df = pd.DataFrame({
                        'Trial': range(1, len(results.values) + 1),
                        'Value': results.values
                    })
                    df.to_excel(writer, sheet_name=f'{forecast_name}', index=False)
        
        elif format.lower() == 'csv':
            # Export summary to CSV
            summary_data = []
            for forecast_name, results in self.results.items():
                summary_data.append({
                    'Forecast': forecast_name,
                    'Mean': results.statistics['mean'],
                    'Std_Dev': results.statistics['std'],
                    'Min': results.statistics['min'],
                    'Max': results.statistics['max'],
                    'P5': results.percentiles['p5'],
                    'P95': results.percentiles['p95']
                })
            
            pd.DataFrame(summary_data).to_csv(filename, index=False)

# Factory class for creating distributions
class DistributionFactory:
    """Factory class for creating probability distributions"""
    
    @staticmethod
    def create_normal(mean: float, std: float) -> NormalDistribution:
        return NormalDistribution(mean, std)
    
    @staticmethod
    def create_uniform(min_val: float, max_val: float) -> UniformDistribution:
        return UniformDistribution(min_val, max_val)
    
    @staticmethod
    def create_triangular(min_val: float, mode: float, max_val: float) -> TriangularDistribution:
        return TriangularDistribution(min_val, mode, max_val)
    
    @staticmethod
    def create_lognormal(mu: float, sigma: float) -> LogNormalDistribution:
        return LogNormalDistribution(mu, sigma)

# ==================== API SERVICE LAYER ====================

# Pydantic models for API requests and responses
class DistributionConfig(BaseModel):
    """Configuration for probability distributions"""
    type: str = Field(..., description="Distribution type: normal, uniform, triangular, lognormal")
    parameters: Dict[str, float] = Field(..., description="Distribution parameters")

class AssumptionConfig(BaseModel):
    """Configuration for model assumptions"""
    name: str = Field(..., description="Name of the assumption")
    description: Optional[str] = Field("", description="Description of the assumption")
    distribution: DistributionConfig = Field(..., description="Probability distribution")

class FormulaConfig(BaseModel):
    """Configuration for forecast formulas"""
    expression: str = Field(..., description="Mathematical expression using assumption names")
    variables: List[str] = Field(..., description="List of variable names used in expression")

class ForecastConfig(BaseModel):
    """Configuration for model forecasts"""
    name: str = Field(..., description="Name of the forecast")
    description: Optional[str] = Field("", description="Description of the forecast")
    formula: FormulaConfig = Field(..., description="Formula configuration")

class SimulationConfig(BaseModel):
    """Complete simulation configuration"""
    assumptions: List[AssumptionConfig] = Field(..., description="List of model assumptions")
    forecasts: List[ForecastConfig] = Field(..., description="List of model forecasts")
    trials: int = Field(10000, description="Number of Monte Carlo trials", ge=100, le=100000)
    seed: Optional[int] = Field(None, description="Random seed for reproducible results")

class SimulationResponse(BaseModel):
    """Response containing simulation results"""
    simulation_id: str
    status: str
    timestamp: str
    config: SimulationConfig
    results: Optional[Dict[str, Any]] = None
    charts: Optional[Dict[str, str]] = None  # Base64 encoded images
    report: Optional[str] = None
    error: Optional[str] = None

# Global storage for simulation instances (in production, use Redis or database)
simulation_cache: Dict[str, CrystalBallMVP] = {}

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Crystal Ball API starting up...")
    yield
    # Shutdown
    print("Crystal Ball API shutting down...")

app = FastAPI(
    title="Crystal Ball MVP API",
    description="Monte Carlo Simulation API for risk analysis and forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for Google Sheets integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def create_distribution(dist_config: DistributionConfig) -> ProbabilityDistribution:
    """Create a probability distribution from configuration"""
    dist_type = dist_config.type.lower()
    params = dist_config.parameters
    
    if dist_type == "normal":
        return DistributionFactory.create_normal(params["mean"], params["std"])
    elif dist_type == "uniform":
        return DistributionFactory.create_uniform(params["min"], params["max"])
    elif dist_type == "triangular":
        return DistributionFactory.create_triangular(params["min"], params["mode"], params["max"])
    elif dist_type == "lognormal":
        return DistributionFactory.create_lognormal(params["mu"], params["sigma"])
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def create_formula_function(formula_config: FormulaConfig) -> Callable:
    """Create a callable function from formula configuration"""
    expression = formula_config.expression
    variables = formula_config.variables
    
    def formula_func(assumptions: Dict[str, np.ndarray]) -> np.ndarray:
        # Create a safe evaluation environment
        allowed_names = {
            "np": np,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "pow": pow,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
        }
        
        # Add assumption variables
        for var in variables:
            if var in assumptions:
                allowed_names[var] = assumptions[var]
            else:
                raise KeyError(f"Variable '{var}' not found in assumptions")
        
        try:
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return np.array(result)
        except Exception as e:
            raise ValueError(f"Error evaluating formula '{expression}': {str(e)}")
    
    return formula_func

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Crystal Ball MVP API",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_simulations": len(simulation_cache)
    }

@app.post("/simulation/create", response_model=SimulationResponse)
async def create_simulation(config: SimulationConfig):
    """Create and run a Monte Carlo simulation"""
    simulation_id = str(uuid.uuid4())
    
    try:
        # Create Crystal Ball instance
        cb = CrystalBallMVP()
        cb.set_trials(config.trials)
        
        if config.seed is not None:
            cb.set_seed(config.seed)
        
        # Add assumptions
        for assumption_config in config.assumptions:
            distribution = create_distribution(assumption_config.distribution)
            assumption = Assumption(
                name=assumption_config.name,
                distribution=distribution,
                description=assumption_config.description
            )
            cb.add_assumption(assumption)
        
        # Add forecasts
        for forecast_config in config.forecasts:
            formula_func = create_formula_function(forecast_config.formula)
            forecast = Forecast(
                name=forecast_config.name,
                formula=formula_func,
                description=forecast_config.description
            )
            cb.add_forecast(forecast)
        
        # Run simulation
        cb.run_simulation()
        
        # Store simulation in cache
        simulation_cache[simulation_id] = cb
        
        # Prepare results
        results = {}
        charts = {}
        
        for forecast_name in cb.forecasts.keys():
            # Get statistics and percentiles
            results[forecast_name] = {
                "statistics": cb.get_statistics(forecast_name),
                "percentiles": cb.get_percentiles(forecast_name),
                "sensitivity": {
                    "correlations": cb.sensitivity_analysis(forecast_name)["correlations"].to_dict("records"),
                    "variance_contributions": cb.sensitivity_analysis(forecast_name)["variance_contributions"].to_dict("records")
                }
            }
            
            # Generate charts as base64 strings
            charts[f"{forecast_name}_histogram"] = cb.plot_forecast_histogram(forecast_name)
            charts[f"{forecast_name}_sensitivity"] = cb.plot_sensitivity_chart(forecast_name)
        
        # Generate report for first forecast
        first_forecast = list(cb.forecasts.keys())[0]
        report = cb.generate_report(first_forecast)
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="completed",
            timestamp=datetime.utcnow().isoformat(),
            config=config,
            results=results,
            charts=charts,
            report=report
        )
        
    except Exception as e:
        return SimulationResponse(
            simulation_id=simulation_id,
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            config=config,
            error=str(e)
        )

@app.get("/simulation/{simulation_id}/results")
async def get_simulation_results(simulation_id: str):
    """Get results for a specific simulation"""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    cb = simulation_cache[simulation_id]
    
    # Prepare results
    results = {}
    for forecast_name in cb.forecasts.keys():
        results[forecast_name] = {
            "statistics": cb.get_statistics(forecast_name),
            "percentiles": cb.get_percentiles(forecast_name),
            "sensitivity": {
                "correlations": cb.sensitivity_analysis(forecast_name)["correlations"].to_dict("records"),
                "variance_contributions": cb.sensitivity_analysis(forecast_name)["variance_contributions"].to_dict("records")
            }
        }
    
    return {"simulation_id": simulation_id, "results": results}

@app.get("/simulation/{simulation_id}/charts/{forecast_name}")
async def get_forecast_charts(simulation_id: str, forecast_name: str, chart_type: str = "histogram"):
    """Get charts for a specific forecast"""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    cb = simulation_cache[simulation_id]
    
    if forecast_name not in cb.forecasts:
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    try:
        if chart_type == "histogram":
            chart_base64 = cb.plot_forecast_histogram(forecast_name)
        elif chart_type == "sensitivity":
            chart_base64 = cb.plot_sensitivity_chart(forecast_name)
        else:
            raise HTTPException(status_code=400, detail="Invalid chart type. Use 'histogram' or 'sensitivity'")
        
        return {"chart": chart_base64, "type": chart_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/{simulation_id}/report/{forecast_name}")
async def get_forecast_report(simulation_id: str, forecast_name: str):
    """Get text report for a specific forecast"""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    cb = simulation_cache[simulation_id]
    
    if forecast_name not in cb.forecasts:
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    try:
        report = cb.generate_report(forecast_name)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/{simulation_id}/export")
async def export_simulation_results(simulation_id: str, format: str = "excel"):
    """Export simulation results to file"""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    cb = simulation_cache[simulation_id]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as tmp_file:
        filename = tmp_file.name
    
    try:
        cb.export_results(filename, format)
        return FileResponse(
            filename,
            media_type='application/octet-stream',
            filename=f"crystal_ball_results_{simulation_id[:8]}.{format}"
        )
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(filename):
            os.unlink(filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Schedule cleanup
        def cleanup():
            if os.path.exists(filename):
                os.unlink(filename)
        # In a real app, you'd use a background task or scheduled job
        import threading
        threading.Timer(60.0, cleanup).start()

@app.delete("/simulation/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Delete a simulation from cache"""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulation_cache[simulation_id]
    return {"message": f"Simulation {simulation_id} deleted successfully"}

@app.get("/simulations")
async def list_simulations():
    """List all active simulations"""
    simulations = []
    for sim_id, cb in simulation_cache.items():
        simulations.append({
            "simulation_id": sim_id,
            "forecasts": list(cb.forecasts.keys()),
            "assumptions": list(cb.assumptions.keys()),
            "trials": cb.trials
        })
    
    return {"simulations": simulations, "count": len(simulations)}

# Google Sheets friendly endpoints
@app.post("/sheets/simple-simulation")
async def simple_simulation_for_sheets(
    assumptions_data: List[Dict[str, Any]],
    forecast_formula: str,
    forecast_variables: List[str],
    trials: int = 10000,
    seed: Optional[int] = None
):
    """Simplified endpoint optimized for Google Sheets integration"""
    try:
        # Build configuration from simplified inputs
        assumptions_config = []
        for assumption in assumptions_data:
            dist_config = DistributionConfig(
                type=assumption["distribution_type"],
                parameters=assumption["parameters"]
            )
            assumptions_config.append(AssumptionConfig(
                name=assumption["name"],
                description=assumption.get("description", ""),
                distribution=dist_config
            ))
        
        forecast_config = ForecastConfig(
            name="Primary_Forecast",
            description="Main forecast output",
            formula=FormulaConfig(
                expression=forecast_formula,
                variables=forecast_variables
            )
        )
        
        config = SimulationConfig(
            assumptions=assumptions_config,
            forecasts=[forecast_config],
            trials=trials,
            seed=seed
        )
        
        # Run simulation using existing endpoint logic
        response = await create_simulation(config)
        
        # Return simplified results for Google Sheets
        if response.status == "completed" and response.results:
            forecast_results = response.results["Primary_Forecast"]
            return {
                "status": "success",
                "simulation_id": response.simulation_id,
                "statistics": forecast_results["statistics"],
                "percentiles": forecast_results["percentiles"],
                "sensitivity": forecast_results["sensitivity"]["correlations"][:5]  # Top 5
            }
        else:
            return {"status": "error", "error": response.error}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/sheets/results/{simulation_id}")
async def get_sheets_results(simulation_id: str):
    """Get results in a format optimized for Google Sheets"""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    cb = simulation_cache[simulation_id]
    
    # Format results for easy import into Google Sheets
    results = []
    
    for forecast_name in cb.forecasts.keys():
        stats = cb.get_statistics(forecast_name)
        percentiles = cb.get_percentiles(forecast_name)
        
        # Create flat structure for sheets
        row = {
            "Forecast": forecast_name,
            "Mean": stats["mean"],
            "Std_Dev": stats["std"],
            "Min": stats["min"],
            "Max": stats["max"],
            "P5": percentiles["p5"],
            "P25": percentiles["p25"],
            "P50": percentiles["p50"],
            "P75": percentiles["p75"],
            "P95": percentiles["p95"]
        }
        results.append(row)
    
    return {"results": results}

# Health and diagnostics
@app.get("/diagnostics")
async def get_diagnostics():
    """Get system diagnostics"""
    import psutil
    import sys
    
    return {
        "system": {
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "application": {
            "active_simulations": len(simulation_cache),
            "cache_memory_usage": sum(sys.getsizeof(cb) for cb in simulation_cache.values())
        }
    }

# Example usage and testing endpoint
@app.get("/example")
async def get_example_config():
    """Get an example configuration for testing"""
    example_config = {
        "assumptions": [
            {
                "name": "Price",
                "description": "Product price per unit",
                "distribution": {
                    "type": "triangular",
                    "parameters": {"min": 90, "mode": 100, "max": 120}
                }
            },
            {
                "name": "Quantity",
                "description": "Expected quantity sold",
                "distribution": {
                    "type": "normal",
                    "parameters": {"mean": 1000, "std": 150}
                }
            },
            {
                "name": "Market_Share",
                "description": "Market share percentage",
                "distribution": {
                    "type": "uniform",
                    "parameters": {"min": 0.15, "max": 0.25}
                }
            }
        ],
        "forecasts": [
            {
                "name": "Revenue",
                "description": "Total revenue forecast",
                "formula": {
                    "expression": "Price * Quantity * Market_Share",
                    "variables": ["Price", "Quantity", "Market_Share"]
                }
            }
        ],
        "trials": 10000,
        "seed": 42
    }
    
    return {"example_config": example_config}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=False
    )