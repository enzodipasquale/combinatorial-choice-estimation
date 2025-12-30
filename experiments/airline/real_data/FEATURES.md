# Route Features for Real Airline Data

## Feature Structure

We use **2 features** for each route (origin-destination pair):

### Feature 1: Population-Weighted Centroid (Modular)

This feature captures the attractiveness of a route based on:
- **Population product**: `pop_origin × pop_dest`
- **Distance**: Great circle distance between cities
- **Softmax normalization**: Over all destinations from the same origin

**Formula:**
```
feature(origin → dest) = exp((pop_origin × pop_dest) / distance) / 
                         Σ_{dest'} exp((pop_origin × pop_dest') / distance)
```

This creates a probability distribution over destinations from each origin, where:
- Routes to larger cities (higher population) are more attractive
- Shorter routes are more attractive
- The softmax ensures probabilities sum to 1 for each origin

**Implementation details:**
- Populations are scaled to millions to avoid numerical overflow
- Uses log-sum-exp trick for numerical stability
- Temperature parameter controls the sharpness of the distribution

### Feature 2: Congestion Cost (Quadratic)

This is the **congestion term** that penalizes airlines for having too many routes from the same hub:

```
congestion = -θ^gs × Σ_{h ∈ hubs} |{routes from hub h}|²
```

This is computed as part of the utility function, not as a route-level feature. It's handled separately in the feature oracle.

## Utility Function

For an airline with hubs H and selected bundle B:

```
U(B) = Σ_{j ∈ B} x_j^T θ^mod - θ^gs × Σ_{h ∈ H} |{ab ∈ B: a = h}|²
```

Where:
- `x_j^T θ^mod` = population-weighted feature × θ₁
- Second term = congestion cost (quadratic in routes from hubs)

## Theta Parameters

- `θ₁`: Weight on population-weighted centroid feature (positive = prefer high-pop, short-distance routes)
- `θ_gs`: Gross substitutability parameter (positive = congestion cost, creates hub-and-spoke patterns)

## Usage

The features are automatically computed when using `integrate_real_data.py`:

```python
from integrate_real_data import create_real_airline_scenario
from load_real_data import RealAirlineDataLoader

loader = RealAirlineDataLoader()
loader.load_cities_from_csv('cities.csv')
processed = loader.process_for_bundlechoice()

scenario = create_real_airline_scenario(
    data_loader=loader,
    num_agents=5,
    theta_gs=0.2,  # Congestion cost parameter
    temperature=1.0,  # Softmax temperature
    seed=42
)
```

The scenario will automatically:
1. Compute population-weighted features for all routes
2. Set up the congestion cost structure
3. Use real airline hub locations (if provided)


