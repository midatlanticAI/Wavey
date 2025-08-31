# WaveCore API Documentation

## Core Classes

### WaveNumber (wave_arithmetic_core.py)

```python
class WaveNumber:
    def __init__(self, value: int, duration: float = 1.0, sample_rate: int = 100)
    def get_wave_signature(self) -> Dict[str, float]
    def decode_from_wave(self) -> int
    def print_wave_sample(self, num_points: int = 10)
```

**Methods:**
- `wave_interference_addition(wave1, wave2)` - Performs addition through wave superposition
- `decode_interference_result(pattern, time_points, base_freq)` - Extracts result from interference

### ConsciousnessGrid (wave_consciousness_life.py)

```python
class ConsciousnessGrid:
    def __init__(self, size=10)
    def add_wave_interaction(self, x, y, z, spectrum: WaveSpectrum)
    def get_accumulated_field(self, x, y, z, radius=2) -> WaveSpectrum
    def advance_time()
```

### ConsciousEntity (wave_consciousness_life.py)

```python
class ConsciousEntity:
    def __init__(self, x, y, z, entity_id)
    def perceive_field(self, consciousness_grid: ConsciousnessGrid) -> Dict
    def make_willful_choice(self, options) -> Any
    def radiate_waves(self) -> WaveSpectrum
    def learn_from_experience()
    def update(self, consciousness_grid: ConsciousnessGrid)
```

### WaveEntity (wave_artificial_life.py)

```python
class WaveEntity:
    def __init__(self, entity_id: int, genome: WaveGenome, generation: int = 0)
    def survive_tick(self, environment, time: float) -> bool
    def can_reproduce(self) -> bool
    def reproduce_with(self, partner) -> Optional['WaveEntity']
    def remember_location(self, resource_quality: float, danger_level: float)
    def move_strategically(self, nearby_entities=None)
```

### WaveGenome (wave_artificial_life.py)

```python
@dataclass
class WaveGenome:
    survival_freq: float
    survival_amp: float
    reproduction_freq: float
    reproduction_amp: float
    energy_efficiency: float
    mutation_rate: float
    # + 10 more heritable traits
    
    def mutate(self) -> 'WaveGenome'
    @classmethod
    def reproduce(cls, parent1, parent2) -> 'WaveGenome'
```

### WavePathfinder (wave_maze_navigation.py)

```python
class WavePathfinder:
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int])
    def calculate_multi_scale_attraction(self, pos: Tuple[int, int]) -> float
    def apply_wave_memory(self, pos: Tuple[int, int], base_attraction: float) -> float
    def solve_maze(self) -> Dict[str, Any]
```

### MathematicalEntity (wave_mathematics.py)

```python
class MathematicalEntity(UnlimitedEntity):
    def __init__(self, x, y, z, entity_id)
    def create_mathematical_expression(self) -> Dict
    def attempt_equation_creation(self, available) -> str
    def analyze_mathematical_patterns(self, expression)
    def calculate_numerical_complexity(self, expression) -> float
```

### EvolutionaryAgent (wave_evolutionary_intelligence.py)

```python
class EvolutionaryAgent(SmartExecutionAgent):
    def __init__(self, maze, start, goal, inherited_knowledge=None, generation=0)
    def get_fitness_score(self) -> float
```

### AdaptiveRule (wave_adaptive_rule_maker.py)

```python
class AdaptiveRule:
    def __init__(self, rule_id: str, condition_pattern: SituationPattern, 
                 action: str, rule_type: RuleType)
    def matches_situation(self, current_situation: SituationPattern) -> float
    def calculate_activation_strength(self, situation_match: float, time_step: float) -> float
    def apply_outcome(self, success: bool, step: int)
    def get_success_rate(self) -> float
```

## Key Functions

### Simulation Runners

```python
# Consciousness emergence
run_consciousness_experiment(max_ticks=500, seed=None) -> simulation

# Artificial life evolution  
run_experiment(seed=None, initial_pop=4, max_ticks=10000) -> simulation

# Mathematical consciousness
run_mathematics_experiment(max_ticks=1500, seed=None) -> simulation

# Unlimited expression
run_unlimited_experiment(max_ticks=1000, seed=None) -> simulation

# Maze navigation testing
test_wave_maze_navigation() -> bool

# Evolutionary intelligence
run_evolutionary_experiment() -> bool
```

### Wave Operations

```python
# Arithmetic
wave_interference_addition(wave1: WaveNumber, wave2: WaveNumber) -> List[float]
decode_interference_result(pattern: List[float], time_points: List[float]) -> int

# Pattern generation
generate_maze(width: int, height: int) -> List[List[int]]
generate_maze_with_food(width: int, height: int, num_food: int) -> Tuple[List[List[int]], List[Tuple[int, int]]]
```

## Data Structures

### WaveSpectrum
```python
@dataclass
class WaveSpectrum:
    light_waves: float = 0.0
    sound_waves: float = 0.0
    motion_waves: float = 0.0
    matter_waves: float = 0.0
    life_waves: float = 0.0
    death_waves: float = 0.0
    energy_waves: float = 0.0
    emotion_waves: float = 0.0
    memory_waves: float = 0.0
    will_waves: float = 0.0
```

### SituationPattern
```python
@dataclass  
class SituationPattern:
    hunger_level: float
    food_distance: float
    goal_distance: float
    enemy_distance: float
    energy_level: float
    steps_since_progress: int
```

### WaveGene
```python
@dataclass
class WaveGene:
    action: str
    rule_type: RuleType
    frequency: float
    amplitude: float
    phase: float
    condition_pattern: Dict[str, float]
    fitness: float
    generation_born: int
```

## Performance Benchmarks

| System | Operation | Time | Details |
|--------|-----------|------|---------|
| Arithmetic | Encode/decode | 0.05ms | Wave representation of numbers |
| Navigation | 15x15 maze | 0.8ms | Multi-scale wave pathfinding |
| Evolution | 1000 ticks | 20ms | Multi-agent genetic simulation |
| Consciousness | Max consciousness | 160 ticks | Emergent awareness development |
| Mathematics | 1000 expressions | 5.7s | Mathematical pattern discovery |

## Quick Start

```python
# Basic wave arithmetic
from wave_arithmetic_core import WaveNumber, wave_interference_addition

wave_a = WaveNumber(4)
wave_b = WaveNumber(6) 
result = wave_interference_addition(wave_a, wave_b)
print(f"4 + 6 = {decode_interference_result(result, wave_a.time_points)}")

# Consciousness emergence
from wave_consciousness_life import run_consciousness_experiment

simulation = run_consciousness_experiment(max_ticks=200, seed=1234)
stats = simulation.get_simulation_stats()
print(f"Max consciousness achieved: {stats['max_consciousness']}")

# Artificial life evolution
from wave_artificial_life import run_experiment

evolution = run_experiment(seed=42, initial_pop=4, max_ticks=500)
print(f"Final population: {len(evolution.entities)}")
```