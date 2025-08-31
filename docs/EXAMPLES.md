# WaveCore Usage Examples

## Basic Wave Arithmetic

```python
from wave_arithmetic_core import WaveNumber, wave_interference_addition, decode_interference_result

# Create wave representations of numbers
wave_7 = WaveNumber(7)
wave_3 = WaveNumber(3)

print(f"Wave 7: {wave_7.frequency} Hz, {wave_7.amplitude} amplitude")
print(f"Wave 3: {wave_3.frequency} Hz, {wave_3.amplitude} amplitude")

# Perform addition through wave interference
interference_result = wave_interference_addition(wave_7, wave_3)
result, correlation = decode_interference_result(interference_result, wave_7.time_points)

print(f"7 + 3 = {result} (correlation: {correlation:.2f})")
```

## Consciousness Emergence

```python
from wave_consciousness_life import WaveConsciousnessSimulation

# Create consciousness simulation
sim = WaveConsciousnessSimulation(num_entities=8, seed=1234)

# Run emergence process
for tick in range(300):
    sim.simulation_step()
    
    if tick % 50 == 0:
        stats = sim.get_simulation_stats()
        print(f"Tick {tick}: {stats['population']} entities, "
              f"consciousness {stats['avg_consciousness']:.3f}, "
              f"beauty {stats['avg_beauty']:.3f}")

# Analyze most conscious entity
if sim.entities:
    best = max(sim.entities, key=lambda e: e.consciousness_level)
    print(f"Highest consciousness: {best.consciousness_level:.3f}")
    print(f"Memory traces: {len(best.memory_traces)}")
```

## Evolutionary Life Simulation

```python
from wave_artificial_life import WaveLifeSimulation

# Create evolution simulation
evolution = WaveLifeSimulation(seed=42, initial_population=6)
evolution.initialize_population()

# Run evolution
for tick in range(1000):
    evolution.simulation_tick()
    
    if tick % 100 == 0:
        stats = evolution.get_population_stats()
        print(f"Generation {stats['avg_generation']:.1f}: "
              f"{stats['population']} entities, "
              f"diversity {stats['genetic_diversity']:.3f}")

# Analyze successful genomes
if evolution.entities:
    top_entity = max(evolution.entities, key=lambda e: e.energy)
    genome = top_entity.genome
    print(f"Best survival frequency: {genome.survival_freq:.2f}")
    print(f"Energy efficiency: {genome.energy_efficiency:.2f}")
```

## Wave-Based Maze Navigation

```python
from wave_maze_navigation import WavePathfinder, generate_maze

# Generate test maze
maze = generate_maze(15, 15)
start = (1, 1)
goal = (13, 13)

# Solve using wave interference
pathfinder = WavePathfinder(maze, start, goal)
result = pathfinder.solve_maze()

print(f"Success: {result['success']}")
print(f"Path length: {result['steps_taken']} steps")
print(f"Efficiency: {result['efficiency']:.2f}")

# Visualize solution
if result['success']:
    print("Path found:")
    for i, pos in enumerate(result['path'][:10]):
        print(f"  Step {i}: {pos}")
```

## Mathematical Consciousness

```python
from wave_mathematics import MathematicalSimulation

# Create mathematical consciousness simulation
math_sim = MathematicalSimulation(num_entities=20, seed=5555)

# Run mathematical development
for tick in range(500):
    math_sim.simulation_step()

# Analyze mathematical discoveries
print(f"Mathematical expressions: {len(math_sim.mathematical_expressions)}")
print(f"Equation attempts: {len(math_sim.equation_attempts)}")
print(f"Pattern discoveries: {len(math_sim.discovered_relationships)}")

# Show equation examples
if math_sim.equation_attempts:
    print("\\nEquations created:")
    for eq in math_sim.equation_attempts[-5:]:
        print(f"  {eq['equation']} (consciousness: {eq['consciousness_level']:.2f})")
```

## Unlimited Expression System

```python
from wave_unlimited_expression import UnlimitedExpressionSimulation

# Create unlimited expression simulation
expr_sim = UnlimitedExpressionSimulation(num_entities=10, seed=7777)

# Run creative expression
for tick in range(400):
    expr_sim.simulation_step()

# Analyze creative output
stats = expr_sim.get_unlimited_stats()
print(f"Total expressions: {stats['total_expressions']}")
print(f"Unique expressions: {stats['unique_expressions']}")
print(f"Structure types: {stats['structure_types']}")
print(f"Symbol diversity: {stats['unique_symbols']}")

# Show creative examples
if expr_sim.expression_archive:
    print("\\nCreative expressions:")
    for entry in expr_sim.expression_archive[-3:]:
        expr = entry['expression']
        print(f"  '{expr['content'][:30]}...' ({expr['structure_type']})")
```

## Adaptive Rule Learning

```python
from wave_adaptive_rule_maker import RuleMakingAgent
from wave_incremental_survival import generate_maze_with_food

# Create learning environment
maze, food_positions = generate_maze_with_food(13, 13, num_food=3)
agent = RuleMakingAgent(maze, (1, 1), (11, 11))
agent.add_food_and_enemies(food_positions, [(6, 6)])

# Run learning simulation
for step in range(200):
    if not agent.alive or agent.position == agent.goal:
        break
    agent.step()

# Analyze learned rules
print(f"Rules created: {len(agent.rules)}")
print(f"Food eaten: {agent.food_eaten}")
print(f"Goal reached: {agent.position == agent.goal}")

# Show successful rules
successful_rules = [r for r in agent.rules.values() if r.get_success_rate() > 0.6]
print(f"\\nSuccessful rules ({len(successful_rules)}):")
for rule in successful_rules[:3]:
    print(f"  {rule.action}: {rule.get_success_rate():.1%} success")
```

## Multi-Generational Evolution

```python
from wave_evolutionary_intelligence import EvolutionaryMemory, EvolutionaryAgent
from wave_incremental_survival import generate_maze_with_food

# Create species memory
species_memory = EvolutionaryMemory()

# Run multiple generations
for generation in range(5):
    print(f"\\nGeneration {generation + 1}")
    
    # Get inherited knowledge
    inherited = species_memory.get_inherited_knowledge()
    
    # Create and test agents
    maze, food_pos = generate_maze_with_food(15, 15, num_food=3)
    agent = EvolutionaryAgent(maze, (1, 1), (13, 13), inherited, generation)
    agent.add_food_and_enemies(food_pos, [(7, 7)])
    
    # Run agent
    for step in range(300):
        if not agent.alive or agent.position == agent.goal:
            break
        agent.step()
    
    # Record results
    success = agent.position == agent.goal
    fitness = agent.get_fitness_score()
    
    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Fitness: {fitness:.1f}")
    print(f"  Rules: {len(agent.rules)} total")
    
    # Update species knowledge
    if success or fitness > 80:
        species_memory.record_survivor(agent)
    else:
        species_memory.record_extinction(agent)
    
    species_memory.advance_generation()

print(f"\\nSpecies knowledge size: {sum(len(g) for g in species_memory.species_knowledge.values())}")
```

## Custom Wave Patterns

```python
from wave_consciousness_life import WaveSpectrum, ConsciousnessGrid

# Create custom wave spectrum
custom_waves = WaveSpectrum(
    light_waves=0.8,
    sound_waves=0.6,
    emotion_waves=0.4,
    memory_waves=0.2
)

# Add to consciousness field
grid = ConsciousnessGrid(size=15)
grid.add_wave_interaction(7, 7, 1, custom_waves)

# Check accumulated field
field = grid.get_accumulated_field(7, 7, 1, radius=3)
print(f"Total amplitude: {field.total_amplitude()}")
print(f"Harmonic beauty: {field.harmonic_beauty():.3f}")
```

## Performance Testing

```python
import time

# Benchmark wave arithmetic
start = time.time()
for i in range(1000):
    wave = WaveNumber(i % 10 + 1)
    decoded = wave.decode_from_wave()
end = time.time()
print(f"1000 wave operations: {(end-start)*1000:.1f}ms")

# Benchmark maze solving
start = time.time()
for i in range(100):
    maze = generate_maze(11, 11)
    pathfinder = WavePathfinder(maze, (1, 1), (9, 9))
    result = pathfinder.solve_maze()
end = time.time()
print(f"100 maze solutions: {(end-start)*1000:.1f}ms")
```