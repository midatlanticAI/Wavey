#!/usr/bin/env python3
"""
Wave-Based Artificial Life Simulation
Start with 2 entities, let evolution do the work
"""

import math
import random
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy

@dataclass
class WaveGenome:
    """Wave-based genetic code"""
    survival_freq: float
    survival_amp: float
    reproduction_freq: float
    reproduction_amp: float
    energy_efficiency: float
    mutation_rate: float
    # Environmental sensing capabilities
    temperature_sensitivity: float  # How well they sense temperature changes
    resource_detection: float       # How well they find food/resources
    danger_awareness: float         # How well they sense threats
    # Memory and movement capabilities
    memory_capacity: float          # How many locations they can remember
    movement_speed: float           # How fast they can move
    # Communication capabilities
    communication_range: float      # How far they can send/receive signals
    social_learning: float          # How much they learn from others
    # Generational knowledge transfer
    teaching_willingness: float     # How much energy parents spend teaching
    learning_from_parents: float    # How much offspring trust parent knowledge
    
    def mutate(self):
        """Apply random mutations"""
        mutation_strength = self.mutation_rate
        return WaveGenome(
            survival_freq=max(0.1, self.survival_freq * random.uniform(1-mutation_strength, 1+mutation_strength)),
            survival_amp=max(0.1, self.survival_amp * random.uniform(1-mutation_strength, 1+mutation_strength)),
            reproduction_freq=max(0.1, self.reproduction_freq * random.uniform(1-mutation_strength, 1+mutation_strength)),
            reproduction_amp=max(0.1, self.reproduction_amp * random.uniform(1-mutation_strength, 1+mutation_strength)),
            energy_efficiency=max(0.1, min(2.0, self.energy_efficiency * random.uniform(1-mutation_strength, 1+mutation_strength))),
            mutation_rate=max(0.01, min(0.3, self.mutation_rate * random.uniform(0.9, 1.1))),
            # Mutate sensing abilities
            temperature_sensitivity=max(0.1, min(2.0, self.temperature_sensitivity * random.uniform(1-mutation_strength, 1+mutation_strength))),
            resource_detection=max(0.1, min(2.0, self.resource_detection * random.uniform(1-mutation_strength, 1+mutation_strength))),
            danger_awareness=max(0.1, min(2.0, self.danger_awareness * random.uniform(1-mutation_strength, 1+mutation_strength))),
            # Mutate memory and movement
            memory_capacity=max(0.1, min(3.0, self.memory_capacity * random.uniform(1-mutation_strength, 1+mutation_strength))),
            movement_speed=max(0.1, min(2.0, self.movement_speed * random.uniform(1-mutation_strength, 1+mutation_strength))),
            # Mutate communication
            communication_range=max(0.1, min(2.0, self.communication_range * random.uniform(1-mutation_strength, 1+mutation_strength))),
            social_learning=max(0.1, min(2.0, self.social_learning * random.uniform(1-mutation_strength, 1+mutation_strength))),
            # Mutate teaching
            teaching_willingness=max(0.1, min(2.0, self.teaching_willingness * random.uniform(1-mutation_strength, 1+mutation_strength))),
            learning_from_parents=max(0.1, min(2.0, self.learning_from_parents * random.uniform(1-mutation_strength, 1+mutation_strength)))
        )
    
    @classmethod
    def reproduce(cls, parent1, parent2):
        """Sexual reproduction - mix genes from two parents"""
        # Random chromosome crossover
        return cls(
            survival_freq=parent1.survival_freq if random.random() < 0.5 else parent2.survival_freq,
            survival_amp=parent1.survival_amp if random.random() < 0.5 else parent2.survival_amp,
            reproduction_freq=parent1.reproduction_freq if random.random() < 0.5 else parent2.reproduction_freq,
            reproduction_amp=parent1.reproduction_amp if random.random() < 0.5 else parent2.reproduction_amp,
            energy_efficiency=(parent1.energy_efficiency + parent2.energy_efficiency) / 2,
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2,
            # Mix sensing abilities
            temperature_sensitivity=parent1.temperature_sensitivity if random.random() < 0.5 else parent2.temperature_sensitivity,
            resource_detection=parent1.resource_detection if random.random() < 0.5 else parent2.resource_detection,
            danger_awareness=parent1.danger_awareness if random.random() < 0.5 else parent2.danger_awareness,
            # Mix memory and movement
            memory_capacity=parent1.memory_capacity if random.random() < 0.5 else parent2.memory_capacity,
            movement_speed=parent1.movement_speed if random.random() < 0.5 else parent2.movement_speed,
            # Mix communication
            communication_range=parent1.communication_range if random.random() < 0.5 else parent2.communication_range,
            social_learning=parent1.social_learning if random.random() < 0.5 else parent2.social_learning,
            # Mix teaching traits
            teaching_willingness=parent1.teaching_willingness if random.random() < 0.5 else parent2.teaching_willingness,
            learning_from_parents=parent1.learning_from_parents if random.random() < 0.5 else parent2.learning_from_parents
        ).mutate()

class WaveEntity:
    """A living wave-based organism"""
    
    def __init__(self, entity_id: int, genome: WaveGenome, generation: int = 0):
        self.id = entity_id
        self.genome = genome
        self.generation = generation
        self.age = 0
        self.energy = 100.0
        self.reproduction_energy = 0.0
        self.alive = True
        self.offspring_count = 0
        self.lineage = [entity_id]
        # Spatial and memory attributes
        self.x = random.uniform(0, 100)  # Position in 100x100 world
        self.y = random.uniform(0, 100)
        self.memories = []  # List of (x, y, resource_quality, danger_level, age_when_visited)
    
    def survive_tick(self, environment, time: float) -> bool:
        """Attempt to survive one time tick"""
        if not self.alive:
            return False
        
        # Calculate survival wave output
        survival_wave = (self.genome.survival_amp * 
                        math.sin(self.genome.survival_freq * time))
        
        # Environment sensing and response (now location-based)
        temperature = environment.temperature
        danger_level = environment.danger
        
        # Get resource availability at current location
        resource_availability = environment.get_local_resources(self.x, self.y)
        
        # How well this entity senses and responds to environment
        temp_response = self.genome.temperature_sensitivity * (1.0 - abs(temperature - 20.0) / 30.0)  # Optimal at 20°C
        resource_finding = self.genome.resource_detection * resource_availability
        danger_avoidance = self.genome.danger_awareness * (1.0 - danger_level)
        
        # Combined environmental adaptation
        environmental_fitness = (temp_response + resource_finding + danger_avoidance) / 3.0
        
        # Survival effectiveness combines wave output and environmental sensing
        survival_effectiveness = abs(survival_wave) * max(0.1, environmental_fitness)
        
        # Energy dynamics
        base_energy_cost = 10.0 / self.genome.energy_efficiency
        
        # Sensing abilities cost energy to maintain (reduced cost)
        sensing_cost = (self.genome.temperature_sensitivity + 
                       self.genome.resource_detection + 
                       self.genome.danger_awareness - 3.0) * 0.5  # Modest cost for above-baseline sensing
        
        # Memory, movement, communication and teaching costs
        memory_cost = (self.genome.memory_capacity - 1.0) * 0.3  # Cost for remembering more
        movement_cost = (self.genome.movement_speed - 1.0) * 0.2  # Cost for moving faster
        communication_cost = (self.genome.communication_range - 1.0) * 0.4 + (self.genome.social_learning - 1.0) * 0.3  # Cost for communication
        teaching_cost = (self.genome.teaching_willingness - 1.0) * 0.1 + (self.genome.learning_from_parents - 1.0) * 0.1  # Cost for teaching abilities
        
        total_energy_cost = base_energy_cost + max(0, sensing_cost) + max(0, memory_cost) + max(0, movement_cost) + max(0, communication_cost) + max(0, teaching_cost)
        energy_gained = survival_effectiveness * 15.0
        
        self.energy += energy_gained - total_energy_cost
        self.age += 1
        
        # Store memory of current location
        self.remember_location(environmental_fitness, danger_level)
        
        # Store reference to nearby entities for social learning  
        # This will be set by the simulation
        nearby_entities = getattr(self, '_nearby_entities', None)
        
        # Move to better location based on memory and social learning
        self.move_strategically(nearby_entities)
        
        # Death conditions
        if self.energy <= 0:
            self.alive = False
            return False
        
        # Environmental dangers can cause death (but less harsh)
        if danger_level > 0.8 and self.genome.danger_awareness < 1.0:
            if random.random() < (danger_level - self.genome.danger_awareness) * 0.1:
                self.alive = False
                return False
        
        # Age-related death probability
        death_probability = max(0, (self.age - 1000) / 5000.0)
        if random.random() < death_probability:
            self.alive = False
            return False
        
        # Accumulate reproduction energy if well-fed
        if self.energy > 80:
            reproduction_wave = (self.genome.reproduction_amp * 
                               math.sin(self.genome.reproduction_freq * time))
            self.reproduction_energy += abs(reproduction_wave) * 2.0
        
        return True
    
    def remember_location(self, resource_quality: float, danger_level: float):
        """Store memory of current location"""
        # Only remember if we have memory capacity
        max_memories = int(self.genome.memory_capacity * 5)  # Can remember 5 locations per capacity point
        
        memory = (self.x, self.y, resource_quality, danger_level, self.age)
        self.memories.append(memory)
        
        # Forget oldest memories if over capacity
        if len(self.memories) > max_memories:
            self.memories = self.memories[-max_memories:]
    
    def move_strategically(self, nearby_entities=None):
        """Move towards better remembered locations, potentially influenced by others"""
        target_x, target_y = None, None
        
        # First, check own memories
        if len(self.memories) >= 2:
            best_memory = None
            best_score = -999
            
            for memory in self.memories[-10:]:  # Check recent memories
                mem_x, mem_y, resources, danger, age_visited = memory
                # Prefer high resources, low danger, and not too old
                score = resources - danger - (self.age - age_visited) * 0.01
                if score > best_score:
                    best_score = score
                    best_memory = memory
            
            if best_memory:
                target_x, target_y = best_memory[0], best_memory[1]
        
        # Social learning: maybe follow successful neighbors
        if nearby_entities and self.genome.social_learning > 1.0:
            for other in nearby_entities:
                if other.alive and other.energy > self.energy * 1.2:  # Follow richer entities
                    distance_to_other = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                    if distance_to_other <= self.genome.communication_range * 10:  # Within communication range
                        # Maybe follow them instead of own plan
                        if random.random() < (self.genome.social_learning - 1.0) * 0.3:
                            target_x, target_y = other.x, other.y
                            break
        
        # Move toward target or randomly
        if target_x is not None:
            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                move_distance = min(self.genome.movement_speed, distance)
                self.x += (dx / distance) * move_distance
                self.y += (dy / distance) * move_distance
        else:
            # Random movement if no target
            self.x += random.uniform(-self.genome.movement_speed, self.genome.movement_speed)
            self.y += random.uniform(-self.genome.movement_speed, self.genome.movement_speed)
        
        # Keep in bounds
        self.x = max(0, min(100, self.x))
        self.y = max(0, min(100, self.y))
    
    def can_reproduce(self) -> bool:
        """Check if entity can reproduce"""
        return (self.alive and 
                self.energy > 80 and 
                self.reproduction_energy > 50 and
                self.age > 25)
    
    def reproduce_with(self, partner) -> Optional['WaveEntity']:
        """Attempt reproduction with another entity"""
        if not (self.can_reproduce() and partner.can_reproduce()):
            return None
        
        # Energy cost of reproduction
        self.energy -= 40
        partner.energy -= 40
        self.reproduction_energy = 0
        partner.reproduction_energy = 0
        
        # Create offspring  
        child_genome = WaveGenome.reproduce(self.genome, partner.genome)
        child_id = max(self.id, partner.id) * 1000 + self.offspring_count + partner.offspring_count
        child = WaveEntity(child_id, child_genome, max(self.generation, partner.generation) + 1)
        child.lineage = self.lineage + [child_id]
        
        # Generational knowledge transfer - parents teach best memories
        self.teach_offspring(child, partner)
        
        self.offspring_count += 1
        partner.offspring_count += 1
        
        return child
    
    def teach_offspring(self, child, partner):
        """Transfer knowledge from parents to offspring"""
        # Combine both parents' teaching willingness
        avg_teaching = (self.genome.teaching_willingness + partner.genome.teaching_willingness) / 2
        
        if avg_teaching > 1.0 and len(self.memories) > 0:
            # Parents only teach if they have excess energy
            teaching_cost = (avg_teaching - 1.0) * 2
            can_teach = self.energy > 100 + teaching_cost/2 and partner.energy > 100 + teaching_cost/2
            
            if can_teach:
                self.energy -= teaching_cost / 2
                partner.energy -= teaching_cost / 2
                
                # Get best memories from both parents
                all_parent_memories = []
                
                # Get my best memories
                for memory in self.memories[-5:]:  # Recent memories
                    x, y, resources, danger, age_visited = memory
                    score = resources - danger  # Quality score
                    all_parent_memories.append((score, memory))
                
                # Get partner's best memories
                for memory in partner.memories[-5:]:
                    x, y, resources, danger, age_visited = memory
                    score = resources - danger
                    all_parent_memories.append((score, memory))
                
                # Sort by quality and take the best
                all_parent_memories.sort(key=lambda x: x[0], reverse=True)
                
                # Child inherits knowledge based on learning_from_parents trait
                if child.genome.learning_from_parents > 1.0:
                    num_to_inherit = int((child.genome.learning_from_parents - 1.0) * 3)  # 0-3 memories
                    for i in range(min(num_to_inherit, len(all_parent_memories))):
                        _, memory = all_parent_memories[i]
                        # Child gets memory but marked as "inherited" (age = 0)
                        x, y, resources, danger, _ = memory
                        child.memories.append((x, y, resources, danger, 0))  # Age 0 = inherited knowledge

class WaveLifeEnvironment:
    """Rich simulated environment for wave entities"""
    
    def __init__(self):
        self.time = 0.0
        self.temperature = 20.0      # Baseline temperature
        self.resources = 1.0         # Food availability
        self.danger = 0.0            # Environmental threats
        self.carrying_capacity = 100
        self.season_length = 1000
        self.weather_cycle = 0
        # Migration-inducing resource patches
        self.resource_patches = [
            {'x': 25, 'y': 25, 'quality': 2.0, 'active': True},
            {'x': 75, 'y': 75, 'quality': 2.0, 'active': False},
            {'x': 25, 'y': 75, 'quality': 1.5, 'active': False},
            {'x': 75, 'y': 25, 'quality': 1.5, 'active': False}
        ]
    
    def update(self, population_size: int):
        """Update environment based on time and population"""
        self.time += 0.1
        
        # Seasonal changes in temperature and resources
        season_cycle = (self.time % self.season_length) / self.season_length
        self.temperature = 20.0 + 15.0 * math.sin(2 * math.pi * season_cycle)  # 5°C to 35°C
        self.resources = 0.7 + 0.6 * math.sin(2 * math.pi * season_cycle)
        
        # Weather events that create danger
        self.weather_cycle += 0.01
        weather_intensity = math.sin(5 * self.weather_cycle)  # Faster weather changes
        if weather_intensity > 0.8:  # Storms/disasters
            self.danger = 0.8 + 0.2 * random.random()
        elif weather_intensity < -0.8:  # Extreme cold/heat
            self.danger = 0.5
            self.temperature += random.uniform(-10, 10)  # Temperature swings
        else:
            self.danger = max(0.0, self.danger - 0.1)  # Danger slowly decreases
        
        # Cycle resource patches every 200 ticks to force migration  
        patch_cycle = int((self.time // 20) % len(self.resource_patches))  # Cycle every 200 ticks
        for i, patch in enumerate(self.resource_patches):
            patch['active'] = (i == patch_cycle)
        
        # Active patch tracking for display only
        
        # Population pressure affects resources
        overcrowding = max(0, (population_size - self.carrying_capacity) / self.carrying_capacity)
        self.resources = max(0.1, self.resources - overcrowding * 0.3)
    
    def get_state(self) -> Dict:
        return {
            'temperature': self.temperature,
            'resources': self.resources,
            'danger': self.danger,
            'time': self.time,
            'resource_patches': self.resource_patches
        }
    
    def get_local_resources(self, x: float, y: float) -> float:
        """Get resource availability at a specific location"""
        base_resources = self.resources
        
        # Check if near any active resource patches
        for patch in self.resource_patches:
            if patch['active']:
                distance = math.sqrt((x - patch['x'])**2 + (y - patch['y'])**2)
                if distance < 15:  # Resource patch radius
                    # Closer = better resources
                    patch_bonus = patch['quality'] * (1.0 - distance / 15)
                    base_resources += patch_bonus
        
        return base_resources

class WaveLifeSimulation:
    """Main simulation of wave-based artificial life"""
    
    def __init__(self, seed=None, initial_population=4):
        if seed is not None:
            random.seed(seed)
        self.entities = []
        self.environment = WaveLifeEnvironment()
        self.generation_stats = []
        self.entity_counter = 0
        self.initial_population = initial_population
        self.seed = seed
    
    def initialize_population(self):
        """Initialize population with specified number of entities"""
        for i in range(self.initial_population):
            genome = WaveGenome(
                survival_freq=random.uniform(0.5, 2.0),
                survival_amp=random.uniform(0.5, 2.0),
                reproduction_freq=random.uniform(0.3, 1.5),
                reproduction_amp=random.uniform(0.3, 1.5),
                energy_efficiency=random.uniform(0.8, 1.2),
                mutation_rate=random.uniform(0.05, 0.15),
                # Initialize sensing abilities
                temperature_sensitivity=random.uniform(0.5, 1.5),
                resource_detection=random.uniform(0.5, 1.5),
                danger_awareness=random.uniform(0.5, 1.5),
                # Initialize memory and movement
                memory_capacity=random.uniform(0.5, 1.5),
                movement_speed=random.uniform(0.5, 1.5),
                # Initialize communication
                communication_range=random.uniform(0.5, 1.5),
                social_learning=random.uniform(0.5, 1.5),
                # Initialize teaching
                teaching_willingness=random.uniform(0.5, 1.5),
                learning_from_parents=random.uniform(0.5, 1.5)
            )
            entity = WaveEntity(self.entity_counter, genome)
            self.entities.append(entity)
            self.entity_counter += 1
    
    def simulation_tick(self):
        """Run one simulation step"""
        env_state = self.environment.get_state()
        
        # Survival phase with social learning
        surviving_entities = []
        for entity in self.entities:
            # Find nearby entities for social learning
            nearby = []
            for other in self.entities:
                if other != entity and other.alive:
                    distance = math.sqrt((entity.x - other.x)**2 + (entity.y - other.y)**2)
                    if distance <= entity.genome.communication_range * 10:  # Communication range
                        nearby.append(other)
            
            # Pass nearby entities to the entity
            entity._nearby_entities = nearby
            
            if entity.survive_tick(self.environment, self.environment.time):  # Pass environment object
                surviving_entities.append(entity)
        
        self.entities = surviving_entities
        
        # Reproduction phase
        new_offspring = []
        reproducers = [e for e in self.entities if e.can_reproduce()]
        
        # Random mating
        random.shuffle(reproducers)
        for i in range(0, len(reproducers) - 1, 2):
            child = reproducers[i].reproduce_with(reproducers[i + 1])
            if child:
                new_offspring.append(child)
        
        self.entities.extend(new_offspring)
        
        # Update environment
        self.environment.update(len(self.entities))
    
    def get_population_stats(self) -> Dict:
        """Get current population statistics"""
        if not self.entities:
            return {
                'population': 0,
                'avg_age': 0,
                'avg_generation': 0,
                'avg_energy': 0,
                'genetic_diversity': 0
            }
        
        ages = [e.age for e in self.entities]
        generations = [e.generation for e in self.entities]
        energies = [e.energy for e in self.entities]
        
        # Genetic diversity (variance in key traits)
        survival_freqs = [e.genome.survival_freq for e in self.entities]
        genetic_diversity = max(survival_freqs) - min(survival_freqs) if len(set(survival_freqs)) > 1 else 0
        
        return {
            'population': len(self.entities),
            'avg_age': sum(ages) / len(ages),
            'avg_generation': sum(generations) / len(generations),
            'avg_energy': sum(energies) / len(energies),
            'genetic_diversity': genetic_diversity,
            'oldest_lineage': max(e.generation for e in self.entities) if self.entities else 0
        }
    
    def run_simulation(self, max_ticks: int = 20000):
        """Run the complete simulation"""
        print("=== WAVE-BASED ARTIFICIAL LIFE SIMULATION ===")
        seed_info = f" (seed: {self.seed})" if self.seed is not None else ""
        print(f"Starting with {self.initial_population} entities{seed_info}, let evolution decide their fate...\n")
        
        self.initialize_population()
        self.extinction_tick = None  # Track when extinction occurred
        
        for tick in range(max_ticks):
            self.simulation_tick()
            
            # Log progress more frequently for long runs
            log_interval = 50 if tick < 2000 else 100
            if tick % log_interval == 0:
                stats = self.get_population_stats()
                self.generation_stats.append(stats)
                env_state = self.environment.get_state()
                
                # Find active patch
                active_patch = None
                for patch in env_state['resource_patches']:
                    if patch['active']:
                        active_patch = f"({patch['x']},{patch['y']})"
                        break
                
                # More detailed reporting for long runs
                if tick >= 1000:
                    avg_teaching = sum(e.genome.teaching_willingness for e in self.entities) / len(self.entities) if self.entities else 0
                    avg_social = sum(e.genome.social_learning for e in self.entities) / len(self.entities) if self.entities else 0
                    avg_memory = sum(len(e.memories) for e in self.entities) / len(self.entities) if self.entities else 0
                    
                    print(f"Tick {tick:4d}: Pop={stats['population']:3d}, "
                          f"Gen={stats['avg_generation']:4.1f}, "
                          f"Teaching={avg_teaching:.2f}, Social={avg_social:.2f}, "
                          f"Memories={avg_memory:.1f}, Patch={active_patch}")
                else:
                    print(f"Tick {tick:4d}: Pop={stats['population']:3d}, "
                          f"Gen={stats['avg_generation']:4.1f}, "
                          f"Energy={stats['avg_energy']:5.1f}, "
                          f"Patch={active_patch}, "
                          f"Danger={env_state['danger']:3.1f}")
                
                # Check for extinction
                if stats['population'] == 0:
                    print(f"\n💀 EXTINCTION at tick {tick}")
                    self.extinction_tick = tick
                    break
                
                # Check for population explosion (but let it continue longer)
                if stats['population'] > 500:
                    print(f"\n🌱 MASSIVE POPULATION at tick {tick} - continuing to observe long-term evolution...")
                elif stats['population'] > 200 and tick < 1000:
                    print(f"\n🌱 THRIVING POPULATION at tick {tick} - continuing evolution...")
                elif stats['population'] > 200 and tick >= 1000:
                    print(f"\n📊 OBSERVING LONG-TERM EVOLUTION at tick {tick}...")
        
        # Final analysis
        self.analyze_evolution()
    
    def analyze_evolution(self):
        """Analyze evolutionary outcomes"""
        print(f"\n=== EVOLUTIONARY ANALYSIS ===")
        
        if not self.entities:
            print("Population went extinct - no analysis possible")
            return
        
        final_stats = self.get_population_stats()
        print(f"Final Population: {final_stats['population']}")
        print(f"Maximum Generation Reached: {final_stats['oldest_lineage']}")
        print(f"Genetic Diversity: {final_stats['genetic_diversity']:.3f}")
        
        # Analyze successful genomes
        print(f"\n=== SUCCESSFUL WAVE GENETICS ===")
        entities_by_fitness = sorted(self.entities, key=lambda e: e.energy + e.age/10, reverse=True)
        
        for i, entity in enumerate(entities_by_fitness[:5]):
            print(f"Entity {entity.id} (Gen {entity.generation}):")
            print(f"  Survival Wave: freq={entity.genome.survival_freq:.2f}, amp={entity.genome.survival_amp:.2f}")
            print(f"  Reproduction Wave: freq={entity.genome.reproduction_freq:.2f}, amp={entity.genome.reproduction_amp:.2f}")
            print(f"  Energy Efficiency: {entity.genome.energy_efficiency:.2f}")
            print(f"  Sensing: Temp={entity.genome.temperature_sensitivity:.2f}, Resource={entity.genome.resource_detection:.2f}, Danger={entity.genome.danger_awareness:.2f}")
            print(f"  Memory/Movement: Memory={entity.genome.memory_capacity:.2f}, Speed={entity.genome.movement_speed:.2f}, Memories={len(entity.memories)}")
            print(f"  Communication: Range={entity.genome.communication_range:.2f}, Social={entity.genome.social_learning:.2f}")
            print(f"  Teaching: Teach={entity.genome.teaching_willingness:.2f}, Learn={entity.genome.learning_from_parents:.2f}")
            print(f"  Location: ({entity.x:.1f}, {entity.y:.1f})")
            print(f"  Mutation Rate: {entity.genome.mutation_rate:.3f}")
            print(f"  Age: {entity.age}, Energy: {entity.energy:.1f}, Offspring: {entity.offspring_count}")
        
        # Save detailed results
        results = {
            'final_population': len(self.entities),
            'generation_stats': self.generation_stats,
            'successful_genomes': [
                {
                    'id': e.id,
                    'generation': e.generation,
                    'genome': e.genome.__dict__,
                    'age': e.age,
                    'energy': e.energy,
                    'offspring': e.offspring_count
                } for e in entities_by_fitness[:10]
            ]
        }
        
        with open('wave_life_evolution.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to wave_life_evolution.json")

def create_enhanced_genome():
    """Create an artificially enhanced genome with advanced capabilities"""
    return WaveGenome(
        survival_freq=3.0,  # Much higher frequency
        survival_amp=3.0,   # Much stronger amplitude  
        reproduction_freq=1.5,
        reproduction_amp=2.0,
        energy_efficiency=1.5,  # Super efficient
        mutation_rate=0.05,     # Low mutation for stability
        # Enhanced sensing abilities
        temperature_sensitivity=2.0,  # Perfect temperature sensing
        resource_detection=2.0,       # Perfect resource finding
        danger_awareness=2.0,         # Perfect danger avoidance
        # Enhanced memory and movement
        memory_capacity=3.0,          # Perfect memory
        movement_speed=2.0,           # Perfect movement
        # Enhanced communication
        communication_range=2.0,      # Perfect communication range
        social_learning=2.0,          # Perfect social learning
        # Enhanced teaching
        teaching_willingness=2.0,     # Perfect teaching
        learning_from_parents=2.0     # Perfect learning from parents
    )

def run_dual_experiment():
    """Run control vs intervention experiment"""
    print("=== DUAL EVOLUTION EXPERIMENT ===")
    print("Control: Natural evolution")
    print("Experimental: Divine intervention at generation 3")
    
    # Control simulation
    print("\n--- CONTROL GROUP (Natural Evolution) ---")
    control = WaveLifeSimulation()
    control.initialize_population()
    
    # Experimental simulation
    print("\n--- EXPERIMENTAL GROUP (With Intervention) ---")
    experimental = WaveLifeSimulation()
    experimental.initialize_population()
    
    # Run both for 300 ticks
    for tick in range(300):
        control.simulation_tick()
        experimental.simulation_tick()
        
        # Intervention at tick 150 (around generation 3)
        if tick == 150:
            # Add 2 enhanced entities to experimental group
            enhanced1 = WaveEntity(999999, create_enhanced_genome(), generation=0)
            enhanced2 = WaveEntity(999998, create_enhanced_genome(), generation=0)
            experimental.entities.extend([enhanced1, enhanced2])
            print(f"\n🔥 DIVINE INTERVENTION at tick {tick} - Added enhanced entities!")
        
        if tick % 50 == 0:
            c_stats = control.get_population_stats()
            e_stats = experimental.get_population_stats()
            print(f"Tick {tick:3d} | Control: Pop={c_stats['population']:3d} Gen={c_stats['avg_generation']:4.1f} | Experimental: Pop={e_stats['population']:3d} Gen={e_stats['avg_generation']:4.1f}")
    
    # Final comparison
    print("\n=== FINAL COMPARISON ===")
    c_final = control.get_population_stats()
    e_final = experimental.get_population_stats()
    
    print(f"Control Final:      Pop={c_final['population']:3d}, Max Gen={c_final['oldest_lineage']}")
    print(f"Experimental Final: Pop={e_final['population']:3d}, Max Gen={e_final['oldest_lineage']}")
    
    # Analyze top entities from each
    if control.entities:
        print("\nTop Control Entity:")
        top_c = max(control.entities, key=lambda e: e.energy)
        print(f"  Survival: freq={top_c.genome.survival_freq:.2f}, amp={top_c.genome.survival_amp:.2f}")
    
    if experimental.entities:
        print("\nTop Experimental Entity:")
        top_e = max(experimental.entities, key=lambda e: e.energy)
        print(f"  Survival: freq={top_e.genome.survival_freq:.2f}, amp={top_e.genome.survival_amp:.2f}")

def run_experiment(seed=None, initial_pop=4, max_ticks=10000):
    """Run a single experiment with given parameters"""
    simulation = WaveLifeSimulation(seed=seed, initial_population=initial_pop)
    simulation.run_simulation(max_ticks)
    return simulation

if __name__ == "__main__":
    # Run single experiment
    run_experiment()