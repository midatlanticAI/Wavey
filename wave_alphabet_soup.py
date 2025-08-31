#!/usr/bin/env python3
"""
WAVE ALPHABET SOUP - Pure Physics Substrate
No cheating. Only wave interference equations govern everything.
Throw entities into soup, let wave physics settle what emerges.
"""
import random
import math
import json
from collections import defaultdict

class WaveField:
    """Pure wave physics substrate - the universe's physics engine"""
    
    def __init__(self, world_size=500):
        self.size = world_size
        self.time = 0.0
        self.dt = 0.1  # Time step
        
        # Multi-frequency wave field - this IS the physics
        # Each point stores wave amplitudes for different frequencies
        self.wave_grid = {}  # (x, y, frequency) -> complex amplitude
        self.wave_sources = []  # Active wave sources
        
        # Climate zones affect wave propagation
        self.climate_zones = self.generate_climate_zones()
        
        # Natural wave sources from environment
        self.environmental_sources = []
        self.generate_environmental_sources()
        
    def generate_climate_zones(self):
        """Create diverse climate affecting wave propagation"""
        zones = {}
        
        # Desert regions - high temperature, different wave speeds
        for _ in range(8):
            center_x = random.randint(50, self.size-50)
            center_y = random.randint(50, self.size-50)
            radius = random.randint(30, 80)
            zones[f"desert_{len(zones)}"] = {
                'type': 'desert',
                'center': (center_x, center_y),
                'radius': radius,
                'temperature': random.uniform(35, 45),  # Hot
                'wave_speed_modifier': 1.3,  # Faster wave propagation
                'wave_damping': 0.02  # Higher damping
            }
        
        # Mountain regions - cold, slow wave propagation
        for _ in range(6):
            center_x = random.randint(50, self.size-50)
            center_y = random.randint(50, self.size-50)
            radius = random.randint(40, 90)
            zones[f"mountain_{len(zones)}"] = {
                'type': 'mountain',
                'center': (center_x, center_y),
                'radius': radius,
                'temperature': random.uniform(5, 15),  # Cold
                'wave_speed_modifier': 0.7,  # Slower wave propagation
                'wave_damping': 0.05  # Higher damping from terrain
            }
        
        # Valley regions - temperate, optimal wave conditions
        for _ in range(10):
            center_x = random.randint(50, self.size-50)
            center_y = random.randint(50, self.size-50)
            radius = random.randint(25, 60)
            zones[f"valley_{len(zones)}"] = {
                'type': 'valley',
                'center': (center_x, center_y),
                'radius': radius,
                'temperature': random.uniform(18, 25),  # Optimal
                'wave_speed_modifier': 1.0,  # Normal wave propagation
                'wave_damping': 0.01  # Low damping
            }
        
        # Ocean/lake regions - wave reflection and resonance
        for _ in range(4):
            center_x = random.randint(80, self.size-80)
            center_y = random.randint(80, self.size-80)
            radius = random.randint(60, 120)
            zones[f"water_{len(zones)}"] = {
                'type': 'water',
                'center': (center_x, center_y),
                'radius': radius,
                'temperature': random.uniform(12, 22),
                'wave_speed_modifier': 1.4,  # Fast propagation
                'wave_damping': 0.005,  # Very low damping
                'wave_reflection': 0.8  # High reflection coefficient
            }
        
        return zones
    
    def generate_environmental_sources(self):
        """Natural wave sources in the environment"""
        
        # Resource deposits create standing wave patterns
        for _ in range(15):
            x = random.uniform(50, self.size-50)
            y = random.uniform(50, self.size-50)
            frequency = random.uniform(0.5, 3.0)
            amplitude = random.uniform(0.3, 1.2)
            
            source = {
                'type': 'resource_deposit',
                'x': x, 'y': y,
                'frequency': frequency,
                'amplitude': amplitude,
                'phase': random.uniform(0, 2*math.pi),
                'resource_density': amplitude * 0.8
            }
            self.environmental_sources.append(source)
        
        # Danger sources create chaotic wave interference
        for _ in range(8):
            x = random.uniform(100, self.size-100)
            y = random.uniform(100, self.size-100)
            
            # Multiple interfering frequencies create chaos
            frequencies = [random.uniform(4.0, 8.0) for _ in range(3)]
            amplitudes = [random.uniform(0.6, 1.5) for _ in range(3)]
            
            source = {
                'type': 'danger_zone',
                'x': x, 'y': y,
                'frequencies': frequencies,
                'amplitudes': amplitudes,
                'phases': [random.uniform(0, 2*math.pi) for _ in range(3)],
                'danger_level': sum(amplitudes) / 3
            }
            self.environmental_sources.append(source)
    
    def get_climate_at_position(self, x, y):
        """Get climate properties affecting wave physics at position"""
        # Default climate
        climate = {
            'temperature': 20.0,
            'wave_speed_modifier': 1.0,
            'wave_damping': 0.015,
            'wave_reflection': 0.0
        }
        
        # Find dominant climate zone
        min_distance = float('inf')
        dominant_zone = None
        
        for zone_data in self.climate_zones.values():
            center_x, center_y = zone_data['center']
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            if distance < zone_data['radius'] and distance < min_distance:
                min_distance = distance
                dominant_zone = zone_data
        
        if dominant_zone:
            # Blend with default based on distance from center
            blend_factor = 1.0 - (min_distance / dominant_zone['radius'])
            climate['temperature'] = (climate['temperature'] * (1-blend_factor) + 
                                    dominant_zone['temperature'] * blend_factor)
            climate['wave_speed_modifier'] = (climate['wave_speed_modifier'] * (1-blend_factor) +
                                            dominant_zone['wave_speed_modifier'] * blend_factor)
            climate['wave_damping'] = (climate['wave_damping'] * (1-blend_factor) +
                                     dominant_zone['wave_damping'] * blend_factor)
            
            if 'wave_reflection' in dominant_zone:
                climate['wave_reflection'] = dominant_zone['wave_reflection'] * blend_factor
        
        return climate
    
    def calculate_wave_at_position(self, x, y, frequency):
        """Pure wave physics - calculate wave amplitude at position/frequency"""
        total_amplitude = 0.0 + 0.0j  # Complex amplitude
        
        climate = self.get_climate_at_position(x, y)
        wave_speed = 340.0 * climate['wave_speed_modifier']  # Base wave speed modified by climate
        
        # Contributions from environmental sources
        for source in self.environmental_sources:
            if source['type'] == 'resource_deposit':
                if abs(source['frequency'] - frequency) < 0.5:  # Frequency matching
                    distance = math.sqrt((x - source['x'])**2 + (y - source['y'])**2)
                    
                    if distance > 0:
                        # Wave equation with climate-modified propagation
                        wave_number = 2 * math.pi * frequency / wave_speed
                        phase = wave_number * distance + source['phase'] + frequency * self.time
                        
                        # Amplitude decreases with distance, modified by climate damping
                        amplitude = source['amplitude'] * math.exp(-climate['wave_damping'] * distance) / math.sqrt(distance + 1)
                        
                        # Complex wave amplitude
                        wave_contribution = amplitude * (math.cos(phase) + 1j * math.sin(phase))
                        total_amplitude += wave_contribution
            
            elif source['type'] == 'danger_zone':
                for i, src_freq in enumerate(source['frequencies']):
                    if abs(src_freq - frequency) < 0.5:
                        distance = math.sqrt((x - source['x'])**2 + (y - source['y'])**2)
                        
                        if distance > 0:
                            wave_number = 2 * math.pi * src_freq / wave_speed
                            phase = wave_number * distance + source['phases'][i] + src_freq * self.time
                            amplitude = source['amplitudes'][i] * math.exp(-climate['wave_damping'] * distance) / math.sqrt(distance + 1)
                            
                            wave_contribution = amplitude * (math.cos(phase) + 1j * math.sin(phase))
                            total_amplitude += wave_contribution
        
        # Contributions from entity wave sources
        for source in self.wave_sources:
            if abs(source['frequency'] - frequency) < 0.3:
                distance = math.sqrt((x - source['x'])**2 + (y - source['y'])**2)
                
                if distance > 0:
                    wave_number = 2 * math.pi * frequency / wave_speed
                    phase = wave_number * distance + source['phase'] + frequency * self.time
                    amplitude = source['amplitude'] * math.exp(-climate['wave_damping'] * distance) / math.sqrt(distance + 1)
                    
                    wave_contribution = amplitude * (math.cos(phase) + 1j * math.sin(phase))
                    total_amplitude += wave_contribution
        
        # Wave reflection in water bodies
        if climate['wave_reflection'] > 0:
            # Simple reflection model - creates standing wave patterns
            reflected_amplitude = total_amplitude * climate['wave_reflection'] * (math.cos(2 * frequency * self.time) + 1j * math.sin(2 * frequency * self.time))
            total_amplitude += reflected_amplitude
        
        return abs(total_amplitude)  # Return magnitude
    
    def get_wave_spectrum_at_position(self, x, y, frequency_range=(0.1, 8.0), num_samples=16):
        """Get full wave spectrum at position - the basis for all computation"""
        spectrum = {}
        frequencies = [frequency_range[0] + i * (frequency_range[1] - frequency_range[0]) / num_samples 
                      for i in range(num_samples)]
        
        for freq in frequencies:
            spectrum[freq] = self.calculate_wave_at_position(x, y, freq)
        
        return spectrum
    
    def advance_time(self):
        """Advance wave field time step"""
        self.time += self.dt
        
        # Clean old wave sources that have decayed
        self.wave_sources = [s for s in self.wave_sources if s.get('duration', float('inf')) > 0]
        for source in self.wave_sources:
            if 'duration' in source:
                source['duration'] -= self.dt
    
    def add_wave_source(self, x, y, frequency, amplitude, phase=0, duration=None):
        """Add temporary wave source (from entity activity)"""
        source = {
            'x': x, 'y': y,
            'frequency': frequency,
            'amplitude': amplitude,
            'phase': phase,
            'creation_time': self.time
        }
        
        if duration:
            source['duration'] = duration
            
        self.wave_sources.append(source)

class PureWaveEntity:
    """Entity whose ALL behavior emerges from wave interactions"""
    
    def __init__(self, entity_id, x, y):
        self.id = entity_id
        self.x = x
        self.y = y
        self.age = 0
        self.energy = 100.0
        
        # Wave sensing capabilities - these determine what they can perceive
        self.sensor_frequencies = []
        for _ in range(random.randint(3, 8)):  # Random number of frequency sensors
            freq = random.uniform(0.1, 8.0)
            sensitivity = random.uniform(0.1, 1.0)
            self.sensor_frequencies.append({'frequency': freq, 'sensitivity': sensitivity})
        
        # Wave production capabilities - how they affect the wave field
        self.wave_generators = []
        for _ in range(random.randint(1, 3)):  # Can generate 1-3 frequencies
            freq = random.uniform(0.5, 6.0)
            max_amplitude = random.uniform(0.2, 0.8)
            self.wave_generators.append({'frequency': freq, 'max_amplitude': max_amplitude})
        
        # Emergent properties from wave interactions
        self.movement_resonance = random.uniform(0.5, 4.0)  # Which frequency drives movement
        self.energy_resonance = random.uniform(0.3, 3.0)    # Which frequency provides energy
        self.social_resonance = random.uniform(1.0, 5.0)    # Which frequency for communication
        
        # Memory of wave patterns - purely emergent learning
        self.wave_memory = []  # Stores (location, spectrum, outcome) associations
        self.behavioral_adaptations = []  # Learned behaviors from wave experiences
        
    def sense_wave_environment(self, wave_field):
        """Sense wave environment - this is their ONLY input about the world"""
        sensed_spectrum = {}
        
        for sensor in self.sensor_frequencies:
            freq = sensor['frequency']
            sensitivity = sensor['sensitivity']
            
            # Get actual wave amplitude at this frequency
            wave_amplitude = wave_field.calculate_wave_at_position(self.x, self.y, freq)
            
            # Entity can only sense what their sensors detect
            perceived_amplitude = wave_amplitude * sensitivity
            sensed_spectrum[freq] = perceived_amplitude
        
        return sensed_spectrum
    
    def compute_behavior_from_waves(self, sensed_spectrum, wave_field):
        """ALL behavior computed from wave interactions - no hardcoded rules"""
        
        # Energy calculation from wave resonance
        energy_wave_amplitude = 0
        for freq, amplitude in sensed_spectrum.items():
            # Energy comes from constructive interference at entity's energy resonance
            if abs(freq - self.energy_resonance) < 0.5:
                energy_wave_amplitude += amplitude
        
        # Energy change computed from wave physics
        if energy_wave_amplitude > 0.3:
            # Good wave conditions - gain energy
            energy_gain = energy_wave_amplitude * 15
            self.energy += energy_gain
        else:
            # Poor wave conditions - lose energy
            self.energy -= 8
        
        # Movement computed from wave gradients
        movement_decision = self.compute_wave_gradient_movement(sensed_spectrum, wave_field)
        
        # Wave production based on internal state
        self.produce_waves(wave_field)
        
        # Learning from wave-outcome associations
        self.learn_from_wave_experience(sensed_spectrum, energy_gain if 'energy_gain' in locals() else -8)
        
        return movement_decision
    
    def compute_wave_gradient_movement(self, sensed_spectrum, wave_field):
        """Movement computed purely from wave gradients"""
        
        # Find the frequency that matches movement resonance
        movement_amplitude = 0
        for freq, amplitude in sensed_spectrum.items():
            if abs(freq - self.movement_resonance) < 0.4:
                movement_amplitude = amplitude
                break
        
        if movement_amplitude < 0.2:
            return False  # No movement - insufficient wave energy
        
        # Sample wave field in nearby directions to find gradient
        sample_distance = 3.0
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        best_direction = None
        best_wave_strength = 0
        
        for dx, dy in directions:
            sample_x = self.x + dx * sample_distance
            sample_y = self.y + dy * sample_distance
            
            # Don't sample outside world bounds
            if sample_x < 0 or sample_x >= wave_field.size or sample_y < 0 or sample_y >= wave_field.size:
                continue
            
            # Get wave strength at sample point for movement frequency
            wave_strength = wave_field.calculate_wave_at_position(sample_x, sample_y, self.movement_resonance)
            
            if wave_strength > best_wave_strength:
                best_wave_strength = wave_strength
                best_direction = (dx, dy)
        
        # Move toward stronger wave field if significantly better
        if best_direction and best_wave_strength > movement_amplitude * 1.2:
            move_distance = min(2.0, movement_amplitude * 3)  # Movement speed from wave amplitude
            
            self.x += best_direction[0] * move_distance
            self.y += best_direction[1] * move_distance
            
            # Keep in bounds
            self.x = max(0, min(wave_field.size - 1, self.x))
            self.y = max(0, min(wave_field.size - 1, self.y))
            
            # Movement costs energy
            self.energy -= movement_amplitude * 2
            
            return True
        
        return False
    
    def produce_waves(self, wave_field):
        """Produce waves based on internal state - affects environment"""
        
        if self.energy < 20:
            return  # Too low energy to produce waves
        
        for generator in self.wave_generators:
            # Wave amplitude based on energy and internal resonances
            base_amplitude = generator['max_amplitude']
            
            # Modulate amplitude based on energy state
            energy_factor = min(1.0, self.energy / 100.0)
            actual_amplitude = base_amplitude * energy_factor
            
            if actual_amplitude > 0.1:  # Only produce significant waves
                # Add wave to field
                wave_field.add_wave_source(
                    self.x, self.y, 
                    generator['frequency'],
                    actual_amplitude,
                    phase=random.uniform(0, 2*math.pi),
                    duration=3.0  # Wave persists for 3 time units
                )
                
                # Producing waves costs energy
                self.energy -= actual_amplitude * 3
    
    def learn_from_wave_experience(self, sensed_spectrum, energy_outcome):
        """Learn associations between wave patterns and outcomes"""
        
        if abs(energy_outcome) > 5:  # Significant outcome worth remembering
            # Identify key frequencies that might have caused outcome
            significant_frequencies = []
            for freq, amplitude in sensed_spectrum.items():
                if amplitude > 0.4:  # Strong enough to matter
                    significant_frequencies.append((freq, amplitude))
            
            if significant_frequencies:
                experience = {
                    'location': (self.x, self.y),
                    'wave_pattern': significant_frequencies,
                    'outcome': energy_outcome,
                    'age': self.age
                }
                
                self.wave_memory.append(experience)
                
                # Keep memory manageable
                if len(self.wave_memory) > 20:
                    self.wave_memory = self.wave_memory[-20:]
        
        # Develop behavioral adaptations based on experience
        self.adapt_behavior_from_memory()
    
    def adapt_behavior_from_memory(self):
        """Adapt behavior based on accumulated wave experiences"""
        
        if len(self.wave_memory) < 5:
            return
        
        # Find patterns in successful experiences
        positive_experiences = [exp for exp in self.wave_memory if exp['outcome'] > 10]
        negative_experiences = [exp for exp in self.wave_memory if exp['outcome'] < -10]
        
        if positive_experiences:
            # Identify frequencies associated with good outcomes
            frequency_success = defaultdict(list)
            for exp in positive_experiences:
                for freq, amplitude in exp['wave_pattern']:
                    frequency_success[freq].append(exp['outcome'])
            
            # Find most consistently successful frequency
            best_freq = None
            best_avg_outcome = 0
            for freq, outcomes in frequency_success.items():
                if len(outcomes) >= 2:
                    avg_outcome = sum(outcomes) / len(outcomes)
                    if avg_outcome > best_avg_outcome:
                        best_avg_outcome = avg_outcome
                        best_freq = freq
            
            # Adapt movement resonance toward successful frequency
            if best_freq and abs(best_freq - self.movement_resonance) > 0.5:
                adaptation_strength = 0.1
                self.movement_resonance += (best_freq - self.movement_resonance) * adaptation_strength
                
                # Record behavioral adaptation
                adaptation = {
                    'type': 'movement_resonance_shift',
                    'old_resonance': self.movement_resonance,
                    'new_target': best_freq,
                    'success_evidence': len(positive_experiences),
                    'age': self.age
                }
                self.behavioral_adaptations.append(adaptation)
    
    def attempt_wave_reproduction(self, other_entities, wave_field):
        """Reproduction based on wave synchronization"""
        
        if self.energy < 80 or self.age < 100:
            return None
        
        # Find entities with compatible wave patterns
        compatible_partners = []
        
        for other in other_entities:
            if other.id != self.id and other.energy > 60:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                
                if distance < 20:  # Must be nearby
                    # Check wave frequency compatibility
                    compatibility = 0
                    for my_gen in self.wave_generators:
                        for other_gen in other.wave_generators:
                            freq_diff = abs(my_gen['frequency'] - other_gen['frequency'])
                            if freq_diff < 1.0:  # Compatible frequencies
                                compatibility += 1 - freq_diff
                    
                    if compatibility > 1.5:  # Sufficient compatibility
                        compatible_partners.append((other, compatibility))
        
        if not compatible_partners:
            return None
        
        # Choose best partner
        partner, compatibility = max(compatible_partners, key=lambda x: x[1])
        
        # Create offspring with mixed wave characteristics
        child_x = (self.x + partner.x) / 2 + random.uniform(-10, 10)
        child_y = (self.y + partner.y) / 2 + random.uniform(-10, 10)
        child_x = max(0, min(wave_field.size - 1, child_x))
        child_y = max(0, min(wave_field.size - 1, child_y))
        
        child = PureWaveEntity(len(other_entities) + random.randint(1000, 9999), child_x, child_y)
        
        # Mix wave characteristics
        child.sensor_frequencies = []
        for i in range(min(len(self.sensor_frequencies), len(partner.sensor_frequencies))):
            if random.random() < 0.5:
                child.sensor_frequencies.append(self.sensor_frequencies[i].copy())
            else:
                child.sensor_frequencies.append(partner.sensor_frequencies[i].copy())
        
        child.wave_generators = []
        for i in range(min(len(self.wave_generators), len(partner.wave_generators))):
            mixed_generator = {
                'frequency': (self.wave_generators[i]['frequency'] + partner.wave_generators[i]['frequency']) / 2,
                'max_amplitude': (self.wave_generators[i]['max_amplitude'] + partner.wave_generators[i]['max_amplitude']) / 2
            }
            child.wave_generators.append(mixed_generator)
        
        # Mix resonance frequencies
        child.movement_resonance = (self.movement_resonance + partner.movement_resonance) / 2
        child.energy_resonance = (self.energy_resonance + partner.energy_resonance) / 2
        child.social_resonance = (self.social_resonance + partner.social_resonance) / 2
        
        # Apply mutations
        mutation_rate = 0.1
        if random.random() < mutation_rate:
            child.movement_resonance *= random.uniform(0.8, 1.2)
        if random.random() < mutation_rate:
            child.energy_resonance *= random.uniform(0.8, 1.2)
        
        # Parents lose energy from reproduction
        self.energy -= 40
        partner.energy -= 40
        
        return child

class WaveAlphabetSoupSimulation:
    """Pure wave physics simulation - let it settle naturally"""
    
    def __init__(self, world_size=500, initial_entities=20, seed=None):
        if seed:
            random.seed(seed)
        
        self.wave_field = WaveField(world_size)
        self.entities = []
        self.tick = 0
        
        # Create initial entities scattered in world
        for i in range(initial_entities):
            x = random.uniform(50, world_size - 50)
            y = random.uniform(50, world_size - 50)
            entity = PureWaveEntity(i, x, y)
            self.entities.append(entity)
        
        # Track emergent phenomena - but don't force them
        self.emergent_behaviors = []
        self.wave_discoveries = []
        self.extinction_events = []
        
    def simulation_step(self):
        """Single step of pure wave physics"""
        self.tick += 1
        
        # Update wave field physics
        self.wave_field.advance_time()
        
        # Each entity responds to waves
        for entity in self.entities[:]:
            entity.age += 1
            
            # Entity senses wave environment
            sensed_waves = entity.sense_wave_environment(self.wave_field)
            
            # Behavior computed from waves
            entity.compute_behavior_from_waves(sensed_waves, self.wave_field)
            
            # Death from energy depletion
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Wave-based reproduction attempts
        new_entities = []
        for entity in self.entities:
            if random.random() < 0.02:  # 2% chance to attempt reproduction
                child = entity.attempt_wave_reproduction(self.entities, self.wave_field)
                if child:
                    new_entities.append(child)
        
        self.entities.extend(new_entities)
        
        # Detect emergent phenomena
        if self.tick % 100 == 0:
            self.detect_emergent_patterns()
    
    def detect_emergent_patterns(self):
        """Detect patterns that emerge naturally from wave physics"""
        
        if not self.entities:
            return
        
        # Spatial clustering - entities gathering due to wave patterns
        position_clusters = defaultdict(list)
        for entity in self.entities:
            cluster_key = (int(entity.x // 50), int(entity.y // 50))  # 50x50 clusters
            position_clusters[cluster_key].append(entity)
        
        large_clusters = {k: v for k, v in position_clusters.items() if len(v) >= 4}
        if large_clusters:
            self.emergent_behaviors.append({
                'type': 'spatial_clustering',
                'tick': self.tick,
                'clusters': len(large_clusters),
                'largest_cluster_size': max(len(entities) for entities in large_clusters.values())
            })
        
        # Behavioral convergence - entities adapting to similar resonances
        movement_resonances = [e.movement_resonance for e in self.entities]
        if len(movement_resonances) > 5:
            # Check for convergence
            mean_resonance = sum(movement_resonances) / len(movement_resonances)
            variance = sum((r - mean_resonance)**2 for r in movement_resonances) / len(movement_resonances)
            
            if variance < 0.5:  # Low variance indicates convergence
                self.emergent_behaviors.append({
                    'type': 'behavioral_convergence',
                    'tick': self.tick,
                    'convergent_frequency': mean_resonance,
                    'population': len(self.entities),
                    'variance': variance
                })
        
        # Wave innovation - entities discovering new beneficial frequencies
        innovators = [e for e in self.entities if len(e.behavioral_adaptations) > 0]
        if len(innovators) > len(self.entities) * 0.3:  # 30% have adapted
            self.wave_discoveries.append({
                'type': 'wave_adaptation_spread',
                'tick': self.tick,
                'innovator_count': len(innovators),
                'total_adaptations': sum(len(e.behavioral_adaptations) for e in innovators)
            })
        
        # Population dynamics
        if len(self.entities) < 5:
            self.extinction_events.append({
                'type': 'population_crash',
                'tick': self.tick,
                'remaining_population': len(self.entities)
            })

def run_wave_alphabet_soup(max_ticks=3000, world_size=500, seed=None):
    """Run pure wave physics simulation"""
    if not seed:
        seed = random.randint(1000, 9999)
    
    print("=== WAVE ALPHABET SOUP EXPERIMENT ===")
    print(f"Pure wave physics substrate - no cheating")
    print(f"World: {world_size}x{world_size}, Max ticks: {max_ticks}, Seed: {seed}")
    print(f"Climate zones: desert, mountain, valley, water")
    print(f"Let the wave physics settle...\n")
    
    simulation = WaveAlphabetSoupSimulation(world_size, initial_entities=25, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 500 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 WAVE SOUP WENT EXTINCT")
                break
            
            # Basic stats
            pop = len(simulation.entities)
            avg_energy = sum(e.energy for e in simulation.entities) / pop
            avg_age = sum(e.age for e in simulation.entities) / pop
            
            # Wave characteristics
            movement_freqs = [e.movement_resonance for e in simulation.entities]
            energy_freqs = [e.energy_resonance for e in simulation.entities]
            
            avg_movement_freq = sum(movement_freqs) / len(movement_freqs)
            avg_energy_freq = sum(energy_freqs) / len(energy_freqs)
            
            print(f"T{tick:4d}: Pop={pop:2d}, Energy={avg_energy:.1f}, Age={avg_age:.1f}")
            print(f"        Wave resonances: Move={avg_movement_freq:.2f}Hz, Energy={avg_energy_freq:.2f}Hz")
            
            # Emergent behaviors
            recent_emergent = len([b for b in simulation.emergent_behaviors if b['tick'] > tick - 500])
            recent_discoveries = len([d for d in simulation.wave_discoveries if d['tick'] > tick - 500])
            
            if recent_emergent > 0:
                print(f"        🌊 {recent_emergent} emergent behaviors detected")
            if recent_discoveries > 0:
                print(f"        🔍 {recent_discoveries} wave discoveries")
            
            # Adaptive learning
            total_adaptations = sum(len(e.behavioral_adaptations) for e in simulation.entities)
            total_memories = sum(len(e.wave_memory) for e in simulation.entities)
            
            if total_adaptations > 0:
                print(f"        🧠 {total_adaptations} behavioral adaptations, {total_memories} wave memories")
    
    # Final analysis
    print(f"\n=== WAVE SOUP SETTLED RESULTS ===")
    
    if simulation.entities:
        print(f"✅ Wave ecosystem survived: {len(simulation.entities)} entities")
        
        # Final wave characteristics
        final_movement_freqs = [e.movement_resonance for e in simulation.entities]
        final_energy_freqs = [e.energy_resonance for e in simulation.entities]
        
        print(f"\n🌊 Final wave resonances:")
        print(f"   Movement frequencies: {min(final_movement_freqs):.2f} - {max(final_movement_freqs):.2f} Hz")
        print(f"   Energy frequencies: {min(final_energy_freqs):.2f} - {max(final_energy_freqs):.2f} Hz")
        
        # Emergent behaviors discovered
        if simulation.emergent_behaviors:
            print(f"\n🌟 Emergent phenomena observed:")
            behavior_types = defaultdict(int)
            for behavior in simulation.emergent_behaviors:
                behavior_types[behavior['type']] += 1
            
            for behavior_type, count in behavior_types.items():
                print(f"   {behavior_type.replace('_', ' ').title()}: {count} occurrences")
        
        # Wave discoveries
        if simulation.wave_discoveries:
            print(f"\n🔍 Wave discoveries:")
            for discovery in simulation.wave_discoveries[-3:]:
                print(f"   {discovery['type']}: {discovery.get('innovator_count', 'N/A')} innovators")
        
        # Most successful entities
        print(f"\n🏆 Most adapted entities:")
        top_entities = sorted(simulation.entities, 
                            key=lambda e: len(e.behavioral_adaptations) + len(e.wave_memory), 
                            reverse=True)[:3]
        
        for entity in top_entities:
            adaptations = len(entity.behavioral_adaptations)
            memories = len(entity.wave_memory)
            print(f"   Entity {entity.id}: {adaptations} adaptations, {memories} wave memories")
            print(f"     Resonances: Move={entity.movement_resonance:.2f}Hz, Energy={entity.energy_resonance:.2f}Hz")
            print(f"     Age: {entity.age}, Energy: {entity.energy:.1f}")
    
    else:
        print("💀 Wave alphabet soup failed to produce stable patterns")
        if simulation.extinction_events:
            print(f"   Extinction events: {len(simulation.extinction_events)}")
    
    return simulation

if __name__ == "__main__":
    run_wave_alphabet_soup(max_ticks=4000, world_size=500)