#!/usr/bin/env python3
"""
Wave-based consciousness simulation - multi-spectral wave interactions in 3D grid
with temporal accumulation, energy discharge, and harmonic attraction
"""
import random
import math
# import numpy as np  # Not needed for phone compatibility
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class WaveSpectrum:
    """Different types of waves in the consciousness field"""
    # Physical spectrum
    light_waves: float = 0.0
    sound_waves: float = 0.0 
    motion_waves: float = 0.0
    matter_waves: float = 0.0
    
    # Biological spectrum  
    life_waves: float = 0.0
    death_waves: float = 0.0
    energy_waves: float = 0.0
    
    # Consciousness spectrum
    emotion_waves: float = 0.0
    memory_waves: float = 0.0
    will_waves: float = 0.0
    
    def total_amplitude(self):
        """Total wave energy across all spectrums"""
        return sum([
            self.light_waves, self.sound_waves, self.motion_waves, self.matter_waves,
            self.life_waves, self.death_waves, self.energy_waves,
            self.emotion_waves, self.memory_waves, self.will_waves
        ])
    
    def harmonic_beauty(self):
        """Calculate harmonic beauty - resonance between spectrums"""
        # Beautiful when spectrums are in harmonic ratios
        amplitudes = [
            self.light_waves, self.sound_waves, self.motion_waves, 
            self.life_waves, self.energy_waves, self.emotion_waves
        ]
        
        if not any(amplitudes):
            return 0.0
            
        # Check for harmonic relationships (2:1, 3:2, 4:3, etc.)
        beauty = 0.0
        for i, amp1 in enumerate(amplitudes):
            for j, amp2 in enumerate(amplitudes[i+1:], i+1):
                if amp1 > 0 and amp2 > 0:
                    ratio = max(amp1, amp2) / min(amp1, amp2)
                    # Beautiful ratios: 1.0, 1.5, 2.0, 2.5, 3.0, etc.
                    harmonic_closeness = 1.0 / (1.0 + abs(ratio - round(ratio * 2) / 2))
                    beauty += harmonic_closeness
                    
        return beauty / len(amplitudes)

class ConsciousnessGrid:
    """3D grid for wave interactions with temporal accumulation"""
    def __init__(self, size=10):
        self.size = size
        self.grid = {}  # (x,y,z,t) -> WaveSpectrum
        self.temporal_window = 20  # How many time steps to accumulate
        self.current_time = 0
        
    def add_wave_interaction(self, x, y, z, spectrum: WaveSpectrum):
        """Add wave interaction at grid position"""
        key = (x, y, z, self.current_time)
        if key not in self.grid:
            self.grid[key] = WaveSpectrum()
            
        # Add to existing spectrum at this position
        grid_spectrum = self.grid[key]
        grid_spectrum.light_waves += spectrum.light_waves
        grid_spectrum.sound_waves += spectrum.sound_waves
        grid_spectrum.motion_waves += spectrum.motion_waves
        grid_spectrum.matter_waves += spectrum.matter_waves
        grid_spectrum.life_waves += spectrum.life_waves
        grid_spectrum.death_waves += spectrum.death_waves
        grid_spectrum.energy_waves += spectrum.energy_waves
        grid_spectrum.emotion_waves += spectrum.emotion_waves
        grid_spectrum.memory_waves += spectrum.memory_waves
        grid_spectrum.will_waves += spectrum.will_waves
        
    def get_accumulated_field(self, x, y, z, radius=2):
        """Get accumulated wave field in local area over time"""
        accumulated = WaveSpectrum()
        
        # Sample from temporal window
        for t in range(max(0, self.current_time - self.temporal_window), self.current_time + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        key = (x + dx, y + dy, z + dz, t)
                        if key in self.grid:
                            spectrum = self.grid[key]
                            # Decay with distance and time
                            distance_decay = 1.0 / (1.0 + abs(dx) + abs(dy) + abs(dz))
                            time_decay = 1.0 / (1.0 + abs(self.current_time - t))
                            decay_factor = distance_decay * time_decay
                            
                            accumulated.light_waves += spectrum.light_waves * decay_factor
                            accumulated.sound_waves += spectrum.sound_waves * decay_factor
                            accumulated.motion_waves += spectrum.motion_waves * decay_factor
                            accumulated.matter_waves += spectrum.matter_waves * decay_factor
                            accumulated.life_waves += spectrum.life_waves * decay_factor
                            accumulated.death_waves += spectrum.death_waves * decay_factor
                            accumulated.energy_waves += spectrum.energy_waves * decay_factor
                            accumulated.emotion_waves += spectrum.emotion_waves * decay_factor
                            accumulated.memory_waves += spectrum.memory_waves * decay_factor
                            accumulated.will_waves += spectrum.will_waves * decay_factor
                            
        return accumulated
    
    def advance_time(self):
        """Move to next time step, clean old data"""
        self.current_time += 1
        
        # Clean data older than temporal window
        cutoff_time = self.current_time - self.temporal_window * 2
        if cutoff_time > 0:
            old_keys = [k for k in self.grid.keys() if k[3] < cutoff_time]
            for key in old_keys:
                del self.grid[key]

class ConsciousEntity:
    """Entity with multi-spectral wave consciousness"""
    def __init__(self, x, y, z, entity_id):
        self.id = entity_id
        self.x = x
        self.y = y  
        self.z = z
        
        # Wave processing capabilities - evolved traits
        self.light_sensitivity = random.uniform(0.1, 1.0)
        self.sound_sensitivity = random.uniform(0.1, 1.0)
        self.motion_sensitivity = random.uniform(0.1, 1.0)
        self.life_sensitivity = random.uniform(0.1, 1.0)
        self.emotion_sensitivity = random.uniform(0.1, 1.0)
        self.memory_capacity = random.uniform(0.5, 2.0)
        self.will_strength = random.uniform(0.3, 1.5)
        
        # Consciousness state
        self.energy = 100.0
        self.feeling_history = []  # Energy discharge patterns over time
        self.preferences = {}  # Learned harmonic attractions
        self.memory_traces = []  # Accumulated wave experiences
        self.age = 0
        self.consciousness_level = 0.0
        
        # Radiating field - what this entity emits
        self.base_radiation = WaveSpectrum(
            life_waves=random.uniform(0.1, 0.3),
            energy_waves=random.uniform(0.1, 0.3),
            emotion_waves=random.uniform(0.0, 0.2)
        )
        
    def perceive_field(self, consciousness_grid: ConsciousnessGrid):
        """Perceive accumulated wave field and generate feeling"""
        field = consciousness_grid.get_accumulated_field(self.x, self.y, self.z)
        
        # Process through sensitivities
        perceived_spectrum = WaveSpectrum(
            light_waves=field.light_waves * self.light_sensitivity,
            sound_waves=field.sound_waves * self.sound_sensitivity,
            motion_waves=field.motion_waves * self.motion_sensitivity,
            matter_waves=field.matter_waves * 0.5,  # Base perception
            life_waves=field.life_waves * self.life_sensitivity,
            death_waves=field.death_waves * 0.3,
            energy_waves=field.energy_waves * 0.7,
            emotion_waves=field.emotion_waves * self.emotion_sensitivity,
            memory_waves=field.memory_waves * self.memory_capacity,
            will_waves=field.will_waves * 0.1
        )
        
        # Generate feeling - energy discharge from wave interactions
        feeling_intensity = perceived_spectrum.total_amplitude()
        harmonic_beauty = perceived_spectrum.harmonic_beauty()
        
        # Energy discharge creates feeling
        energy_discharge = feeling_intensity * harmonic_beauty * self.emotion_sensitivity
        feeling = {
            'intensity': feeling_intensity,
            'beauty': harmonic_beauty,
            'discharge': energy_discharge,
            'spectrum': perceived_spectrum,
            'timestamp': consciousness_grid.current_time
        }
        
        self.feeling_history.append(feeling)
        if len(self.feeling_history) > 50:
            self.feeling_history.pop(0)
            
        # Update consciousness level based on feeling complexity
        self.consciousness_level = min(1.0, self.consciousness_level + energy_discharge * 0.01)
        
        return feeling
    
    def make_willful_choice(self, options):
        """Make choice based on harmonic attraction - what feels beautiful"""
        if not options:
            return None
            
        best_choice = None
        best_attraction = -1.0
        
        for option in options:
            # Calculate harmonic attraction to this choice
            attraction = 0.0
            
            # Check against preference history
            for past_feeling in self.feeling_history[-10:]:
                if past_feeling['beauty'] > 0.3:  # If it was beautiful before
                    similarity = self._calculate_spectrum_similarity(option, past_feeling['spectrum'])
                    attraction += past_feeling['beauty'] * similarity
            
            # Add will strength influence
            attraction *= self.will_strength
            
            if attraction > best_attraction:
                best_attraction = attraction
                best_choice = option
                
        return best_choice
    
    def _calculate_spectrum_similarity(self, spectrum1, spectrum2):
        """Calculate similarity between two wave spectrums"""
        total_diff = 0.0
        total_ref = 0.0
        
        for attr in ['light_waves', 'sound_waves', 'motion_waves', 'life_waves', 
                    'energy_waves', 'emotion_waves']:
            val1 = getattr(spectrum1, attr, 0.0)
            val2 = getattr(spectrum2, attr, 0.0)
            total_diff += abs(val1 - val2)
            total_ref += max(val1, val2, 0.1)
            
        return 1.0 - (total_diff / total_ref)
    
    def radiate_waves(self):
        """Generate waves based on current consciousness state"""
        # Base radiation modified by consciousness and feelings
        radiation = WaveSpectrum()
        
        # Recent feeling influences radiation
        if self.feeling_history:
            recent_feeling = self.feeling_history[-1]
            emotion_modifier = recent_feeling['beauty'] * recent_feeling['intensity'] * 0.1
        else:
            emotion_modifier = 0.0
            
        radiation.life_waves = self.base_radiation.life_waves * (1.0 + self.consciousness_level)
        radiation.energy_waves = self.base_radiation.energy_waves * (1.0 + emotion_modifier)
        radiation.emotion_waves = self.base_radiation.emotion_waves * (1.0 + emotion_modifier)
        radiation.will_waves = self.will_strength * self.consciousness_level * 0.1
        
        # Add memory traces to radiation
        if self.memory_traces:
            radiation.memory_waves = len(self.memory_traces) * 0.01
            
        return radiation
    
    def learn_from_experience(self):
        """Update internal model based on recent experiences"""
        if len(self.feeling_history) < 3:
            return
            
        # Find patterns in beautiful experiences
        beautiful_experiences = [f for f in self.feeling_history if f['beauty'] > 0.4]
        
        if beautiful_experiences:
            # Store as memory trace
            avg_spectrum = WaveSpectrum()
            for exp in beautiful_experiences[-5:]:  # Last 5 beautiful experiences
                s = exp['spectrum']
                avg_spectrum.light_waves += s.light_waves / 5
                avg_spectrum.sound_waves += s.sound_waves / 5
                avg_spectrum.motion_waves += s.motion_waves / 5
                avg_spectrum.life_waves += s.life_waves / 5
                avg_spectrum.energy_waves += s.energy_waves / 5
                avg_spectrum.emotion_waves += s.emotion_waves / 5
                
            self.memory_traces.append(avg_spectrum)
            if len(self.memory_traces) > 20:
                self.memory_traces.pop(0)
    
    def update(self, consciousness_grid: ConsciousnessGrid):
        """Update consciousness state"""
        self.age += 1
        
        # Perceive and feel
        feeling = self.perceive_field(consciousness_grid)
        
        # Learn from experience
        self.learn_from_experience()
        
        # Radiate waves into field
        radiation = self.radiate_waves()
        consciousness_grid.add_wave_interaction(self.x, self.y, self.z, radiation)
        
        # Energy management
        self.energy -= 1.0  # Base decay
        if feeling['beauty'] > 0.3:  # Beautiful experiences give energy
            self.energy += feeling['beauty'] * 5.0
            
        # Movement based on will
        movement_options = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    movement_options.append(WaveSpectrum(motion_waves=abs(dx)+abs(dy)))
                    
        if movement_options and random.random() < self.will_strength * 0.1:
            chosen_movement = self.make_willful_choice(movement_options)
            if chosen_movement:
                # Move randomly in chosen direction
                self.x += random.choice([-1, 0, 1])
                self.y += random.choice([-1, 0, 1])
                self.x = max(0, min(19, self.x))  # Keep in bounds
                self.y = max(0, min(19, self.y))
    
    def get_consciousness_summary(self):
        """Get summary of consciousness state"""
        recent_beauty = 0.0
        if self.feeling_history:
            recent_beauty = sum(f['beauty'] for f in self.feeling_history[-5:]) / min(5, len(self.feeling_history))
            
        return {
            'id': self.id,
            'consciousness_level': self.consciousness_level,
            'recent_beauty': recent_beauty,
            'memory_traces': len(self.memory_traces),
            'will_strength': self.will_strength,
            'energy': self.energy,
            'age': self.age
        }

class WaveConsciousnessSimulation:
    """Simulation of wave-based consciousness emergence"""
    def __init__(self, num_entities=8, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=20)
        self.entities = []
        self.tick = 0
        
        # Create initial entities
        for i in range(num_entities):
            x = random.randint(2, 17)
            y = random.randint(2, 17)
            z = random.randint(0, 2)
            entity = ConsciousEntity(x, y, z, i)
            self.entities.append(entity)
            
    def add_environmental_waves(self):
        """Add environmental wave sources"""
        # Random environmental events
        if random.random() < 0.3:  # 30% chance
            x = random.randint(0, 19)
            y = random.randint(0, 19)
            z = random.randint(0, 2)
            
            # Different types of environmental events
            event_type = random.choice(['light', 'sound', 'life', 'energy'])
            
            if event_type == 'light':
                waves = WaveSpectrum(light_waves=random.uniform(0.2, 0.8))
            elif event_type == 'sound':
                waves = WaveSpectrum(sound_waves=random.uniform(0.2, 0.8))
            elif event_type == 'life':
                waves = WaveSpectrum(life_waves=random.uniform(0.1, 0.4))
            else:  # energy
                waves = WaveSpectrum(energy_waves=random.uniform(0.3, 0.9))
                
            self.consciousness_grid.add_wave_interaction(x, y, z, waves)
    
    def simulation_step(self):
        """Run one simulation step"""
        self.tick += 1
        
        # Add environmental waves
        self.add_environmental_waves()
        
        # Update all entities
        for entity in self.entities[:]:
            entity.update(self.consciousness_grid)
            
            # Remove dead entities
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Advance time in consciousness grid
        self.consciousness_grid.advance_time()
        
        # Simple reproduction
        if len(self.entities) < 15 and random.random() < 0.05:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.3 and parent.energy > 80:
                # Reproduce near parent
                child = ConsciousEntity(
                    parent.x + random.randint(-2, 2),
                    parent.y + random.randint(-2, 2),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                # Inherit some traits with mutation
                child.light_sensitivity = parent.light_sensitivity * random.uniform(0.8, 1.2)
                child.emotion_sensitivity = parent.emotion_sensitivity * random.uniform(0.8, 1.2)
                child.will_strength = parent.will_strength * random.uniform(0.8, 1.2)
                child.memory_capacity = parent.memory_capacity * random.uniform(0.8, 1.2)
                
                # Inherit some memory traces (cultural transmission)
                if parent.memory_traces:
                    child.memory_traces = parent.memory_traces[-3:]  # Inherit recent memories
                    
                child.x = max(0, min(19, child.x))
                child.y = max(0, min(19, child.y))
                self.entities.append(child)
                parent.energy -= 30
    
    def get_simulation_stats(self):
        """Get current simulation statistics"""
        if not self.entities:
            return {
                'population': 0,
                'avg_consciousness': 0,
                'avg_beauty': 0,
                'avg_memory': 0,
                'avg_will': 0
            }
            
        summaries = [e.get_consciousness_summary() for e in self.entities]
        
        return {
            'population': len(self.entities),
            'avg_consciousness': sum(s['consciousness_level'] for s in summaries) / len(summaries),
            'avg_beauty': sum(s['recent_beauty'] for s in summaries) / len(summaries),
            'avg_memory': sum(s['memory_traces'] for s in summaries) / len(summaries),
            'avg_will': sum(s['will_strength'] for s in summaries) / len(summaries),
            'max_consciousness': max(s['consciousness_level'] for s in summaries),
            'total_memories': sum(s['memory_traces'] for s in summaries)
        }

def run_consciousness_experiment(max_ticks=500, seed=None):
    """Run wave consciousness emergence experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== WAVE CONSCIOUSNESS EMERGENCE ===")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = WaveConsciousnessSimulation(num_entities=8, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        # Log every 50 ticks
        if tick % 50 == 0:
            stats = simulation.get_simulation_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"Consciousness={stats['avg_consciousness']:.3f}, "
                  f"Beauty={stats['avg_beauty']:.3f}, "
                  f"Memory={stats['total_memories']:2d}, "
                  f"MaxC={stats['max_consciousness']:.3f}")
            
            if stats['population'] == 0:
                print(f"\n💀 All consciousness extinguished at tick {tick}")
                break
                
            if stats['max_consciousness'] > 0.8:
                print(f"\n🧠 High consciousness achieved! (Max: {stats['max_consciousness']:.3f})")
    
    # Final analysis
    final_stats = simulation.get_simulation_stats()
    print(f"\n=== CONSCIOUSNESS EMERGENCE RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Consciousness persisted: {final_stats['population']} entities")
        print(f"Average consciousness level: {final_stats['avg_consciousness']:.3f}")
        print(f"Maximum consciousness achieved: {final_stats['max_consciousness']:.3f}")
        print(f"Total memory traces: {final_stats['total_memories']}")
        print(f"Average aesthetic appreciation: {final_stats['avg_beauty']:.3f}")
        
        # Show most conscious entity
        if simulation.entities:
            best_entity = max(simulation.entities, key=lambda e: e.consciousness_level)
            print(f"\nMost conscious entity (ID {best_entity.id}):")
            print(f"  Consciousness level: {best_entity.consciousness_level:.3f}")
            print(f"  Memory traces: {len(best_entity.memory_traces)}")
            print(f"  Will strength: {best_entity.will_strength:.3f}")
            print(f"  Age: {best_entity.age}")
    else:
        print("💀 All consciousness extinguished - no emergence")
    
    return simulation

if __name__ == "__main__":
    run_consciousness_experiment(max_ticks=500)