#!/usr/bin/env python3
"""
Comprehensive Wave-Based Survival Agent
Multi-modal sensory processing, learning, adaptation, and goal hierarchy
No cheating - pure wave computation architecture
"""

import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

class WaveType(Enum):
    SPATIAL = "spatial"
    OLFACTORY = "olfactory" 
    AUDITORY = "auditory"
    HUNGER = "hunger"
    MEMORY = "memory"
    GOAL = "goal"

@dataclass
class WaveState:
    """Individual wave component with frequency, amplitude, phase"""
    frequency: float
    amplitude: float
    phase: float
    decay_rate: float = 0.99
    resonance_strength: float = 1.0

class HierarchicalWaveProcessor:
    """
    Multi-scale wave processing system
    Different frequency bands for different cognitive functions
    """
    
    def __init__(self):
        # Frequency bands (Hz)
        self.bands = {
            'reflexive': (50, 100),    # Immediate survival responses
            'tactical': (10, 50),      # Local navigation, sensory processing  
            'regional': (1, 10),       # Area exploration, threat avoidance
            'global': (0.1, 1)         # Long-term strategy, maze-wide patterns
        }
        
        # Wave states for each band and type
        self.wave_states = {}
        for band in self.bands:
            self.wave_states[band] = {}
            for wave_type in WaveType:
                self.wave_states[band][wave_type] = []
    
    def add_wave(self, band: str, wave_type: WaveType, frequency: float, amplitude: float, phase: float = 0):
        """Add new wave to specific band and type"""
        if band in self.wave_states and frequency >= self.bands[band][0] and frequency <= self.bands[band][1]:
            wave = WaveState(frequency, amplitude, phase)
            self.wave_states[band][wave_type].append(wave)
    
    def calculate_interference(self, band: str, wave_type: WaveType, position: Tuple[float, float], time_step: float) -> float:
        """Calculate wave interference at position and time"""
        total_amplitude = 0
        waves = self.wave_states[band][wave_type]
        
        for wave in waves:
            # Wave equation: A * sin(2π * f * t + φ)
            wave_value = wave.amplitude * math.sin(2 * math.pi * wave.frequency * time_step + wave.phase)
            
            # Spatial modulation based on position
            x, y = position
            spatial_factor = math.cos(wave.frequency * x * 0.1) * math.cos(wave.frequency * y * 0.1)
            
            total_amplitude += wave_value * spatial_factor * wave.resonance_strength
            
            # Natural decay
            wave.amplitude *= wave.decay_rate
        
        return total_amplitude
    
    def cross_frequency_coupling(self, source_band: str, target_band: str, coupling_strength: float = 0.1):
        """Create coupling between different frequency bands"""
        for wave_type in WaveType:
            source_waves = self.wave_states[source_band][wave_type]
            target_waves = self.wave_states[target_band][wave_type]
            
            for source_wave in source_waves:
                for target_wave in target_waves:
                    # Beat frequency coupling
                    if abs(source_wave.frequency - target_wave.frequency) < 5:  # Resonance condition
                        coupling_amplitude = coupling_strength * source_wave.amplitude * target_wave.amplitude
                        target_wave.resonance_strength += coupling_amplitude

class MultiModalSensor:
    """
    Wave-based multi-modal sensory system
    """
    
    def __init__(self, wave_processor: HierarchicalWaveProcessor):
        self.wave_processor = wave_processor
        self.sensory_range = {
            'smell': 8,    # Can smell food within 8 cells
            'hearing': 12, # Can hear enemies within 12 cells
            'vision': 3    # Can see obstacles within 3 cells
        }
    
    def detect_food(self, agent_pos: Tuple[int, int], food_positions: List[Tuple[int, int]], time_step: float):
        """Convert food proximity to olfactory waves"""
        x_agent, y_agent = agent_pos
        
        for food_pos in food_positions:
            x_food, y_food = food_pos
            distance = math.sqrt((x_food - x_agent)**2 + (y_food - y_agent)**2)
            
            if distance <= self.sensory_range['smell']:
                # Closer food creates higher frequency, stronger amplitude
                frequency = 15 + (10 / (distance + 1))  # Tactical band
                amplitude = 2.0 / (distance + 1)
                
                # Phase based on direction
                direction_angle = math.atan2(y_food - y_agent, x_food - x_agent)
                phase = direction_angle
                
                self.wave_processor.add_wave('tactical', WaveType.OLFACTORY, frequency, amplitude, phase)
    
    def detect_enemies(self, agent_pos: Tuple[int, int], enemy_positions: List[Tuple[int, int]], time_step: float):
        """Convert enemy proximity to auditory waves"""
        x_agent, y_agent = agent_pos
        
        for enemy_pos in enemy_positions:
            x_enemy, y_enemy = enemy_pos
            distance = math.sqrt((x_enemy - x_agent)**2 + (y_enemy - y_agent)**2)
            
            if distance <= self.sensory_range['hearing']:
                # Closer enemies create higher frequency, stronger amplitude
                frequency = 60 + (20 / (distance + 1))  # Reflexive band - danger!
                amplitude = 3.0 / (distance + 1)
                
                # Phase based on direction
                direction_angle = math.atan2(y_enemy - y_agent, x_enemy - x_agent)
                phase = direction_angle
                
                self.wave_processor.add_wave('reflexive', WaveType.AUDITORY, frequency, amplitude, phase)
    
    def update_hunger_state(self, hunger_level: float, time_step: float):
        """Convert internal hunger state to hunger waves"""
        # Hunger creates increasing frequency and amplitude as it grows
        frequency = 5 + (hunger_level * 10)  # Regional band
        amplitude = hunger_level * 2
        phase = time_step * 0.1  # Slowly evolving phase
        
        self.wave_processor.add_wave('regional', WaveType.HUNGER, frequency, amplitude, phase)

class WaveLearningSystem:
    """
    Learning through wave pattern adaptation and memory formation
    """
    
    def __init__(self, wave_processor: HierarchicalWaveProcessor):
        self.wave_processor = wave_processor
        self.experience_buffer = []
        self.concept_patterns = {}  # Learned wave patterns
        self.success_threshold = 0.8
    
    def record_experience(self, state: Dict, action: str, reward: float, outcome: str):
        """Record experience with wave state snapshot"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'outcome': outcome,
            'wave_snapshot': self._capture_wave_snapshot(),
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer) > 1000:
            self.experience_buffer.pop(0)
    
    def _capture_wave_snapshot(self) -> Dict:
        """Capture current wave state across all bands and types"""
        snapshot = {}
        for band in self.wave_processor.wave_states:
            snapshot[band] = {}
            for wave_type in WaveType:
                waves = self.wave_processor.wave_states[band][wave_type]
                snapshot[band][wave_type.value] = [
                    {'freq': w.frequency, 'amp': w.amplitude, 'phase': w.phase, 'resonance': w.resonance_strength}
                    for w in waves
                ]
        return snapshot
    
    def form_concepts(self):
        """Form concepts from repeated successful patterns"""
        if len(self.experience_buffer) < 10:
            return
        
        # Analyze successful experiences
        successful_experiences = [exp for exp in self.experience_buffer[-50:] if exp['reward'] > 0]
        
        if len(successful_experiences) >= 5:
            # Extract common wave patterns
            pattern_frequencies = {}
            
            for exp in successful_experiences:
                for band in exp['wave_snapshot']:
                    for wave_type in exp['wave_snapshot'][band]:
                        for wave in exp['wave_snapshot'][band][wave_type]:
                            freq_key = f"{band}_{wave_type}_{int(wave['freq'])}"
                            if freq_key not in pattern_frequencies:
                                pattern_frequencies[freq_key] = []
                            pattern_frequencies[freq_key].append(wave['amp'])
            
            # Identify consistent patterns
            for pattern, amplitudes in pattern_frequencies.items():
                if len(amplitudes) >= 3:  # Seen at least 3 times
                    avg_amplitude = sum(amplitudes) / len(amplitudes)
                    if avg_amplitude > 0.5:  # Strong enough pattern
                        self.concept_patterns[pattern] = avg_amplitude
    
    def apply_learned_concepts(self, current_state: Dict):
        """Apply learned concepts to enhance current wave patterns"""
        for concept, strength in self.concept_patterns.items():
            parts = concept.split('_')
            if len(parts) >= 3:
                band, wave_type_str, freq = parts[0], parts[1], int(parts[2])
                
                # Boost resonance for learned successful patterns
                try:
                    wave_type = WaveType(wave_type_str)
                    waves = self.wave_processor.wave_states[band][wave_type]
                    
                    for wave in waves:
                        if abs(wave.frequency - freq) < 2:  # Close frequency match
                            wave.resonance_strength += strength * 0.1
                except:
                    continue

class WaveSurvivalAgent:
    """
    Complete wave-based survival agent
    """
    
    def __init__(self, maze_width: int, maze_height: int):
        self.maze_width = maze_width
        self.maze_height = maze_height
        
        # Core systems
        self.wave_processor = HierarchicalWaveProcessor()
        self.sensors = MultiModalSensor(self.wave_processor)
        self.learning = WaveLearningSystem(self.wave_processor)
        
        # Agent state
        self.position = (1, 1)
        self.hunger_level = 0.0  # 0 = satisfied, 1 = starving
        self.hunger_rate = 0.005  # Hunger increases per step
        self.energy = 1.0
        self.alive = True
        
        # Experience tracking
        self.steps_taken = 0
        self.food_eaten = 0
        self.escapes = 0
        self.deaths = 0
        
        # Time tracking
        self.time_step = 0
    
    def perceive_environment(self, maze: List[List[int]], food_positions: List[Tuple[int, int]], 
                           enemy_positions: List[Tuple[int, int]]):
        """Update sensory wave patterns based on environment"""
        self.time_step += 1
        
        # Multi-modal sensory input
        self.sensors.detect_food(self.position, food_positions, self.time_step)
        self.sensors.detect_enemies(self.position, enemy_positions, self.time_step)
        self.sensors.update_hunger_state(self.hunger_level, self.time_step)
        
        # Spatial navigation waves (from previous maze system)
        self._update_spatial_waves(maze)
        
        # Cross-frequency coupling between different senses
        self.wave_processor.cross_frequency_coupling('tactical', 'regional', 0.2)
        self.wave_processor.cross_frequency_coupling('reflexive', 'tactical', 0.3)
        
        # Apply learned concepts
        self.learning.apply_learned_concepts({'position': self.position, 'hunger': self.hunger_level})
    
    def _update_spatial_waves(self, maze: List[List[int]]):
        """Update spatial navigation waves"""
        x, y = self.position
        
        # Create spatial waves for navigation
        frequency = 5 + (x + y) * 0.1
        amplitude = 1.0
        phase = self.time_step * 0.05
        
        self.wave_processor.add_wave('regional', WaveType.SPATIAL, frequency, amplitude, phase)
    
    def decide_action(self, maze: List[List[int]], food_positions: List[Tuple[int, int]], 
                     enemy_positions: List[Tuple[int, int]], exit_position: Tuple[int, int]) -> str:
        """
        Decide action based on wave interference patterns
        No hard-coded rules - pure wave computation
        """
        actions = ['north', 'south', 'east', 'west', 'stay']
        action_strengths = {}
        
        for action in actions:
            # Calculate potential position
            new_pos = self._get_new_position(self.position, action)
            
            if not self._is_valid_position(new_pos, maze):
                action_strengths[action] = -1000  # Invalid move
                continue
            
            # Calculate wave interference for this potential position
            total_strength = 0
            
            # Combine all wave types and bands
            for band in ['reflexive', 'tactical', 'regional', 'global']:
                for wave_type in WaveType:
                    interference = self.wave_processor.calculate_interference(
                        band, wave_type, new_pos, self.time_step
                    )
                    
                    # Weight different wave types based on urgency
                    weight = self._get_wave_type_weight(wave_type)
                    total_strength += interference * weight
            
            action_strengths[action] = total_strength
        
        # Select action with highest wave interference strength
        best_action = max(action_strengths.keys(), key=lambda k: action_strengths[k])
        
        # Record decision for learning
        self.learning.record_experience(
            state={'pos': self.position, 'hunger': self.hunger_level},
            action=best_action,
            reward=0,  # Will be updated after action execution
            outcome='pending'
        )
        
        return best_action
    
    def _get_wave_type_weight(self, wave_type: WaveType) -> float:
        """Dynamic weighting based on current needs"""
        if wave_type == WaveType.AUDITORY:
            return 5.0  # Danger is always high priority
        elif wave_type == WaveType.HUNGER:
            return 3.0 * self.hunger_level  # Increases with hunger
        elif wave_type == WaveType.OLFACTORY:
            return 2.0 if self.hunger_level > 0.3 else 1.0
        elif wave_type == WaveType.SPATIAL:
            return 1.0
        return 1.0
    
    def _get_new_position(self, pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Calculate new position based on action"""
        x, y = pos
        if action == 'north':
            return (x, y - 1)
        elif action == 'south':
            return (x, y + 1)
        elif action == 'east':
            return (x + 1, y)
        elif action == 'west':
            return (x - 1, y)
        else:  # stay
            return pos
    
    def _is_valid_position(self, pos: Tuple[int, int], maze: List[List[int]]) -> bool:
        """Check if position is valid (not wall, within bounds)"""
        x, y = pos
        if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
            return maze[y][x] == 0  # 0 = open, 1 = wall
        return False
    
    def execute_action(self, action: str, maze: List[List[int]], food_positions: List[Tuple[int, int]], 
                      enemy_positions: List[Tuple[int, int]], exit_position: Tuple[int, int]) -> Dict[str, Any]:
        """Execute action and update agent state"""
        old_position = self.position
        new_position = self._get_new_position(self.position, action)
        
        if self._is_valid_position(new_position, maze):
            self.position = new_position
        
        self.steps_taken += 1
        self.hunger_level = min(1.0, self.hunger_level + self.hunger_rate)
        
        # Check outcomes
        reward = 0
        outcome = 'move'
        
        # Food consumption
        if self.position in food_positions:
            food_positions.remove(self.position)
            self.hunger_level = max(0.0, self.hunger_level - 0.3)
            self.food_eaten += 1
            reward += 100
            outcome = 'food_eaten'
        
        # Enemy encounter
        if self.position in enemy_positions:
            self.alive = False
            self.deaths += 1
            reward -= 1000
            outcome = 'death'
        
        # Exit reached
        if self.position == exit_position:
            self.escapes += 1
            reward += 500
            outcome = 'escaped'
        
        # Starvation
        if self.hunger_level >= 1.0:
            self.alive = False
            self.deaths += 1
            reward -= 500
            outcome = 'starved'
        
        # Small negative reward for each step (encourages efficiency)
        reward -= 1
        
        # Update learning with reward
        if self.learning.experience_buffer:
            self.learning.experience_buffer[-1]['reward'] = reward
            self.learning.experience_buffer[-1]['outcome'] = outcome
        
        # Form concepts periodically
        if self.steps_taken % 50 == 0:
            self.learning.form_concepts()
        
        return {
            'position': self.position,
            'hunger': self.hunger_level,
            'alive': self.alive,
            'reward': reward,
            'outcome': outcome,
            'energy': self.energy
        }

# Test function will be added in next message due to length