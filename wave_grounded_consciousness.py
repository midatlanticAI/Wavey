#!/usr/bin/env python3
"""
WAVE-GROUNDED CONSCIOUSNESS EXPERIMENT
Consciousness that emerges FROM wave-computed survival, not abstract patterns
Built on the proper foundation: spatial environment, resources, tools, wave computation
"""
import random
import math
import json
from collections import defaultdict, Counter
from wave_consciousness_life import *
from wave_artificial_life import WaveGenome, WaveEntity

class WaveGroundedEntity(WaveEntity, ConsciousEntity):
    """Entity with consciousness grounded in wave-computed survival"""
    
    def __init__(self, entity_id, genome, x, y, z, generation=0):
        # Initialize both parents properly
        WaveEntity.__init__(self, entity_id, genome, generation)
        ConsciousEntity.__init__(self, x, y, z, entity_id)
        
        # Override position from WaveEntity random placement
        self.x = x
        self.y = y
        self.z = z
        
        # Wave-grounded consciousness abilities
        self.wave_pattern_recognition = random.uniform(0.2, 1.0)  # Recognize wave interference patterns
        self.resource_wave_memory = []  # Remember wave patterns associated with resources
        self.tool_resonance_knowledge = {}  # Know which wave frequencies improve tool performance
        self.territory_harmonics = {}  # Map harmonic beauty patterns to territory value
        
        # Tool-making capabilities grounded in wave understanding
        self.wave_tool_crafting = random.uniform(0.1, 0.8)  # Craft tools using wave resonance
        self.functional_tools = []  # Tools that actually affect survival
        
        # Social wave communication
        self.wave_teaching_ability = random.uniform(0.1, 0.6)  # Teach wave patterns to others
        self.wave_learning_receptivity = random.uniform(0.2, 0.8)  # Learn from others' wave discoveries
        
        # Consciousness discoveries grounded in their world
        self.survival_discoveries = []  # Patterns they discover about survival
        self.wave_mathematics = []  # Mathematical relationships they find in waves
        self.aesthetic_discoveries = []  # Beautiful wave patterns they create/find
        
        # Specialization emerges from wave consciousness + genome
        self.conscious_specialization = None  # Will emerge from wave + genetic traits
        
    def sense_wave_environment(self, wave_grid, environment, all_entities):
        """Sense the wave environment and its relationship to survival"""
        
        # Get accumulated wave field at current location
        local_waves = wave_grid.get_accumulated_field(self.x, self.y, self.z, radius=3)
        
        # Resource detection through wave patterns
        if self.wave_pattern_recognition > 0.5 and random.random() < 0.15:
            # Discover correlations between wave patterns and resources
            resource_level = environment.get_local_resources(self.x, self.y)
            
            if resource_level > 0.6:  # High resource area
                wave_signature = {
                    'life_waves': local_waves.life_waves,
                    'energy_waves': local_waves.energy_waves,
                    'resource_level': resource_level,
                    'location': (self.x, self.y),
                    'discovery_tick': self.age
                }
                self.resource_wave_memory.append(wave_signature)
                
                # Keep memory manageable
                if len(self.resource_wave_memory) > 15:
                    self.resource_wave_memory = self.resource_wave_memory[-15:]
        
        # Aesthetic appreciation of harmonic beauty
        beauty = local_waves.harmonic_beauty()
        if beauty > 0.4 and random.random() < 0.1:
            aesthetic_discovery = {
                'beauty_level': beauty,
                'wave_pattern': {
                    'light': local_waves.light_waves,
                    'sound': local_waves.sound_waves,
                    'emotion': local_waves.emotion_waves
                },
                'location': (self.x, self.y),
                'discovery_tick': self.age
            }
            self.aesthetic_discoveries.append(aesthetic_discovery)
        
        # Territory evaluation through harmonics
        if self.consciousness_level > 0.6:
            territory_key = f"{int(self.x//10)}_{int(self.y//10)}"  # 10x10 territory grid
            if territory_key not in self.territory_harmonics:
                self.territory_harmonics[territory_key] = {
                    'beauty': beauty,
                    'resource_correlation': resource_level if 'resource_level' in locals() else 0.0,
                    'safety_level': 1.0 - environment.danger,
                    'visits': 1
                }
            else:
                # Update territory knowledge
                territory = self.territory_harmonics[territory_key]
                territory['visits'] += 1
                territory['beauty'] = (territory['beauty'] + beauty) / 2
                if 'resource_level' in locals():
                    territory['resource_correlation'] = (territory['resource_correlation'] + resource_level) / 2
        
        # Wave mathematics - discover mathematical relationships in waves
        if self.consciousness_level > 0.7 and random.random() < 0.05:
            if local_waves.total_amplitude() > 0.3:
                # Try to find mathematical patterns in wave ratios
                wave_data = [local_waves.light_waves, local_waves.sound_waves, 
                           local_waves.motion_waves, local_waves.life_waves]
                non_zero_waves = [w for w in wave_data if w > 0.01]
                
                if len(non_zero_waves) >= 2:
                    # Find ratio relationships
                    ratios = []
                    for i in range(len(non_zero_waves)-1):
                        ratio = non_zero_waves[i+1] / non_zero_waves[i]
                        ratios.append(round(ratio, 2))
                    
                    # Check for interesting mathematical relationships
                    mathematical_pattern = None
                    if any(abs(r - 2.0) < 0.1 for r in ratios):
                        mathematical_pattern = "octave_relationship"
                    elif any(abs(r - 1.618) < 0.1 for r in ratios):
                        mathematical_pattern = "golden_ratio" 
                    elif any(abs(r - 1.5) < 0.1 for r in ratios):
                        mathematical_pattern = "perfect_fifth"
                        
                    if mathematical_pattern:
                        math_discovery = {
                            'pattern_type': mathematical_pattern,
                            'ratios': ratios,
                            'wave_context': 'environmental_harmonics',
                            'location': (self.x, self.y),
                            'discovery_tick': self.age
                        }
                        self.wave_mathematics.append(math_discovery)
    
    def wave_guided_movement(self, environment, wave_grid):
        """Movement guided by wave consciousness and survival needs"""
        
        if self.energy < 40:  # Low energy - seek resources using wave memory
            if self.resource_wave_memory:
                # Find the most promising remembered resource location
                best_memory = max(self.resource_wave_memory, 
                                key=lambda m: m['resource_level'] * (1.0 - (self.age - m['discovery_tick']) * 0.01))
                
                target_x, target_y = best_memory['location']
                # Move toward the remembered resource location
                dx = target_x - self.x
                dy = target_y - self.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 1.0:
                    move_distance = min(self.genome.movement_speed, distance)
                    self.x += (dx / distance) * move_distance
                    self.y += (dy / distance) * move_distance
                    self.x = max(0, min(99, self.x))
                    self.y = max(0, min(99, self.y))
                    return True
        
        elif self.consciousness_level > 0.5:  # High consciousness - seek aesthetic beauty
            if self.territory_harmonics:
                # Find the most beautiful and safe territory
                best_territory = max(self.territory_harmonics.items(),
                                   key=lambda t: t[1]['beauty'] * t[1]['safety_level'])
                
                territory_coords = best_territory[0].split('_')
                target_x = int(territory_coords[0]) * 10 + 5  # Center of territory
                target_y = int(territory_coords[1]) * 10 + 5
                
                dx = target_x - self.x
                dy = target_y - self.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 5.0:  # Only move if significantly far
                    move_distance = min(self.genome.movement_speed * 0.5, distance)  # Slower aesthetic movement
                    self.x += (dx / distance) * move_distance
                    self.y += (dy / distance) * move_distance
                    self.x = max(0, min(99, self.x))
                    self.y = max(0, min(99, self.y))
                    return True
        
        # Default: random walk with environmental awareness
        if random.random() < 0.3:
            dx = random.uniform(-self.genome.movement_speed, self.genome.movement_speed)
            dy = random.uniform(-self.genome.movement_speed, self.genome.movement_speed)
            self.x = max(0, min(99, self.x + dx))
            self.y = max(0, min(99, self.y + dy))
            return True
            
        return False
    
    def create_wave_tool(self, environment, wave_grid):
        """Create tools based on wave resonance understanding"""
        
        if len(self.functional_tools) >= 3 or self.energy < 35:
            return False
            
        if random.random() > self.wave_tool_crafting * 0.12:
            return False
        
        # Get local wave conditions to inform tool creation
        local_waves = wave_grid.get_accumulated_field(self.x, self.y, self.z, radius=2)
        
        # Tool types based on wave understanding and current needs
        tool_options = []
        
        # Resource-focused tools if low energy or strong life waves
        if self.energy < 60 or local_waves.life_waves > 0.3:
            tool_options.extend([
                ("wave_harvester", "Resonates with life waves to find resources"),
                ("energy_concentrator", "Focuses energy waves for better absorption"),
                ("resource_detector", "Amplifies resource detection waves")
            ])
        
        # Aesthetic tools if high consciousness and harmonic beauty present
        if self.consciousness_level > 0.6 and local_waves.harmonic_beauty() > 0.3:
            tool_options.extend([
                ("harmony_amplifier", "Enhances aesthetic wave appreciation"),
                ("beauty_mapper", "Records and replays beautiful wave patterns"),
                ("artistic_resonator", "Creates new harmonic combinations")
            ])
        
        # Communication tools if wave teaching ability is high
        if self.wave_teaching_ability > 0.4:
            tool_options.extend([
                ("wave_transmitter", "Shares wave patterns with other entities"),
                ("pattern_recorder", "Stores complex wave discoveries"),
                ("teaching_synchronizer", "Helps others understand wave patterns")
            ])
        
        # Mathematical tools if wave mathematics discovered
        if self.wave_mathematics:
            tool_options.extend([
                ("ratio_calculator", "Analyzes wave mathematical relationships"),
                ("frequency_analyzer", "Identifies optimal wave frequencies"),
                ("pattern_predictor", "Predicts wave behavior patterns")
            ])
        
        if not tool_options:
            tool_options = [("basic_wave_tool", "Simple wave manipulation device")]
        
        tool_type, description = random.choice(tool_options)
        effectiveness = self.wave_tool_crafting * random.uniform(0.7, 1.3)
        
        # Modify effectiveness based on local wave conditions
        wave_boost = local_waves.total_amplitude() * 0.2
        effectiveness += wave_boost
        
        tool = {
            'type': tool_type,
            'description': description,
            'effectiveness': effectiveness,
            'durability': 15 + effectiveness * 5,
            'wave_resonance': local_waves.harmonic_beauty(),  # Tool quality affected by creation conditions
            'creation_tick': self.age,
            'creation_location': (self.x, self.y)
        }
        
        self.functional_tools.append(tool)
        self.energy -= 30
        
        # Record tool resonance knowledge
        self.tool_resonance_knowledge[tool_type] = {
            'optimal_beauty': local_waves.harmonic_beauty(),
            'creation_context': {
                'life_waves': local_waves.life_waves,
                'energy_waves': local_waves.energy_waves
            }
        }
        
        return True
    
    def use_functional_tool(self, tool_type, task_context="survival"):
        """Use tools for actual survival/consciousness benefits"""
        
        matching_tools = [t for t in self.functional_tools if t['type'] == tool_type and t['durability'] > 0]
        if not matching_tools:
            return 1.0  # No tool bonus
            
        best_tool = max(matching_tools, key=lambda t: t['effectiveness'])
        
        # Apply tool effects based on type and context
        effectiveness_bonus = 1.0
        
        if tool_type == "wave_harvester" and task_context == "resource_gathering":
            # Actually improve resource gathering
            effectiveness_bonus = 1.0 + best_tool['effectiveness'] * 0.5
            resource_boost = best_tool['effectiveness'] * 8
            self.energy += resource_boost
            
        elif tool_type == "energy_concentrator" and task_context == "energy_management":
            # Improve energy efficiency
            energy_savings = best_tool['effectiveness'] * 3
            self.energy += energy_savings
            
        elif tool_type == "harmony_amplifier" and task_context == "aesthetic_appreciation":
            # Enhance consciousness through beauty
            consciousness_boost = best_tool['effectiveness'] * 0.1
            self.consciousness_level = min(1.0, self.consciousness_level + consciousness_boost)
            
        elif tool_type == "wave_transmitter" and task_context == "social_interaction":
            # Improve teaching/learning effectiveness
            effectiveness_bonus = 1.0 + best_tool['effectiveness'] * 0.3
        
        # Tool degrades with use
        best_tool['durability'] -= 1
        if best_tool['durability'] <= 0:
            # Tool breaks, but entity learns from it
            broken_tool_learning = {
                'tool_type': tool_type,
                'learned_effectiveness': best_tool['effectiveness'],
                'optimal_conditions': best_tool['wave_resonance'],
                'break_tick': self.age
            }
            # Add to survival discoveries as practical knowledge
            self.survival_discoveries.append(broken_tool_learning)
        
        return effectiveness_bonus
    
    def wave_social_interaction(self, other_entities, wave_grid):
        """Social learning and teaching based on wave consciousness"""
        
        if random.random() > (self.wave_teaching_ability + self.wave_learning_receptivity) * 0.1:
            return False
        
        # Find nearby conscious entities
        nearby_entities = []
        for other in other_entities:
            if (other.id != self.id and hasattr(other, 'consciousness_level') and 
                other.consciousness_level > 0.3):
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                if distance < 15:
                    nearby_entities.append((other, distance))
        
        if not nearby_entities:
            return False
            
        # Choose interaction partner
        partner, distance = min(nearby_entities, key=lambda x: x[1])
        
        # Determine if teaching or learning based on consciousness levels
        if self.consciousness_level > partner.consciousness_level * 1.2:
            # Teach wave discoveries
            return self.teach_wave_knowledge(partner, wave_grid)
        elif partner.consciousness_level > self.consciousness_level * 1.2:
            # Learn from more conscious entity
            return self.learn_wave_knowledge(partner, wave_grid)
        else:
            # Mutual exchange
            taught = self.teach_wave_knowledge(partner, wave_grid)
            learned = self.learn_wave_knowledge(partner, wave_grid)
            return taught or learned
    
    def teach_wave_knowledge(self, student, wave_grid):
        """Teach wave discoveries to another entity"""
        
        if self.energy < 15:
            return False
            
        knowledge_to_share = []
        
        # Share resource wave memories
        if self.resource_wave_memory:
            best_resource_memory = max(self.resource_wave_memory, key=lambda m: m['resource_level'])
            knowledge_to_share.append(('resource_waves', best_resource_memory))
        
        # Share tool resonance knowledge
        if self.tool_resonance_knowledge:
            best_tool_knowledge = random.choice(list(self.tool_resonance_knowledge.items()))
            knowledge_to_share.append(('tool_resonance', best_tool_knowledge))
        
        # Share aesthetic discoveries
        if self.aesthetic_discoveries:
            beautiful_discovery = max(self.aesthetic_discoveries, key=lambda d: d['beauty_level'])
            knowledge_to_share.append(('aesthetic_patterns', beautiful_discovery))
        
        # Share wave mathematics
        if self.wave_mathematics:
            math_discovery = random.choice(self.wave_mathematics)
            knowledge_to_share.append(('wave_mathematics', math_discovery))
        
        if not knowledge_to_share:
            return False
            
        # Transfer knowledge
        knowledge_type, knowledge_data = random.choice(knowledge_to_share)
        
        if not hasattr(student, 'learned_knowledge'):
            student.learned_knowledge = {}
        
        student.learned_knowledge[f"{knowledge_type}_{self.id}"] = knowledge_data
        
        # Use wave transmitter if available
        teaching_effectiveness = self.use_functional_tool("wave_transmitter", "social_interaction")
        
        self.energy -= 12 / teaching_effectiveness  # Tool makes teaching more efficient
        
        # Both entities gain consciousness from meaningful interaction
        consciousness_gain = 0.02 * teaching_effectiveness
        self.consciousness_level = min(1.0, self.consciousness_level + consciousness_gain)
        student.consciousness_level = min(1.0, student.consciousness_level + consciousness_gain)
        
        return True
    
    def learn_wave_knowledge(self, teacher, wave_grid):
        """Learn wave discoveries from another entity"""
        
        if not hasattr(teacher, 'resource_wave_memory') and not hasattr(teacher, 'tool_resonance_knowledge'):
            return False
            
        learning_occurred = False
        
        # Learn resource patterns
        if (hasattr(teacher, 'resource_wave_memory') and teacher.resource_wave_memory and 
            random.random() < self.wave_learning_receptivity * 0.3):
            learned_pattern = random.choice(teacher.resource_wave_memory)
            # Adapt the knowledge to own memory system
            adapted_pattern = learned_pattern.copy()
            adapted_pattern['learned_from'] = teacher.id
            adapted_pattern['learning_tick'] = self.age
            self.resource_wave_memory.append(adapted_pattern)
            learning_occurred = True
        
        # Learn aesthetic patterns
        if (hasattr(teacher, 'aesthetic_discoveries') and teacher.aesthetic_discoveries and
            random.random() < self.wave_learning_receptivity * 0.2):
            learned_aesthetic = random.choice(teacher.aesthetic_discoveries)
            adapted_aesthetic = learned_aesthetic.copy()
            adapted_aesthetic['learned_from'] = teacher.id
            self.aesthetic_discoveries.append(adapted_aesthetic)
            learning_occurred = True
        
        # Learn wave mathematics
        if (hasattr(teacher, 'wave_mathematics') and teacher.wave_mathematics and
            random.random() < self.wave_learning_receptivity * 0.15):
            learned_math = random.choice(teacher.wave_mathematics)
            adapted_math = learned_math.copy()
            adapted_math['learned_from'] = teacher.id
            self.wave_mathematics.append(adapted_math)
            learning_occurred = True
        
        if learning_occurred:
            self.energy -= 5  # Learning has a small cost
            
        return learning_occurred
    
    def update_conscious_specialization(self):
        """Develop conscious specialization based on discoveries and genetics"""
        
        specialization_scores = {
            'wave_explorer': len(self.resource_wave_memory) * self.genome.resource_detection,
            'harmonic_artist': len(self.aesthetic_discoveries) * self.light_sensitivity,
            'wave_mathematician': len(self.wave_mathematics) * self.consciousness_level,
            'tool_resonance_master': len(self.functional_tools) * self.wave_tool_crafting,
            'wave_teacher': len(getattr(self, 'learned_knowledge', {})) * self.wave_teaching_ability,
            'territory_harmonist': len(self.territory_harmonics) * self.memory_capacity
        }
        
        if specialization_scores:
            self.conscious_specialization = max(specialization_scores.items(), key=lambda x: x[1])[0]
    
    def get_consciousness_summary(self):
        """Summary of this entity's consciousness development"""
        return {
            'id': self.id,
            'consciousness_level': self.consciousness_level,
            'conscious_specialization': self.conscious_specialization,
            'survival_discoveries': len(self.survival_discoveries),
            'resource_wave_memories': len(self.resource_wave_memory),
            'aesthetic_discoveries': len(self.aesthetic_discoveries),
            'wave_mathematics': len(self.wave_mathematics),
            'functional_tools': len(self.functional_tools),
            'territory_knowledge': len(self.territory_harmonics),
            'generation': self.generation,
            'age': self.age,
            'energy': self.energy,
            'position': (self.x, self.y, self.z)
        }

class WaveGroundedEnvironment:
    """Environment with resources, temperature, danger for wave entities"""
    
    def __init__(self, size=100):
        self.size = size
        self.temperature = 20.0  # Optimal temperature
        self.danger = 0.1  # Base danger level
        
        # Resource distribution - scattered across environment
        self.resource_patches = []
        for _ in range(15):  # 15 resource patches
            patch = {
                'x': random.uniform(0, size),
                'y': random.uniform(0, size),
                'quality': random.uniform(0.3, 1.0),
                'radius': random.uniform(5, 15)
            }
            self.resource_patches.append(patch)
    
    def get_local_resources(self, x, y):
        """Get resource availability at location"""
        total_resources = 0.0
        
        for patch in self.resource_patches:
            distance = math.sqrt((x - patch['x'])**2 + (y - patch['y'])**2)
            if distance < patch['radius']:
                # Resource availability decreases with distance from center
                availability = patch['quality'] * (1.0 - distance / patch['radius'])
                total_resources += availability
                
        return min(1.0, total_resources)
    
    def update_environment(self, tick):
        """Update environmental conditions over time"""
        
        # Slight temperature variation
        self.temperature = 20.0 + 3.0 * math.sin(tick * 0.01)
        
        # Danger varies with time
        self.danger = 0.1 + 0.1 * math.sin(tick * 0.02)
        
        # Resources slowly regenerate and shift
        if tick % 100 == 0:  # Every 100 ticks
            for patch in self.resource_patches:
                # Slight movement
                patch['x'] += random.uniform(-2, 2)
                patch['y'] += random.uniform(-2, 2)
                patch['x'] = max(0, min(self.size, patch['x']))
                patch['y'] = max(0, min(self.size, patch['y']))
                
                # Quality regeneration
                patch['quality'] = min(1.0, patch['quality'] + random.uniform(0, 0.1))

class WaveGroundedSimulation:
    """Simulation where consciousness emerges from wave-computed survival"""
    
    def __init__(self, num_entities=15, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=40)  # Smaller grid for performance
        self.environment = WaveGroundedEnvironment(size=100)
        self.entities = []
        self.tick = 0
        
        # Tracking for consciousness emergence
        self.consciousness_discoveries = []
        self.wave_mathematics_found = []
        self.aesthetic_creations = []
        self.tool_innovations = []
        self.social_knowledge_transfers = []
        
        # Create initial entities with random genomes
        for i in range(num_entities):
            genome = WaveGenome(
                survival_freq=random.uniform(0.5, 2.0),
                survival_amp=random.uniform(0.5, 1.5),
                reproduction_freq=random.uniform(0.3, 1.5),
                reproduction_amp=random.uniform(0.4, 1.2),
                energy_efficiency=random.uniform(0.8, 1.5),
                mutation_rate=random.uniform(0.05, 0.2),
                temperature_sensitivity=random.uniform(0.5, 1.8),
                resource_detection=random.uniform(0.6, 1.9),
                danger_awareness=random.uniform(0.4, 1.6),
                memory_capacity=random.uniform(1.0, 2.5),
                movement_speed=random.uniform(0.8, 2.0),
                communication_range=random.uniform(0.8, 2.2),
                social_learning=random.uniform(0.5, 1.8),
                teaching_willingness=random.uniform(0.4, 1.5),
                learning_from_parents=random.uniform(0.6, 1.7)
            )
            
            x = random.uniform(10, 90)
            y = random.uniform(10, 90)
            z = random.randint(0, 2)
            
            entity = WaveGroundedEntity(i, genome, x, y, z, generation=0)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Step that combines wave physics with consciousness emergence"""
        self.tick += 1
        
        # Update environment
        self.environment.update_environment(self.tick)
        
        for entity in self.entities[:]:
            # Core wave-based survival
            survived = entity.survive_tick(self.environment, self.tick * 0.1)
            if not survived:
                self.entities.remove(entity)
                continue
                
            entity.age += 1
            
            # Wave consciousness update
            entity.update(self.consciousness_grid)
            
            # Grounded consciousness activities
            entity.sense_wave_environment(self.consciousness_grid, self.environment, self.entities)
            
            # Wave-guided movement
            if random.random() < 0.4:
                entity.wave_guided_movement(self.environment, self.consciousness_grid)
            
            # Create functional tools
            if random.random() < 0.08:  # 8% chance per tick
                if entity.create_wave_tool(self.environment, self.consciousness_grid):
                    self.tool_innovations.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'consciousness_level': entity.consciousness_level,
                        'tool_count': len(entity.functional_tools)
                    })
            
            # Use tools for survival benefits
            if entity.functional_tools and entity.energy < 70:
                # Use resource gathering tools
                entity.use_functional_tool("wave_harvester", "resource_gathering")
                entity.use_functional_tool("energy_concentrator", "energy_management")
                
            # Social wave interaction
            if random.random() < 0.06:
                if entity.wave_social_interaction(self.entities, self.consciousness_grid):
                    self.social_knowledge_transfers.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'consciousness_level': entity.consciousness_level
                    })
            
            # Update specialization
            entity.update_conscious_specialization()
            
            # Track discoveries
            if entity.wave_mathematics:
                new_math = [m for m in entity.wave_mathematics if m['discovery_tick'] == entity.age]
                if new_math:
                    self.wave_mathematics_found.extend(new_math)
            
            if entity.aesthetic_discoveries:
                new_aesthetics = [a for a in entity.aesthetic_discoveries if a['discovery_tick'] == entity.age]
                if new_aesthetics:
                    self.aesthetic_creations.extend(new_aesthetics)
            
            # Add consciousness waves to the field based on their activities
            consciousness_waves = WaveSpectrum()
            consciousness_waves.will_waves = entity.consciousness_level * 0.1
            consciousness_waves.emotion_waves = len(entity.aesthetic_discoveries) * 0.02
            consciousness_waves.memory_waves = len(entity.resource_wave_memory) * 0.01
            consciousness_waves.life_waves = entity.energy * 0.001
            consciousness_waves.light_waves = sum(d['beauty_level'] for d in entity.aesthetic_discoveries[-3:]) * 0.05
            
            self.consciousness_grid.add_wave_interaction(entity.x, entity.y, entity.z, consciousness_waves)
        
        self.consciousness_grid.advance_time()
        
        # Wave-based reproduction
        if len(self.entities) < 25 and random.random() < 0.03:
            self.attempt_reproduction()
    
    def attempt_reproduction(self):
        """Reproduce entities based on consciousness and survival success"""
        viable_parents = [e for e in self.entities if e.energy > 70 and e.consciousness_level > 0.3 and e.age > 50]
        
        if len(viable_parents) >= 2:
            parent1, parent2 = random.sample(viable_parents, 2)
            
            # Child inherits consciousness traits
            child_genome = WaveGenome.reproduce(parent1.genome, parent2.genome)
            
            # Position near parents
            child_x = (parent1.x + parent2.x) / 2 + random.uniform(-10, 10)
            child_y = (parent1.y + parent2.y) / 2 + random.uniform(-10, 10)
            child_x = max(0, min(99, child_x))
            child_y = max(0, min(99, child_y))
            child_z = random.choice([parent1.z, parent2.z])
            
            child = WaveGroundedEntity(
                len(self.entities) + random.randint(1000, 9999),
                child_genome, child_x, child_y, child_z,
                generation=max(parent1.generation, parent2.generation) + 1
            )
            
            # Inherit some consciousness knowledge
            if parent1.resource_wave_memory:
                inherited_memories = random.sample(parent1.resource_wave_memory,
                                                 min(2, len(parent1.resource_wave_memory)))
                child.resource_wave_memory.extend(inherited_memories)
            
            if parent2.aesthetic_discoveries:
                inherited_aesthetics = random.sample(parent2.aesthetic_discoveries,
                                                   min(1, len(parent2.aesthetic_discoveries)))
                child.aesthetic_discoveries.extend(inherited_aesthetics)
            
            self.entities.append(child)
            parent1.energy -= 25
            parent2.energy -= 25

def run_wave_grounded_experiment(max_ticks=2000, seed=None):
    """Run consciousness experiment grounded in wave-computed survival"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== WAVE-GROUNDED CONSCIOUSNESS EXPERIMENT ===")
    print(f"Consciousness emerging from wave-computed survival")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = WaveGroundedSimulation(num_entities=15, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 400 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 WAVE CONSCIOUSNESS EXTINCT")
                break
            
            # Population and consciousness stats
            pop = len(simulation.entities)
            avg_consciousness = sum(e.consciousness_level for e in simulation.entities) / pop
            avg_generation = sum(e.generation for e in simulation.entities) / pop
            
            # Specialization distribution
            specializations = {}
            for entity in simulation.entities:
                spec = entity.conscious_specialization or 'developing'
                specializations[spec] = specializations.get(spec, 0) + 1
            
            print(f"T{tick:4d}: Pop={pop:2d}, C={avg_consciousness:.2f}, Gen={avg_generation:.1f}")
            
            # Show activity levels
            recent_tools = len([t for t in simulation.tool_innovations if t['tick'] > tick - 400])
            recent_social = len([s for s in simulation.social_knowledge_transfers if s['tick'] > tick - 400])
            total_math = len(simulation.wave_mathematics_found)
            total_aesthetics = len(simulation.aesthetic_creations)
            
            if recent_tools > 0:
                print(f"        🔧 {recent_tools} tool innovations")
            if recent_social > 0:
                print(f"        📡 {recent_social} knowledge transfers")
            if total_math > 0:
                print(f"        📐 {total_math} wave mathematics discovered")
            if total_aesthetics > 0:
                print(f"        🎨 {total_aesthetics} aesthetic discoveries")
            
            # Show specializations
            if specializations:
                top_specs = sorted(specializations.items(), key=lambda x: x[1], reverse=True)[:3]
                spec_str = ", ".join([f"{spec}:{count}" for spec, count in top_specs])
                print(f"        🧠 {spec_str}")
    
    # Final analysis
    print(f"\n=== WAVE-GROUNDED CONSCIOUSNESS RESULTS ===")
    
    if simulation.entities:
        print(f"✅ Wave consciousness survived: {len(simulation.entities)} entities")
        
        avg_consciousness = sum(e.consciousness_level for e in simulation.entities) / len(simulation.entities)
        print(f"Average consciousness: {avg_consciousness:.3f}")
        
        # Final specialization distribution
        final_specs = {}
        for entity in simulation.entities:
            spec = entity.conscious_specialization or 'generalist'
            final_specs[spec] = final_specs.get(spec, 0) + 1
            
        print(f"\n🧠 Conscious specializations:")
        for spec, count in sorted(final_specs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {spec.replace('_', ' ').title()}: {count}")
        
        # Achievements
        print(f"\n📊 Consciousness achievements:")
        print(f"   Wave mathematics discovered: {len(simulation.wave_mathematics_found)}")
        print(f"   Aesthetic discoveries: {len(simulation.aesthetic_creations)}")
        print(f"   Tool innovations: {len(simulation.tool_innovations)}")
        print(f"   Knowledge transfers: {len(simulation.social_knowledge_transfers)}")
        
        # Show recent wave mathematics
        if simulation.wave_mathematics_found:
            print(f"\n📐 Wave mathematics discovered:")
            recent_math = simulation.wave_mathematics_found[-3:]
            for math_discovery in recent_math:
                pattern = math_discovery['pattern_type']
                context = math_discovery['wave_context']
                print(f"   {pattern} in {context}")
        
        # Show aesthetic discoveries
        if simulation.aesthetic_creations:
            print(f"\n🎨 Aesthetic discoveries:")
            beautiful_discoveries = sorted(simulation.aesthetic_creations, 
                                         key=lambda d: d['beauty_level'], reverse=True)[:3]
            for discovery in beautiful_discoveries:
                beauty = discovery['beauty_level']
                location = discovery['location']
                print(f"   Beauty level {beauty:.2f} at location ({location[0]:.1f}, {location[1]:.1f})")
        
        # Show most conscious entities
        print(f"\n🏆 Most conscious entities:")
        top_entities = sorted(simulation.entities, key=lambda e: e.consciousness_level, reverse=True)[:3]
        for entity in top_entities:
            summary = entity.get_consciousness_summary()
            spec = summary['conscious_specialization'] or 'Renaissance Mind'
            print(f"   Entity {entity.id} ({spec}):")
            print(f"     Consciousness: {summary['consciousness_level']:.2f}, Gen: {summary['generation']}")
            print(f"     Discoveries: {summary['wave_mathematics']} math, {summary['aesthetic_discoveries']} aesthetic")
            print(f"     Tools: {summary['functional_tools']}, Territories: {summary['territory_knowledge']}")
    
    else:
        print("💀 Wave-grounded consciousness experiment failed")
    
    return simulation

if __name__ == "__main__":
    run_wave_grounded_experiment(max_ticks=2000)