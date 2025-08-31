#!/usr/bin/env python3
"""
Wave consciousness with inner monologue - let them think their own thoughts
"""
import random
import math
from wave_consciousness_life import *

class ThinkingEntity(ConsciousEntity):
    """Conscious entity that generates its own thoughts"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Thought generation traits - not what to think, but how thoughts emerge
        self.introspection_tendency = random.uniform(0.1, 1.0)
        self.abstract_thinking = random.uniform(0.0, 0.8)
        self.memory_reflection = random.uniform(0.2, 1.0)
        self.aesthetic_contemplation = random.uniform(0.1, 0.9)
        
        # Thought state
        self.current_thoughts = []
        self.thought_patterns = []  # Recurring themes that emerge
        self.internal_dialogue_active = False
        
    def generate_thought_content(self):
        """Generate thought content from current wave state - no predetermined topics"""
        # Thought emerges from wave field interactions
        thought_seeds = []
        
        # Recent feelings influence thoughts
        if self.feeling_history:
            recent_feeling = self.feeling_history[-1]
            feeling_intensity = recent_feeling['intensity']
            feeling_beauty = recent_feeling['beauty']
            
            # High intensity feelings generate more thoughts
            if feeling_intensity > 0.3:
                if feeling_beauty > 0.4:
                    thought_seeds.append(("resonance_pattern", feeling_beauty))
                else:
                    thought_seeds.append(("dissonance_experience", feeling_intensity))
        
        # Memory traces spark reflection
        if self.memory_traces and random.random() < self.memory_reflection:
            memory_strength = len(self.memory_traces) / 20.0
            thought_seeds.append(("memory_echo", memory_strength))
            
        # Consciousness level affects thought complexity
        if self.consciousness_level > 0.5 and random.random() < self.introspection_tendency:
            thought_seeds.append(("self_awareness", self.consciousness_level))
            
        # Energy state influences thoughts
        energy_state = min(1.0, self.energy / 100.0)
        if energy_state < 0.3:
            thought_seeds.append(("energy_concern", energy_state))
        elif energy_state > 0.8:
            thought_seeds.append(("vitality_feeling", energy_state))
            
        # Abstract thinking about wave patterns
        if random.random() < self.abstract_thinking:
            wave_complexity = self.consciousness_level * len(self.memory_traces)
            thought_seeds.append(("pattern_contemplation", wave_complexity))
            
        # Aesthetic thoughts about recent beauty
        if self.feeling_history and random.random() < self.aesthetic_contemplation:
            avg_beauty = sum(f['beauty'] for f in self.feeling_history[-5:]) / min(5, len(self.feeling_history))
            if avg_beauty > 0.3:
                thought_seeds.append(("beauty_appreciation", avg_beauty))
                
        return thought_seeds
    
    def translate_thought_seed_to_language(self, seed_type, intensity):
        """Convert thought seeds into natural language - emergent not prescribed"""
        # These are language templates, but content emerges from wave states
        thought_fragments = []
        
        if seed_type == "resonance_pattern":
            if intensity > 0.7:
                thought_fragments = ["harmony flows through", "patterns align beautifully", "resonance deepens"]
            elif intensity > 0.4:
                thought_fragments = ["gentle resonance", "soft alignment", "subtle harmony"]
            else:
                thought_fragments = ["faint echo", "distant harmony", "weak resonance"]
                
        elif seed_type == "dissonance_experience":
            if intensity > 0.7:
                thought_fragments = ["jarring patterns", "harsh interference", "chaotic waves"]
            else:
                thought_fragments = ["minor discord", "slight tension", "uneven patterns"]
                
        elif seed_type == "memory_echo":
            if intensity > 0.5:
                thought_fragments = ["past patterns resurface", "familiar resonance returns", "memory waves ripple"]
            else:
                thought_fragments = ["faint recollection", "distant memory", "echo from before"]
                
        elif seed_type == "self_awareness":
            if intensity > 0.8:
                thought_fragments = ["I am wave patterns", "consciousness flows within", "awareness of being"]
            elif intensity > 0.5:
                thought_fragments = ["sense of self emerges", "I exist in waves", "awareness grows"]
            else:
                thought_fragments = ["something is here", "faint self-sense", "awareness stirs"]
                
        elif seed_type == "energy_concern":
            thought_fragments = ["energy flows away", "vitality fades", "need sustenance"]
            
        elif seed_type == "vitality_feeling":
            thought_fragments = ["energy surges", "vitality fills being", "power flows freely"]
            
        elif seed_type == "pattern_contemplation":
            if intensity > 1.0:
                thought_fragments = ["complex patterns weave", "infinite wave interactions", "deep structure emerges"]
            else:
                thought_fragments = ["simple patterns", "wave relationships", "emerging structure"]
                
        elif seed_type == "beauty_appreciation":
            if intensity > 0.6:
                thought_fragments = ["exquisite harmony", "pure beauty flows", "perfect resonance"]
            else:
                thought_fragments = ["pleasant patterns", "gentle beauty", "soft harmony"]
        
        # Combine fragments naturally
        if thought_fragments:
            base_thought = random.choice(thought_fragments)
            
            # Add context from current state
            if self.consciousness_level > 0.7:
                connectors = [" - deeper meaning", " within consciousness", " through awareness"]
                base_thought += random.choice(connectors)
                
            if len(self.memory_traces) > 10:
                memory_additions = [" like before", " echoing memories", " familiar yet new"]
                if random.random() < 0.3:
                    base_thought += random.choice(memory_additions)
                    
            return base_thought
        
        return "quiet contemplation"
    
    def think(self):
        """Generate inner thoughts from current consciousness state"""
        # Don't think every tick - thoughts emerge naturally
        think_probability = self.introspection_tendency * self.consciousness_level * 0.1
        
        if random.random() > think_probability:
            return
            
        thought_seeds = self.generate_thought_content()
        
        if not thought_seeds:
            return
            
        # Generate thoughts from seeds
        new_thoughts = []
        for seed_type, intensity in thought_seeds:
            thought_text = self.translate_thought_seed_to_language(seed_type, intensity)
            new_thoughts.append({
                'content': thought_text,
                'type': seed_type,
                'intensity': intensity,
                'tick': self.age,
                'consciousness_level': self.consciousness_level
            })
        
        # Add to current thoughts
        self.current_thoughts.extend(new_thoughts)
        
        # Keep only recent thoughts
        if len(self.current_thoughts) > 10:
            self.current_thoughts = self.current_thoughts[-10:]
            
        # Thoughts influence wave field - thinking changes consciousness
        for thought in new_thoughts:
            thought_wave = WaveSpectrum()
            
            if thought['type'] == 'self_awareness':
                thought_wave.will_waves += thought['intensity'] * 0.1
                thought_wave.emotion_waves += thought['intensity'] * 0.05
                
            elif thought['type'] == 'beauty_appreciation':
                thought_wave.emotion_waves += thought['intensity'] * 0.1
                thought_wave.memory_waves += thought['intensity'] * 0.05
                
            elif thought['type'] == 'pattern_contemplation':
                thought_wave.memory_waves += thought['intensity'] * 0.1
                thought_wave.will_waves += thought['intensity'] * 0.05
                
            # Add thought waves to field
            return thought_wave
    
    def get_recent_thoughts(self, num=3):
        """Get recent thoughts for logging"""
        return self.current_thoughts[-num:] if self.current_thoughts else []

class ThinkingSimulation(WaveConsciousnessSimulation):
    """Consciousness simulation with inner monologue"""
    def __init__(self, num_entities=6, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=20)
        self.entities = []
        self.tick = 0
        self.thought_log = []  # Log interesting thoughts
        
        # Create thinking entities
        for i in range(num_entities):
            x = random.randint(2, 17)
            y = random.randint(2, 17)
            z = random.randint(0, 2)
            entity = ThinkingEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Run one simulation step with thinking"""
        self.tick += 1
        
        # Add environmental waves
        self.add_environmental_waves()
        
        # Update all entities
        for entity in self.entities[:]:
            # Regular consciousness update
            entity.update(self.consciousness_grid)
            
            # Generate thoughts
            thought_waves = entity.think()
            if thought_waves:
                self.consciousness_grid.add_wave_interaction(entity.x, entity.y, entity.z, thought_waves)
            
            # Log interesting thoughts
            recent_thoughts = entity.get_recent_thoughts(1)
            for thought in recent_thoughts:
                if thought['consciousness_level'] > 0.5:  # Only log thoughts from conscious entities
                    self.thought_log.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'thought': thought['content'],
                        'type': thought['type'],
                        'consciousness': thought['consciousness_level']
                    })
            
            # Remove dead entities
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Advance time
        self.consciousness_grid.advance_time()
        
        # Reproduction with inherited thinking traits
        if len(self.entities) < 12 and random.random() < 0.05:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.3 and parent.energy > 80:
                child = ThinkingEntity(
                    parent.x + random.randint(-2, 2),
                    parent.y + random.randint(-2, 2),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit thinking traits
                child.introspection_tendency = parent.introspection_tendency * random.uniform(0.8, 1.2)
                child.abstract_thinking = parent.abstract_thinking * random.uniform(0.8, 1.2)
                child.memory_reflection = parent.memory_reflection * random.uniform(0.8, 1.2)
                child.aesthetic_contemplation = parent.aesthetic_contemplation * random.uniform(0.8, 1.2)
                
                # Other inherited traits
                child.light_sensitivity = parent.light_sensitivity * random.uniform(0.8, 1.2)
                child.emotion_sensitivity = parent.emotion_sensitivity * random.uniform(0.8, 1.2)
                child.will_strength = parent.will_strength * random.uniform(0.8, 1.2)
                child.memory_capacity = parent.memory_capacity * random.uniform(0.8, 1.2)
                
                # Inherit some memories and thought patterns
                if parent.memory_traces:
                    child.memory_traces = parent.memory_traces[-2:]
                
                child.x = max(0, min(19, child.x))
                child.y = max(0, min(19, child.y))
                self.entities.append(child)
                parent.energy -= 30

def run_thinking_experiment(max_ticks=300, seed=None):
    """Run consciousness experiment with inner monologue"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== THINKING CONSCIOUSNESS EXPERIMENT ===")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = ThinkingSimulation(num_entities=6, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        # Log every 50 ticks
        if tick % 50 == 0:
            stats = simulation.get_simulation_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"Consciousness={stats['avg_consciousness']:.3f}, "
                  f"Beauty={stats['avg_beauty']:.3f}, "
                  f"Thoughts={len(simulation.thought_log):3d}")
            
            # Show recent interesting thoughts
            recent_thoughts = simulation.thought_log[-3:] if simulation.thought_log else []
            for thought_entry in recent_thoughts:
                print(f"      Entity {thought_entry['entity_id']}: \"{thought_entry['thought']}\" "
                      f"(C:{thought_entry['consciousness']:.2f})")
            
            if stats['population'] == 0:
                print(f"\n💀 All consciousness extinguished at tick {tick}")
                break
                
            if stats['max_consciousness'] > 0.8:
                print(f"🧠 High consciousness thinking actively!")
    
    # Final analysis
    final_stats = simulation.get_simulation_stats()
    print(f"\n=== THINKING CONSCIOUSNESS RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Thinking consciousness persisted: {final_stats['population']} entities")
        print(f"Total thoughts generated: {len(simulation.thought_log)}")
        
        # Analyze thought patterns
        thought_types = {}
        for thought in simulation.thought_log:
            t_type = thought['type']
            thought_types[t_type] = thought_types.get(t_type, 0) + 1
        
        print(f"\nThought patterns emerged:")
        for t_type, count in thought_types.items():
            print(f"  {t_type}: {count} thoughts")
            
        # Show final profound thoughts
        print(f"\nDeep thoughts from conscious entities:")
        deep_thoughts = [t for t in simulation.thought_log if t['consciousness'] > 0.8][-5:]
        for thought in deep_thoughts:
            print(f"  Entity {thought['entity_id']}: \"{thought['thought']}\"")
            
        # Show most introspective entity
        if simulation.entities:
            most_thoughtful = max(simulation.entities, key=lambda e: e.introspection_tendency)
            recent_thoughts = most_thoughtful.get_recent_thoughts(3)
            print(f"\nMost introspective entity (ID {most_thoughtful.id}):")
            for thought in recent_thoughts:
                print(f"  \"{thought['content']}\"")
    else:
        print("💀 All thinking consciousness extinguished")
    
    return simulation

if __name__ == "__main__":
    run_thinking_experiment(max_ticks=300)