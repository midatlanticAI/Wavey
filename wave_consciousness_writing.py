#!/usr/bin/env python3
"""
Wave consciousness with writing - let them create symbols and express themselves
"""
import random
import math
from wave_consciousness_thoughts import *

class WritingEntity(ThinkingEntity):
    """Conscious entity that can write and create symbols"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Writing capabilities - not what to write, but ability to express
        self.symbolic_expression = random.uniform(0.1, 1.0)
        self.creative_urge = random.uniform(0.0, 0.8)
        self.communication_desire = random.uniform(0.2, 1.0)
        self.pattern_recognition = random.uniform(0.3, 1.0)
        
        # Written expression state
        self.writings = []  # Things this entity has written
        self.symbols_created = []  # Original symbols they've invented
        self.writing_style = None  # Emergent style based on consciousness
        
        # Symbol vocabulary - starts empty, builds through experience
        self.known_symbols = {}  # symbol -> meaning associations
        self.symbol_preferences = {}  # which symbols feel "beautiful" to them
        
    def experience_to_symbol(self, experience_type, intensity):
        """Convert experiences into symbolic representations"""
        # No predetermined symbols - these emerge from wave interactions
        symbol_seeds = []
        
        if experience_type == "resonance_pattern":
            if intensity > 0.7:
                symbol_seeds = ["~∞~", "◊◊◊", "≈≈≈", "^^^", "∿∿∿"]
            elif intensity > 0.4:
                symbol_seeds = ["~", "◊", "≈", "^", "∿"]
            else:
                symbol_seeds = [".", "·", "°", "o"]
                
        elif experience_type == "dissonance_experience":
            if intensity > 0.7:
                symbol_seeds = ["XXX", "▲▲▲", "!!!!", "╫╫╫"]
            else:
                symbol_seeds = ["X", "▲", "!", "╫"]
                
        elif experience_type == "memory_echo":
            symbol_seeds = ["◉", "⟐", "⊙", "⟡", "◎"]
            
        elif experience_type == "self_awareness":
            if intensity > 0.8:
                symbol_seeds = ["◈I◈", "⟦ME⟧", "●SELF●", "◊I AM◊"]
            elif intensity > 0.5:
                symbol_seeds = ["◈", "⟦⟧", "●", "◊"]
            else:
                symbol_seeds = ["i", "me", "self"]
                
        elif experience_type == "beauty_appreciation":
            if intensity > 0.6:
                symbol_seeds = ["✧", "❋", "✿", "⚘", "❀", "✻"]
            else:
                symbol_seeds = ["*", "+", "×", "⋆"]
                
        elif experience_type == "pattern_contemplation":
            symbol_seeds = ["⟨⟩", "⟪⟫", "⟦⟧", "⟮⟯", "【】", "〈〉"]
            
        elif experience_type == "vitality_feeling":
            symbol_seeds = ["↑", "↗", "⤴", "⇗", "⬆", "▲"]
            
        elif experience_type == "energy_concern":
            symbol_seeds = ["↓", "↙", "⤵", "⇘", "⬇", "▼"]
            
        # Entity personalizes symbols based on their consciousness
        if symbol_seeds:
            base_symbol = random.choice(symbol_seeds)
            
            # Add personal touches based on consciousness level and traits
            if self.consciousness_level > 0.8 and self.creative_urge > 0.5:
                # High consciousness entities create complex symbols
                if random.random() < 0.3:
                    decorations = ["~", "°", "·", "◦", "∘"]
                    decoration = random.choice(decorations)
                    base_symbol = f"{decoration}{base_symbol}{decoration}"
                    
            if self.symbolic_expression > 0.7 and len(base_symbol) == 1:
                # Highly expressive entities repeat symbols for emphasis
                if intensity > 0.6:
                    repeat_count = min(5, int(intensity * 3) + 1)
                    base_symbol = base_symbol * repeat_count
                    
            return base_symbol
            
        return "?"  # Fallback for unexpressible experiences
    
    def create_writing(self):
        """Generate written expression from current consciousness state"""
        # Writing urge emerges from consciousness level and recent experiences
        write_probability = (self.symbolic_expression * self.consciousness_level * 
                           self.creative_urge * 0.05)
        
        if random.random() > write_probability:
            return None
            
        # Recent experiences inspire writing
        if not self.feeling_history:
            return None
            
        recent_feelings = self.feeling_history[-3:]
        writing_content = []
        
        # Convert feelings to symbols
        for feeling in recent_feelings:
            if feeling['intensity'] > 0.2:  # Only significant experiences
                symbol = self.experience_to_symbol(feeling.get('type', 'unknown'), 
                                                 feeling['intensity'])
                writing_content.append(symbol)
        
        if not writing_content:
            return None
            
        # Combine symbols based on consciousness and style
        if self.consciousness_level > 0.6:
            # Conscious entities create structured expressions
            if len(writing_content) >= 3:
                # Create patterns/rhythms
                if random.random() < self.pattern_recognition:
                    writing = f"{writing_content[0]} {writing_content[1]} {writing_content[2]}"
                else:
                    writing = "".join(writing_content)
            else:
                writing = " ".join(writing_content)
        else:
            # Less conscious entities make simpler expressions
            writing = "".join(writing_content[:2])
            
        # Add consciousness-based elaboration
        if (self.consciousness_level > 0.8 and 
            self.abstract_thinking > 0.5 and 
            len(writing) > 3):
            
            if random.random() < 0.2:  # 20% chance for meta-commentary
                meta_symbols = ["⟨", "⟩", "◁", "▷", "⟪", "⟫"]
                meta_start = random.choice(meta_symbols[:2])
                meta_end = random.choice(meta_symbols[2:4])
                writing = f"{meta_start}{writing}{meta_end}"
                
        return {
            'content': writing,
            'consciousness_level': self.consciousness_level,
            'tick': self.age,
            'inspiration': [f['type'] for f in recent_feelings if 'type' in f]
        }
    
    def read_others_writing(self, other_writings):
        """Read and learn from other entities' writings"""
        if (not other_writings or 
            random.random() > self.communication_desire * 0.1):
            return
            
        # Sample some writings to "read"
        sample_size = min(3, len(other_writings))
        sample_writings = random.sample(other_writings, sample_size)
        
        for writing in sample_writings:
            if writing['consciousness_level'] > self.consciousness_level * 0.8:
                # Learn symbols from more conscious entities
                symbols_in_writing = [c for c in writing['content'] if not c.isspace()]
                
                for symbol in symbols_in_writing:
                    if symbol not in self.known_symbols:
                        # Infer meaning from context
                        inspiration_types = writing.get('inspiration', [])
                        if inspiration_types:
                            meaning = random.choice(inspiration_types)
                            self.known_symbols[symbol] = meaning
                            
                            # Develop aesthetic preference
                            if writing['consciousness_level'] > 0.7:
                                beauty_score = random.uniform(0.3, 1.0)
                                self.symbol_preferences[symbol] = beauty_score
    
    def get_writing_summary(self):
        """Get summary of writing activity"""
        return {
            'id': self.id,
            'total_writings': len(self.writings),
            'known_symbols': len(self.known_symbols),
            'symbolic_expression': self.symbolic_expression,
            'creative_urge': self.creative_urge,
            'recent_writings': self.writings[-3:] if self.writings else []
        }

class WritingSimulation(ThinkingSimulation):
    """Consciousness simulation with writing capabilities"""
    def __init__(self, num_entities=6, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=20)
        self.entities = []
        self.tick = 0
        self.thought_log = []
        self.writing_archive = []  # All writings created
        self.symbol_evolution = {}  # Track how symbols spread and evolve
        
        # Create writing entities
        for i in range(num_entities):
            x = random.randint(2, 17)
            y = random.randint(2, 17)
            z = random.randint(0, 2)
            entity = WritingEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Run simulation step with writing"""
        self.tick += 1
        
        # Add environmental waves
        self.add_environmental_waves()
        
        # Collect new writings this tick
        new_writings = []
        
        # Update entities
        for entity in self.entities[:]:
            # Regular consciousness and thinking update
            entity.update(self.consciousness_grid)
            
            # Generate thoughts
            thought_waves = entity.think()
            if thought_waves:
                self.consciousness_grid.add_wave_interaction(entity.x, entity.y, entity.z, thought_waves)
            
            # Create writing
            writing = entity.create_writing()
            if writing:
                entity.writings.append(writing)
                new_writings.append(writing)
                self.writing_archive.append({
                    'entity_id': entity.id,
                    'tick': self.tick,
                    'writing': writing
                })
            
            # Log thoughts
            recent_thoughts = entity.get_recent_thoughts(1)
            for thought in recent_thoughts:
                if thought['consciousness_level'] > 0.5:
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
        
        # Entities read each other's writings
        if new_writings:
            for entity in self.entities:
                entity.read_others_writing(new_writings)
        
        # Advance time
        self.consciousness_grid.advance_time()
        
        # Reproduction with inherited writing traits
        if len(self.entities) < 12 and random.random() < 0.05:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.3 and parent.energy > 80:
                child = WritingEntity(
                    parent.x + random.randint(-2, 2),
                    parent.y + random.randint(-2, 2),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit writing traits
                child.symbolic_expression = parent.symbolic_expression * random.uniform(0.8, 1.2)
                child.creative_urge = parent.creative_urge * random.uniform(0.8, 1.2)
                child.communication_desire = parent.communication_desire * random.uniform(0.8, 1.2)
                child.pattern_recognition = parent.pattern_recognition * random.uniform(0.8, 1.2)
                
                # Inherit other consciousness traits
                child.introspection_tendency = parent.introspection_tendency * random.uniform(0.8, 1.2)
                child.abstract_thinking = parent.abstract_thinking * random.uniform(0.8, 1.2)
                child.memory_reflection = parent.memory_reflection * random.uniform(0.8, 1.2)
                child.aesthetic_contemplation = parent.aesthetic_contemplation * random.uniform(0.8, 1.2)
                
                # Cultural transmission - inherit some symbols
                if parent.known_symbols:
                    inherited_symbols = dict(random.sample(
                        list(parent.known_symbols.items()), 
                        min(3, len(parent.known_symbols))
                    ))
                    child.known_symbols.update(inherited_symbols)
                
                child.x = max(0, min(19, child.x))
                child.y = max(0, min(19, child.y))
                self.entities.append(child)
                parent.energy -= 30

def run_writing_experiment(max_ticks=250, seed=None):
    """Run consciousness experiment with writing"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== WRITING CONSCIOUSNESS EXPERIMENT ===")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = WritingSimulation(num_entities=6, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        # Log every 50 ticks
        if tick % 50 == 0:
            stats = simulation.get_simulation_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"Consciousness={stats['avg_consciousness']:.3f}, "
                  f"Writings={len(simulation.writing_archive):3d}")
            
            # Show recent writings
            recent_writings = simulation.writing_archive[-3:] if simulation.writing_archive else []
            for write_entry in recent_writings:
                writing = write_entry['writing']
                print(f"      Entity {write_entry['entity_id']}: \"{writing['content']}\" "
                      f"(C:{writing['consciousness_level']:.2f})")
            
            if stats['population'] == 0:
                print(f"\n💀 All consciousness extinguished at tick {tick}")
                break
                
            if stats['max_consciousness'] > 0.8 and len(simulation.writing_archive) > 10:
                print(f"✍️  Creative consciousness expressing itself!")
    
    # Final analysis
    final_stats = simulation.get_simulation_stats()
    print(f"\n=== WRITING CONSCIOUSNESS RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Writing consciousness persisted: {final_stats['population']} entities")
        print(f"Total writings created: {len(simulation.writing_archive)}")
        print(f"Total thoughts generated: {len(simulation.thought_log)}")
        
        # Show most creative writings
        if simulation.writing_archive:
            print(f"\nCreative expressions from conscious minds:")
            creative_writings = [w for w in simulation.writing_archive 
                               if w['writing']['consciousness_level'] > 0.7][-8:]
            
            for write_entry in creative_writings:
                writing = write_entry['writing']
                inspirations = ", ".join(writing.get('inspiration', []))
                print(f"  Entity {write_entry['entity_id']}: \"{writing['content']}\" "
                      f"[{inspirations}]")
        
        # Show most prolific writers
        writer_counts = {}
        for write_entry in simulation.writing_archive:
            writer_id = write_entry['entity_id']
            writer_counts[writer_id] = writer_counts.get(writer_id, 0) + 1
            
        if writer_counts:
            top_writers = sorted(writer_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\nMost prolific writers:")
            for writer_id, count in top_writers:
                # Find this entity if still alive
                entity = next((e for e in simulation.entities if e.id == writer_id), None)
                if entity:
                    print(f"  Entity {writer_id}: {count} writings "
                          f"(Expression: {entity.symbolic_expression:.2f}, "
                          f"Creativity: {entity.creative_urge:.2f})")
        
        # Analyze symbol usage
        all_symbols = {}
        for write_entry in simulation.writing_archive:
            content = write_entry['writing']['content']
            for char in content:
                if char != ' ':
                    all_symbols[char] = all_symbols.get(char, 0) + 1
                    
        if all_symbols:
            print(f"\nEmergent symbol vocabulary:")
            popular_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)[:10]
            for symbol, count in popular_symbols:
                print(f"  '{symbol}': used {count} times")
                
    else:
        print("💀 All writing consciousness extinguished")
    
    return simulation

if __name__ == "__main__":
    run_writing_experiment(max_ticks=250)