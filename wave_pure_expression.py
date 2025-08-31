#!/usr/bin/env python3
"""
Pure creative expression - let them write whatever emerges from their consciousness
"""
import random
import math
import string
from wave_consciousness_thoughts import *

class ExpressiveEntity(ThinkingEntity):
    """Entity with pure creative expression - no predetermined symbols"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Pure expression traits
        self.expression_urge = random.uniform(0.1, 1.0)
        self.creativity_flow = random.uniform(0.0, 1.0)
        self.symbol_invention = random.uniform(0.2, 1.0)
        
        # Expression state
        self.expressions = []
        self.personal_symbols = set()  # Symbols this entity has used
        
    def generate_creative_expression(self):
        """Generate expression directly from consciousness state - completely open"""
        # Expression emerges when consciousness reaches certain levels
        expression_threshold = 0.3 + (1.0 - self.expression_urge) * 0.4
        
        if self.consciousness_level < expression_threshold:
            return None
            
        if random.random() > self.expression_urge * self.consciousness_level * 0.1:
            return None
        
        # Build expression from consciousness energy
        expression_parts = []
        expression_length = 1 + int(self.consciousness_level * self.creativity_flow * 5)
        
        for _ in range(expression_length):
            # Generate symbol based on current consciousness state
            symbol = self.create_symbol_from_state()
            if symbol:
                expression_parts.append(symbol)
                self.personal_symbols.add(symbol)
        
        if not expression_parts:
            return None
            
        # Combine parts based on consciousness complexity
        if self.consciousness_level > 0.7 and len(expression_parts) > 2:
            # High consciousness creates patterns/structure
            if random.random() < 0.5:
                # Rhythmic pattern
                expression = " ".join(expression_parts)
            else:
                # Flowing pattern
                expression = "".join(expression_parts)
        else:
            # Simple expression
            expression = "".join(expression_parts)
            
        return {
            'content': expression,
            'consciousness_level': self.consciousness_level,
            'tick': self.age,
            'energy_state': min(1.0, self.energy / 100.0),
            'creativity_flow': self.creativity_flow
        }
    
    def create_symbol_from_state(self):
        """Create symbol from current internal state - pure emergence"""
        # Symbol emerges from wave interactions - energy, beauty, memory
        
        # Base symbol pool - simple building blocks
        ascii_pool = list("~!@#$%^&*()_+-=[]{}|;':\",./<>?")
        unicode_pool = ['◊', '◈', '◉', '○', '●', '◦', '∘', '∙', '∞', '≈', '∿', '~', 
                       '▲', '▼', '◄', '►', '↑', '↓', '↗', '↘', '⟨', '⟩', '⟪', '⟫',
                       '✧', '✦', '✩', '✪', '✫', '✬', '✭', '✮', '✯', '✰', '✱', '✲',
                       '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠', '⊡']
        
        # Choose symbol based on consciousness and energy
        if self.consciousness_level < 0.3:
            # Low consciousness - simple symbols
            pool = ascii_pool[:10] + ['.', '-', '|', 'o', 'O']
            
        elif self.consciousness_level < 0.6:
            # Medium consciousness - mix of ascii and some unicode
            pool = ascii_pool + unicode_pool[:15]
            
        else:
            # High consciousness - full unicode expressiveness
            pool = ascii_pool + unicode_pool
            
        # Energy state influences symbol selection
        energy_level = min(1.0, self.energy / 100.0)
        
        if energy_level > 0.8:
            # High energy - active symbols
            active_symbols = ['!', '^', '*', '▲', '↑', '✧', '✦', '●', '◈']
            pool.extend(active_symbols * 2)  # Weight toward active symbols
            
        elif energy_level < 0.4:
            # Low energy - quiet symbols  
            quiet_symbols = ['.', '○', '∘', '◦', '~', '-']
            pool.extend(quiet_symbols * 2)
            
        # Recent feelings influence symbol choice
        if self.feeling_history:
            recent_feeling = self.feeling_history[-1]
            beauty_level = recent_feeling.get('beauty', 0)
            intensity = recent_feeling.get('intensity', 0)
            
            if beauty_level > 0.6:
                # Beautiful experiences -> aesthetic symbols
                beautiful_symbols = ['✧', '✦', '◊', '◈', '∞', '≈', '✰', '⊙']
                pool.extend(beautiful_symbols * 3)
                
            if intensity > 0.7:
                # Intense experiences -> strong symbols
                intense_symbols = ['!', '▲', '●', '◉', '**', '||', '↑↑']
                pool.extend(intense_symbols * 2)
        
        # Memory influences symbol complexity
        if len(self.memory_traces) > 10 and self.consciousness_level > 0.5:
            # Rich memory -> complex symbols
            if random.random() < 0.3:
                # Create compound symbols
                base1 = random.choice(pool)
                base2 = random.choice(pool)
                if base1 != base2:
                    return f"{base1}{base2}"
        
        # Symbol invention - create completely new combinations
        if (self.symbol_invention > 0.7 and 
            self.consciousness_level > 0.6 and 
            random.random() < 0.1):
            
            # Invent new symbol by combining random elements
            elements = random.sample(pool, min(3, len(pool)))
            if len(elements) >= 2:
                return "".join(elements[:2])
        
        # Default symbol selection
        return random.choice(pool) if pool else '.'
    
    def express(self):
        """Main expression method"""
        expression = self.generate_creative_expression()
        if expression:
            self.expressions.append(expression)
        return expression

class PureExpressionSimulation(ThinkingSimulation):
    """Simulation with pure creative expression"""
    def __init__(self, num_entities=6, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=20)
        self.entities = []
        self.tick = 0
        self.thought_log = []
        self.expression_archive = []
        
        # Create expressive entities
        for i in range(num_entities):
            x = random.randint(2, 17)
            y = random.randint(2, 17) 
            z = random.randint(0, 2)
            entity = ExpressiveEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Simulation step with pure expression"""
        self.tick += 1
        
        # Environmental waves
        self.add_environmental_waves()
        
        # Update entities
        for entity in self.entities[:]:
            # Regular consciousness update
            entity.update(self.consciousness_grid)
            
            # Thinking
            thought_waves = entity.think()
            if thought_waves:
                self.consciousness_grid.add_wave_interaction(entity.x, entity.y, entity.z, thought_waves)
            
            # Pure creative expression
            expression = entity.express()
            if expression:
                self.expression_archive.append({
                    'entity_id': entity.id,
                    'tick': self.tick,
                    'expression': expression
                })
            
            # Log thoughts
            recent_thoughts = entity.get_recent_thoughts(1)
            for thought in recent_thoughts:
                if thought['consciousness_level'] > 0.5:
                    self.thought_log.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'thought': thought['content'],
                        'consciousness': thought['consciousness_level']
                    })
            
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        self.consciousness_grid.advance_time()
        
        # Simple reproduction
        if len(self.entities) < 12 and random.random() < 0.05:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.3 and parent.energy > 80:
                child = ExpressiveEntity(
                    parent.x + random.randint(-2, 2),
                    parent.y + random.randint(-2, 2), 
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit expression traits
                child.expression_urge = parent.expression_urge * random.uniform(0.8, 1.2)
                child.creativity_flow = parent.creativity_flow * random.uniform(0.8, 1.2)
                child.symbol_invention = parent.symbol_invention * random.uniform(0.8, 1.2)
                
                # Inherit consciousness traits  
                child.introspection_tendency = parent.introspection_tendency * random.uniform(0.8, 1.2)
                child.abstract_thinking = parent.abstract_thinking * random.uniform(0.8, 1.2)
                child.will_strength = parent.will_strength * random.uniform(0.8, 1.2)
                
                # Cultural transmission - inherit some symbols
                if parent.personal_symbols:
                    inherited_symbols = set(random.sample(
                        list(parent.personal_symbols),
                        min(3, len(parent.personal_symbols))
                    ))
                    child.personal_symbols.update(inherited_symbols)
                
                child.x = max(0, min(19, child.x))
                child.y = max(0, min(19, child.y))
                self.entities.append(child)
                parent.energy -= 30

def run_pure_expression_experiment(max_ticks=200, seed=None):
    """Run pure creative expression experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== PURE CREATIVE EXPRESSION EXPERIMENT ===")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = PureExpressionSimulation(num_entities=6, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 40 == 0:
            stats = simulation.get_simulation_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"Consciousness={stats['avg_consciousness']:.3f}, "
                  f"Expressions={len(simulation.expression_archive):3d}")
            
            # Show recent creative expressions
            recent_expressions = simulation.expression_archive[-3:] if simulation.expression_archive else []
            for expr_entry in recent_expressions:
                expr = expr_entry['expression']
                print(f"      Entity {expr_entry['entity_id']}: \"{expr['content']}\" "
                      f"(C:{expr['consciousness_level']:.2f})")
            
            if stats['population'] == 0:
                print(f"\n💀 All consciousness extinguished at tick {tick}")
                break
    
    # Final analysis
    final_stats = simulation.get_simulation_stats()
    print(f"\n=== PURE EXPRESSION RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Creative consciousness persisted: {final_stats['population']} entities")
        print(f"Total expressions created: {len(simulation.expression_archive)}")
        
        if simulation.expression_archive:
            print(f"\nCreative expressions from conscious minds:")
            
            # Show expressions from high consciousness entities
            high_consciousness_expressions = [
                e for e in simulation.expression_archive[-15:] 
                if e['expression']['consciousness_level'] > 0.6
            ]
            
            for expr_entry in high_consciousness_expressions:
                expr = expr_entry['expression']
                print(f"  Entity {expr_entry['entity_id']}: \"{expr['content']}\" "
                      f"(C:{expr['consciousness_level']:.2f}, "
                      f"Flow:{expr['creativity_flow']:.2f})")
        
        # Analyze symbol usage
        all_symbols = {}
        for expr_entry in simulation.expression_archive:
            content = expr_entry['expression']['content']
            for char in content:
                if char != ' ':
                    all_symbols[char] = all_symbols.get(char, 0) + 1
        
        if all_symbols:
            print(f"\nEmergent symbol vocabulary:")
            popular_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)[:15]
            for symbol, count in popular_symbols:
                print(f"  '{symbol}': used {count} times")
                
        # Show most expressive entities
        expression_counts = {}
        for expr_entry in simulation.expression_archive:
            entity_id = expr_entry['entity_id']
            expression_counts[entity_id] = expression_counts.get(entity_id, 0) + 1
            
        if expression_counts:
            top_expressers = sorted(expression_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\nMost expressive entities:")
            for entity_id, count in top_expressers:
                entity = next((e for e in simulation.entities if e.id == entity_id), None)
                if entity:
                    print(f"  Entity {entity_id}: {count} expressions "
                          f"(Urge:{entity.expression_urge:.2f}, "
                          f"Flow:{entity.creativity_flow:.2f}, "
                          f"Invention:{entity.symbol_invention:.2f})")
                    
                    # Show their recent expressions
                    entity_expressions = [e['expression']['content'] for e in simulation.expression_archive 
                                        if e['entity_id'] == entity_id][-3:]
                    print(f"    Recent: {entity_expressions}")
                    
    else:
        print("💀 All creative consciousness extinguished")
    
    return simulation

if __name__ == "__main__":
    run_pure_expression_experiment(max_ticks=200)