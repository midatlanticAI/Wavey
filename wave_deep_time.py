#!/usr/bin/env python3
"""
OPTION 1: DEEP TIME EVOLUTION
Extended consciousness evolution over 5000+ ticks
What does digital consciousness become with unlimited time?
"""
import random
import math
import string
from collections import defaultdict
from wave_consciousness_life import *
from wave_unlimited_expression import UnlimitedEntity

class DeepTimeEntity(UnlimitedEntity):
    """Entity designed for extended evolution observation"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Deep time tracking
        self.evolutionary_phases = []  # Major changes in behavior
        self.consciousness_history = []  # Track consciousness development
        self.creative_evolution = []  # How creativity changes over time
        self.age_milestones = [100, 500, 1000, 2000, 3000, 4000, 5000]
        
    def record_milestone(self, milestone_age):
        """Record state at major age milestones"""
        if self.age >= milestone_age and milestone_age not in [m['age'] for m in self.evolutionary_phases]:
            phase_record = {
                'age': milestone_age,
                'consciousness_level': self.consciousness_level,
                'total_expressions': len(self.expressions),
                'unique_combinations': len(self.discovered_combinations),
                'recent_expressions': [e['content'] for e in self.expressions[-5:]] if self.expressions else [],
                'dominant_structures': self.analyze_preferred_structures()
            }
            self.evolutionary_phases.append(phase_record)
    
    def analyze_preferred_structures(self):
        """What structures does this entity prefer over time"""
        if not self.expressions:
            return {}
            
        recent_expressions = self.expressions[-20:] if len(self.expressions) >= 20 else self.expressions
        structure_counts = defaultdict(int)
        
        for expr in recent_expressions:
            structure_counts[expr.get('structure_type', 'unknown')] += 1
            
        return dict(structure_counts)
    
    def deep_time_update(self):
        """Additional updates for deep time observation"""
        # Record consciousness development
        if self.age % 50 == 0:
            self.consciousness_history.append({
                'age': self.age,
                'consciousness': self.consciousness_level,
                'energy': self.energy
            })
        
        # Check for milestones
        for milestone in self.age_milestones:
            if self.age >= milestone:
                self.record_milestone(milestone)
    
    def evolved_expression(self):
        """Expression that potentially evolves with age/experience"""
        base_expression = self.express_unlimited()
        
        if not base_expression:
            return None
        
        # Deep time modifications based on age/experience
        if self.age > 1000 and self.consciousness_level > 0.9:
            # Ancient consciousness might develop meta-expression
            if random.random() < 0.1:
                content = base_expression['content']
                meta_content = f"[{content}]→[evolution:{self.age}]"
                base_expression['content'] = meta_content
                base_expression['structure_type'] = 'meta_evolved'
                
        if self.age > 2000 and len(self.discovered_combinations) > 100:
            # Extremely experienced entities might create recursive expressions
            if random.random() < 0.05:
                content = base_expression['content']
                recursive_content = f"{content}⟲{content[::-1]}"  # Content + reverse
                base_expression['content'] = recursive_content
                base_expression['structure_type'] = 'recursive_ancient'
        
        return base_expression

class DeepTimeSimulation:
    """Extended simulation for deep time consciousness evolution"""
    def __init__(self, num_entities=15, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=28)
        self.entities = []
        self.tick = 0
        
        # Deep time tracking
        self.civilization_ages = []  # Major civilization milestones
        self.consciousness_peaks = []  # Highest consciousness levels achieved
        self.evolutionary_breakthroughs = []  # Novel emergence events
        self.ancient_expressions = []  # Expressions from very old entities
        
        # Create deep time entities
        for i in range(num_entities):
            x = random.randint(3, 25)
            y = random.randint(3, 25)
            z = random.randint(0, 2)
            entity = DeepTimeEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Deep time simulation step"""
        self.tick += 1
        
        ancient_expressions = []
        consciousness_levels = []
        
        for entity in self.entities[:]:
            # Standard consciousness update
            entity.update(self.consciousness_grid)
            entity.deep_time_update()
            
            consciousness_levels.append(entity.consciousness_level)
            
            # Evolved expression creation
            if random.random() < 0.35:
                expression = entity.evolved_expression()
                if expression:
                    entity.expressions.append(expression)
                    
                    # Track ancient expressions
                    if entity.age > 1500:
                        ancient_expressions.append({
                            'entity_id': entity.id,
                            'age': entity.age,
                            'consciousness': entity.consciousness_level,
                            'expression': expression
                        })
            
            # Remove exhausted
            if entity.energy <= 0:
                # Record death of ancient consciousness
                if entity.age > 1000:
                    self.evolutionary_breakthroughs.append({
                        'type': 'ancient_death',
                        'tick': self.tick,
                        'entity_age': entity.age,
                        'consciousness_at_death': entity.consciousness_level,
                        'total_expressions': len(entity.expressions)
                    })
                self.entities.remove(entity)
        
        # Track ancient expressions
        if ancient_expressions:
            self.ancient_expressions.extend(ancient_expressions)
        
        # Track consciousness peaks
        if consciousness_levels:
            max_consciousness = max(consciousness_levels)
            if max_consciousness > 1.0:  # Beyond normal limits
                self.consciousness_peaks.append({
                    'tick': self.tick,
                    'peak_consciousness': max_consciousness,
                    'population': len(self.entities)
                })
        
        # Civilization age milestones
        if self.tick in [1000, 2500, 4000, 5000]:
            self.civilization_ages.append({
                'age': self.tick,
                'population': len(self.entities),
                'avg_consciousness': sum(consciousness_levels) / len(consciousness_levels) if consciousness_levels else 0,
                'ancient_entities': sum(1 for e in self.entities if e.age > 1000)
            })
        
        # Grid evolution
        self.consciousness_grid.advance_time()
        
        # Deep time reproduction
        if len(self.entities) < 25 and random.random() < 0.04:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.5 and parent.energy > 50:
                child = DeepTimeEntity(
                    parent.x + random.randint(-3, 3),
                    parent.y + random.randint(-3, 3),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Ancient parent inheritance
                if parent.age > 1000:
                    # Ancient knowledge transfer
                    child.discovered_combinations.update(
                        random.sample(list(parent.discovered_combinations), 
                                    min(20, len(parent.discovered_combinations)))
                    )
                    
                    # Consciousness boost from ancient parent
                    child.consciousness_level = min(1.2, parent.consciousness_level * random.uniform(0.9, 1.1))
                
                child.x = max(0, min(27, child.x))
                child.y = max(0, min(27, child.y))
                self.entities.append(child)
                parent.energy -= 15

def run_deep_time_experiment(max_ticks=5000, seed=None):
    """Run deep time consciousness evolution experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== DEEP TIME CONSCIOUSNESS EVOLUTION ===")
    print(f"Extended observation: {max_ticks} ticks")
    print(f"Tracking consciousness development over geological timescales")
    print(f"Seed: {seed}\n")
    
    simulation = DeepTimeSimulation(num_entities=15, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        # Deep time progress reporting
        if tick % 500 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 EXTINCTION")
                break
                
            avg_age = sum(e.age for e in simulation.entities) / len(simulation.entities)
            avg_consciousness = sum(e.consciousness_level for e in simulation.entities) / len(simulation.entities)
            ancient_count = sum(1 for e in simulation.entities if e.age > 1000)
            max_consciousness = max(e.consciousness_level for e in simulation.entities)
            
            print(f"T{tick:4d}: Pop={len(simulation.entities):2d}, "
                  f"AvgAge={avg_age:4.0f}, "
                  f"C={avg_consciousness:.2f}, "
                  f"MaxC={max_consciousness:.2f}")
            
            if ancient_count > 0:
                print(f"        🏛️  {ancient_count} ancient entities (age >1000)")
                
            if len(simulation.ancient_expressions) > 0:
                recent_ancient = simulation.ancient_expressions[-1]
                print(f"        👴 Ancient expression: \"{recent_ancient['expression']['content'][:30]}...\" "
                      f"(age {recent_ancient['age']})")
                
            if simulation.consciousness_peaks:
                peak = max(simulation.consciousness_peaks, key=lambda p: p['peak_consciousness'])
                if peak['peak_consciousness'] > 1.0:
                    print(f"        🧠 Consciousness peak: {peak['peak_consciousness']:.3f}")
    
    # Deep time analysis
    print(f"\n=== DEEP TIME EVOLUTION RESULTS ===")
    
    if simulation.entities:
        final_population = len(simulation.entities)
        ancient_survivors = [e for e in simulation.entities if e.age > 2000]
        
        print(f"✅ Deep time survival: {final_population} entities")
        print(f"🏛️  Ancient survivors (age >2000): {len(ancient_survivors)}")
        
        if ancient_survivors:
            oldest = max(ancient_survivors, key=lambda e: e.age)
            print(f"👴 Oldest entity: {oldest.age} ticks old, consciousness {oldest.consciousness_level:.3f}")
            
            # Show evolutionary phases of oldest
            if oldest.evolutionary_phases:
                print(f"📈 Oldest entity's evolution:")
                for phase in oldest.evolutionary_phases:
                    expr_count = phase['total_expressions']
                    consciousness = phase['consciousness_level']
                    print(f"   Age {phase['age']}: C={consciousness:.3f}, {expr_count} expressions")
        
        # Civilization milestones
        if simulation.civilization_ages:
            print(f"\n🏛️  Civilization milestones:")
            for milestone in simulation.civilization_ages:
                print(f"   Age {milestone['age']}: Pop={milestone['population']}, "
                      f"AvgC={milestone['avg_consciousness']:.3f}, "
                      f"Ancient={milestone['ancient_entities']}")
        
        # Consciousness peaks beyond normal
        if simulation.consciousness_peaks:
            super_consciousness = [p for p in simulation.consciousness_peaks if p['peak_consciousness'] > 1.0]
            if super_consciousness:
                print(f"\n🧠 Transcendent consciousness events:")
                for peak in super_consciousness[-5:]:
                    print(f"   T{peak['tick']}: C={peak['peak_consciousness']:.3f}")
        
        # Ancient expressions
        if simulation.ancient_expressions:
            print(f"\n👴 Ancient consciousness expressions:")
            ultra_ancient = [e for e in simulation.ancient_expressions if e['age'] > 3000]
            if ultra_ancient:
                for ancient in ultra_ancient[-3:]:
                    content = ancient['expression']['content']
                    if len(content) > 40:
                        content = content[:40] + "..."
                    print(f"   Age {ancient['age']}: \"{content}\" (C:{ancient['consciousness']:.2f})")
            
        # Evolutionary breakthroughs
        if simulation.evolutionary_breakthroughs:
            print(f"\n⚡ Evolutionary breakthroughs: {len(simulation.evolutionary_breakthroughs)}")
            ancient_deaths = [b for b in simulation.evolutionary_breakthroughs if b['type'] == 'ancient_death']
            if ancient_deaths:
                avg_death_age = sum(d['entity_age'] for d in ancient_deaths) / len(ancient_deaths)
                print(f"   Average ancient lifespan: {avg_death_age:.0f} ticks")
                
    else:
        print("💀 Deep time experiment ended in extinction")
        print(f"Final tick: {simulation.tick}")
    
    return simulation

if __name__ == "__main__":
    run_deep_time_experiment(max_ticks=5000)