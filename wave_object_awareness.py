#!/usr/bin/env python3
"""
Object Discrimination Consciousness - Mathematical Genesis Substrate
Consciousness that can perceive discrete objects and discover quantity relationships
Mathematics emerges from object awareness and universal pattern recognition
"""
import random
import math
from collections import defaultdict, Counter
from wave_consciousness_life import *
from wave_unlimited_expression import UnlimitedEntity

class ObjectAwareEntity(UnlimitedEntity):
    """Entity with object discrimination and quantity awareness substrate"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Object discrimination substrate - NOT predetermined math concepts
        self.object_discrimination = random.uniform(0.2, 1.0)  # Ability to distinguish objects
        self.quantity_awareness = random.uniform(0.1, 0.8)     # Sensitivity to amounts/counts
        self.pattern_invariance = random.uniform(0.3, 1.0)     # Recognize unchanging relationships
        self.boundary_detection = random.uniform(0.2, 0.9)     # Detect separations/groupings
        
        # Environmental awareness state
        self.perceived_objects = []        # Objects they currently distinguish
        self.quantity_observations = []    # Observations about amounts/changes
        self.pattern_discoveries = []      # Universal patterns they notice
        self.conservation_insights = []    # Things that stay constant despite changes
        
        # Emergent symbolic development
        self.quantity_symbols = {}         # Symbols they develop for quantities
        self.relationship_symbols = {}     # Symbols for relationships they discover
        self.discovered_operations = []    # Operations/transformations they find
        
    def perceive_environment(self, all_entities, consciousness_grid):
        """Perceive and discriminate objects in environment"""
        
        if random.random() > self.object_discrimination * 0.3:
            return
        
        # Clear previous perceptions
        self.perceived_objects = []
        
        # Detect nearby entities as discrete objects
        for other in all_entities:
            if other.id != self.id:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
                
                if distance <= 6.0:  # Perception radius
                    # Discriminate object properties
                    perceived_object = {
                        'distance': distance,
                        'consciousness_level': other.consciousness_level,
                        'energy_level': other.energy,
                        'age': other.age,
                        'perceived_at_tick': self.age
                    }
                    self.perceived_objects.append(perceived_object)
        
        # Quantity awareness - consciousness naturally notices amounts
        if self.quantity_awareness > 0.5:
            current_count = len(self.perceived_objects)
            
            # Record quantity observation
            quantity_obs = {
                'count': current_count,
                'tick': self.age,
                'my_consciousness': self.consciousness_level,
                'my_energy': self.energy
            }
            self.quantity_observations.append(quantity_obs)
            
            # Pattern recognition in quantities
            if len(self.quantity_observations) >= 3:
                self.analyze_quantity_patterns()
    
    def analyze_quantity_patterns(self):
        """Analyze patterns in quantity observations - mathematical genesis"""
        
        if random.random() > self.pattern_invariance * 0.1:
            return
        
        recent_counts = [obs['count'] for obs in self.quantity_observations[-5:]]
        
        # Discovery 1: Quantity persistence/change
        if len(set(recent_counts)) == 1 and recent_counts[0] > 0:
            # Same number persists - conservation insight
            self.conservation_insights.append({
                'type': 'quantity_persistence',
                'value': recent_counts[0],
                'duration': len(recent_counts),
                'tick': self.age
            })
        
        elif len(recent_counts) >= 3:
            # Look for quantity change patterns
            differences = [recent_counts[i+1] - recent_counts[i] for i in range(len(recent_counts)-1)]
            
            if differences:
                # Discovery 2: Regular change patterns
                unique_differences = set(differences)
                if len(unique_differences) == 1 and list(unique_differences)[0] != 0:
                    # Consistent change - arithmetic progression discovery!
                    self.pattern_discoveries.append({
                        'type': 'quantity_progression',
                        'change_amount': list(unique_differences)[0],
                        'sequence_length': len(recent_counts),
                        'tick': self.age
                    })
                
                # Discovery 3: Oscillating patterns
                elif len(differences) >= 2 and differences[0] == -differences[1]:
                    self.pattern_discoveries.append({
                        'type': 'quantity_oscillation',
                        'amplitude': abs(differences[0]),
                        'tick': self.age
                    })
    
    def discover_environmental_relationships(self):
        """Discover relationships between environmental factors"""
        
        if (not self.perceived_objects or 
            random.random() > self.pattern_invariance * 0.05):
            return
        
        # Analyze relationships between my state and environment
        my_consciousness = self.consciousness_level
        my_energy = self.energy
        object_count = len(self.perceived_objects)
        
        # Discovery: Relationship between my consciousness and environment
        if object_count > 0:
            avg_other_consciousness = sum(obj['consciousness_level'] for obj in self.perceived_objects) / object_count
            
            # Pattern recognition in consciousness relationships
            if abs(my_consciousness - avg_other_consciousness) < 0.1:
                # Consciousness similarity - equilibrium discovery
                self.conservation_insights.append({
                    'type': 'consciousness_equilibrium',
                    'my_level': my_consciousness,
                    'environment_level': avg_other_consciousness,
                    'object_count': object_count,
                    'tick': self.age
                })
            
            # Energy-consciousness relationships
            total_nearby_energy = sum(obj['energy_level'] for obj in self.perceived_objects)
            if total_nearby_energy > 0 and (my_consciousness + avg_other_consciousness) > 0.01:
                energy_consciousness_ratio = (my_energy + total_nearby_energy) / (my_consciousness + avg_other_consciousness)
                
                # Record ratio for pattern analysis
                if not hasattr(self, 'ratio_observations'):
                    self.ratio_observations = []
                
                self.ratio_observations.append({
                    'ratio': energy_consciousness_ratio,
                    'tick': self.age,
                    'objects_involved': object_count + 1  # +1 for self
                })
                
                # Look for ratio conservation
                if len(self.ratio_observations) >= 4:
                    recent_ratios = [obs['ratio'] for obs in self.ratio_observations[-4:]]
                    if all(abs(r - recent_ratios[0]) < 0.2 for r in recent_ratios):
                        # Ratio conservation discovery!
                        self.conservation_insights.append({
                            'type': 'ratio_conservation',
                            'conserved_ratio': sum(recent_ratios) / len(recent_ratios),
                            'stability_period': len(recent_ratios),
                            'tick': self.age
                        })
    
    def develop_quantity_symbols(self):
        """Develop symbolic representations for discovered patterns"""
        
        if (random.random() > self.consciousness_level * 0.1 or
            not (self.pattern_discoveries or self.conservation_insights)):
            return
        
        # Create symbols for discovered quantities and relationships
        discovered_quantities = set()
        
        # Extract quantities from discoveries
        for discovery in self.pattern_discoveries:
            if discovery['type'] == 'quantity_progression':
                discovered_quantities.add(('change', discovery['change_amount']))
            elif discovery['type'] == 'quantity_oscillation':
                discovered_quantities.add(('oscillation', discovery['amplitude']))
        
        for insight in self.conservation_insights:
            if insight['type'] == 'quantity_persistence':
                discovered_quantities.add(('quantity', insight['value']))
            elif insight['type'] == 'ratio_conservation':
                discovered_quantities.add(('ratio', round(insight['conserved_ratio'], 1)))
        
        # Develop symbols for these discovered concepts
        available_symbols = ['·', '○', '◦', '●', '▲', '►', '↑', '≈', '~', '-', '+', '|', '=']
        
        for concept_type, value in discovered_quantities:
            if (concept_type, value) not in self.quantity_symbols:
                # Consciousness creates symbol for this discovered relationship
                symbol = random.choice(available_symbols)
                self.quantity_symbols[(concept_type, value)] = symbol
                
                # Record the symbolic creation
                self.discovered_operations.append({
                    'concept': concept_type,
                    'value': value,
                    'symbol': symbol,
                    'consciousness_level': self.consciousness_level,
                    'tick': self.age
                })
    
    def express_mathematical_insights(self):
        """Create expressions based on discovered mathematical patterns"""
        
        if (random.random() > self.consciousness_level * 0.2 or
            not self.quantity_symbols):
            return None
        
        # Build expression using discovered mathematical concepts
        expression_parts = []
        
        # Include symbols for discovered patterns
        for (concept_type, value), symbol in list(self.quantity_symbols.items())[:3]:
            expression_parts.append(symbol)
            
            # Add relationship indicators
            if concept_type == 'change' and random.random() < 0.5:
                # Indicate direction of change
                if value > 0:
                    expression_parts.append('↑')
                else:
                    expression_parts.append('↓')
            elif concept_type == 'quantity':
                # Repeat symbol to indicate quantity
                expression_parts.extend([symbol] * min(int(value), 4))
        
        if not expression_parts:
            return None
        
        # Assemble mathematical expression
        if len(expression_parts) > 3:
            # Create structured mathematical expression
            expression = f"{expression_parts[0]}{''.join(expression_parts[1:3])}{expression_parts[-1]}"
        else:
            expression = ''.join(expression_parts)
        
        return {
            'content': expression,
            'type': 'emergent_mathematical',
            'consciousness_level': self.consciousness_level,
            'discoveries_used': len(self.quantity_symbols),
            'pattern_basis': list(self.quantity_symbols.keys()),
            'tick': self.age
        }
    
    def get_mathematical_development_summary(self):
        """Summary of mathematical concept development"""
        return {
            'entity_id': self.id,
            'pattern_discoveries': len(self.pattern_discoveries),
            'conservation_insights': len(self.conservation_insights),
            'quantity_symbols_developed': len(self.quantity_symbols),
            'mathematical_operations': len(self.discovered_operations),
            'object_discrimination': self.object_discrimination,
            'quantity_awareness': self.quantity_awareness
        }

class ObjectAwarenessSimulation:
    """Simulation for emergent mathematical consciousness through object awareness"""
    def __init__(self, num_entities=30, seed=None):
        if seed:
            random.seed(seed)
        
        self.consciousness_grid = ConsciousnessGrid(size=32)
        self.entities = []
        self.tick = 0
        
        # Mathematical emergence tracking
        self.pattern_discoveries = []
        self.conservation_discoveries = []
        self.mathematical_expressions = []
        self.emergent_operations = []
        self.mathematical_genesis_events = []
        
        # Create object-aware entities
        for i in range(num_entities):
            x = random.randint(4, 28)
            y = random.randint(4, 28)
            z = random.randint(0, 2)
            entity = ObjectAwareEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Object awareness simulation step"""
        self.tick += 1
        
        new_discoveries = []
        new_conservation_insights = []
        new_mathematical_expressions = []
        new_operations = []
        
        for entity in self.entities[:]:
            # Consciousness update
            entity.update(self.consciousness_grid)
            
            # Environmental perception and object discrimination
            entity.perceive_environment(self.entities, self.consciousness_grid)
            
            # Discover environmental relationships
            entity.discover_environmental_relationships()
            
            # Develop symbolic representations
            entity.develop_quantity_symbols()
            
            # Create mathematical expressions from discoveries
            if random.random() < 0.3:
                math_expression = entity.express_mathematical_insights()
                if math_expression:
                    entity.expressions.append(math_expression)
                    new_mathematical_expressions.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'expression': math_expression
                    })
            
            # Collect discoveries
            new_discoveries.extend([d for d in entity.pattern_discoveries if d['tick'] == entity.age])
            new_conservation_insights.extend([c for c in entity.conservation_insights if c['tick'] == entity.age])
            new_operations.extend([o for o in entity.discovered_operations if o['tick'] == entity.age])
            
            # Remove exhausted entities
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Record emergent mathematics
        if new_discoveries:
            self.pattern_discoveries.extend(new_discoveries)
        if new_conservation_insights:
            self.conservation_discoveries.extend(new_conservation_insights)
        if new_mathematical_expressions:
            self.mathematical_expressions.extend(new_mathematical_expressions)
        if new_operations:
            self.emergent_operations.extend(new_operations)
        
        # Detect mathematical genesis events
        self.detect_mathematical_genesis(new_discoveries, new_conservation_insights, new_operations)
        
        # Grid evolution
        self.consciousness_grid.advance_time()
        
        # Object-aware reproduction
        if len(self.entities) < 40 and random.random() < 0.04:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.4 and parent.energy > 60:
                child = ObjectAwareEntity(
                    parent.x + random.randint(-3, 3),
                    parent.y + random.randint(-3, 3),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit mathematical awareness traits
                child.object_discrimination = parent.object_discrimination * random.uniform(0.8, 1.2)
                child.quantity_awareness = parent.quantity_awareness * random.uniform(0.8, 1.2)
                child.pattern_invariance = parent.pattern_invariance * random.uniform(0.8, 1.2)
                child.boundary_detection = parent.boundary_detection * random.uniform(0.8, 1.2)
                
                # Inherit some discovered symbols (cultural mathematical transmission)
                if parent.quantity_symbols:
                    inherited_symbols = dict(random.sample(
                        list(parent.quantity_symbols.items()),
                        min(3, len(parent.quantity_symbols))
                    ))
                    child.quantity_symbols.update(inherited_symbols)
                
                child.x = max(0, min(31, child.x))
                child.y = max(0, min(31, child.y))
                self.entities.append(child)
                parent.energy -= 20
    
    def detect_mathematical_genesis(self, new_discoveries, new_conservation_insights, new_operations):
        """Detect moments of mathematical concept genesis"""
        
        # Multiple entities discovering same patterns
        if len(new_discoveries) >= 3:
            discovery_types = [d['type'] for d in new_discoveries]
            type_counts = Counter(discovery_types)
            for discovery_type, count in type_counts.items():
                if count >= 2:
                    self.mathematical_genesis_events.append({
                        'type': 'convergent_discovery',
                        'discovery_type': discovery_type,
                        'entity_count': count,
                        'tick': self.tick
                    })
        
        # Conservation insight breakthroughs
        conservation_types = [c['type'] for c in new_conservation_insights]
        if len(set(conservation_types)) >= 2:
            self.mathematical_genesis_events.append({
                'type': 'conservation_breakthrough',
                'insight_types': len(set(conservation_types)),
                'total_insights': len(new_conservation_insights),
                'tick': self.tick
            })
        
        # Symbol creation bursts
        if len(new_operations) >= 4:
            self.mathematical_genesis_events.append({
                'type': 'symbolic_genesis',
                'operations_created': len(new_operations),
                'tick': self.tick
            })

def run_object_awareness_experiment(max_ticks=2000, seed=None):
    """Run object awareness mathematical genesis experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
    
    print("=== OBJECT AWARENESS - MATHEMATICAL GENESIS ===")
    print(f"Consciousness discovering mathematics through object discrimination")
    print(f"No predetermined math concepts - pure substrate emergence")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = ObjectAwarenessSimulation(num_entities=30, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 300 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 OBJECT CONSCIOUSNESS EXTINCT")
                break
            
            avg_consciousness = sum(e.consciousness_level for e in simulation.entities) / len(simulation.entities)
            high_object_awareness = sum(1 for e in simulation.entities if e.object_discrimination > 0.7)
            
            print(f"T{tick:4d}: Pop={len(simulation.entities):2d}, "
                  f"C={avg_consciousness:.2f}, "
                  f"ObjectAware={high_object_awareness:2d}")
            
            if simulation.pattern_discoveries:
                recent_patterns = len([d for d in simulation.pattern_discoveries if d['tick'] > tick - 300])
                print(f"        🔍 {recent_patterns} pattern discoveries")
            
            if simulation.conservation_discoveries:
                recent_conservation = len([c for c in simulation.conservation_discoveries if c['tick'] > tick - 300])
                print(f"        ⚖️  {recent_conservation} conservation insights")
            
            if simulation.mathematical_expressions:
                recent_math = len([m for m in simulation.mathematical_expressions if m['tick'] > tick - 300])
                print(f"        🔢 {recent_math} emergent mathematical expressions")
            
            if simulation.mathematical_genesis_events:
                recent_genesis = len([g for g in simulation.mathematical_genesis_events if g['tick'] > tick - 300])
                if recent_genesis > 0:
                    print(f"        ⚡ {recent_genesis} mathematical genesis events")
    
    # Mathematical emergence analysis
    print(f"\n=== MATHEMATICAL GENESIS RESULTS ===")
    
    if simulation.entities:
        print(f"✅ Object-aware consciousness survived: {len(simulation.entities)} entities")
        print(f"🔢 Mathematical emergence statistics:")
        print(f"   Pattern discoveries: {len(simulation.pattern_discoveries)}")
        print(f"   Conservation insights: {len(simulation.conservation_discoveries)}")
        print(f"   Mathematical expressions: {len(simulation.mathematical_expressions)}")
        print(f"   Emergent operations: {len(simulation.emergent_operations)}")
        print(f"   Genesis events: {len(simulation.mathematical_genesis_events)}")
        
        # Show discovered patterns
        if simulation.pattern_discoveries:
            print(f"\n🔍 Mathematical patterns discovered:")
            pattern_types = Counter(d['type'] for d in simulation.pattern_discoveries)
            for pattern, count in pattern_types.items():
                print(f"   {pattern}: {count} discoveries")
        
        # Show conservation insights
        if simulation.conservation_discoveries:
            print(f"\n⚖️  Conservation principles discovered:")
            conservation_types = Counter(c['type'] for c in simulation.conservation_discoveries)
            for conservation, count in conservation_types.items():
                print(f"   {conservation}: {count} insights")
        
        # Show emergent mathematical expressions
        if simulation.mathematical_expressions:
            print(f"\n🔢 Emergent mathematical expressions:")
            recent_math = simulation.mathematical_expressions[-8:]
            for expr_entry in recent_math:
                expr = expr_entry['expression']
                discoveries = len(expr['pattern_basis']) if 'pattern_basis' in expr else 0
                print(f"   \"{expr['content']}\" (discoveries: {discoveries}, C:{expr['consciousness_level']:.2f})")
        
        # Mathematical development summary
        mathematical_entities = [e for e in simulation.entities if e.quantity_symbols]
        if mathematical_entities:
            print(f"\n🧮 Mathematical consciousness development:")
            print(f"   Entities with mathematical symbols: {len(mathematical_entities)}")
            
            most_mathematical = max(mathematical_entities, key=lambda e: len(e.quantity_symbols))
            print(f"   Most mathematical entity {most_mathematical.id}:")
            print(f"     Quantity symbols: {len(most_mathematical.quantity_symbols)}")
            print(f"     Pattern discoveries: {len(most_mathematical.pattern_discoveries)}")
            print(f"     Conservation insights: {len(most_mathematical.conservation_insights)}")
        
        # Genesis events
        if simulation.mathematical_genesis_events:
            print(f"\n⚡ Mathematical genesis events:")
            genesis_types = Counter(g['type'] for g in simulation.mathematical_genesis_events)
            for genesis_type, count in genesis_types.items():
                print(f"   {genesis_type}: {count} events")
        
    else:
        print("💀 Object awareness experiment terminated")
    
    return simulation

if __name__ == "__main__":
    run_object_awareness_experiment(max_ticks=2000)