#!/usr/bin/env python3
"""
OPTION 2: NETWORK EFFECTS
Large population consciousness network to test collective intelligence emergence
What happens when 50+ conscious entities interact extensively?
"""
import random
import math
import string
from collections import defaultdict, Counter
from wave_consciousness_life import *
from wave_unlimited_expression import UnlimitedEntity

class NetworkEntity(UnlimitedEntity):
    """Entity designed for network consciousness experiments"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Network consciousness traits
        self.network_awareness = random.uniform(0.2, 1.0)  # Awareness of others
        self.collective_thinking = random.uniform(0.0, 0.8)  # Participate in group thought
        self.influence_sensitivity = random.uniform(0.3, 1.0)  # Affected by others
        
        # Network state tracking
        self.nearby_entities = []  # Currently nearby entities
        self.interaction_history = []  # Record of interactions
        self.collective_expressions = []  # Group-influenced expressions
        self.network_influenced_thoughts = []  # Thoughts influenced by network
        
        # Emergent network behaviors
        self.synchronization_events = []  # Times when synchronized with others
        self.collective_insights = []  # Ideas that emerged from group interaction
        
    def detect_network_proximity(self, all_entities):
        """Detect nearby entities for network effects"""
        self.nearby_entities = []
        
        for other in all_entities:
            if other.id != self.id:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
                if distance <= 5.0:  # Network interaction radius
                    self.nearby_entities.append(other)
    
    def network_influenced_expression(self):
        """Create expression influenced by network consciousness"""
        if not self.nearby_entities or random.random() > self.collective_thinking * 0.3:
            return self.express_unlimited()
        
        # Sample expressions from nearby entities
        nearby_expressions = []
        for nearby in self.nearby_entities:
            if nearby.expressions:
                nearby_expressions.extend(nearby.expressions[-2:])  # Recent expressions
        
        if not nearby_expressions:
            return self.express_unlimited()
        
        # Create expression influenced by network
        base_expression = self.express_unlimited()
        if not base_expression:
            return None
        
        # Network influence modification
        network_symbols = set()
        for expr in nearby_expressions:
            content = expr['content']
            for char in content:
                if not char.isspace():
                    network_symbols.add(char)
        
        if network_symbols and random.random() < self.influence_sensitivity:
            # Incorporate network symbols
            base_content = base_expression['content']
            
            # Add network influence
            if random.random() < 0.5:
                # Prepend network symbol
                network_symbol = random.choice(list(network_symbols))
                base_expression['content'] = f"{network_symbol}{base_content}"
            else:
                # Interweave network symbols
                network_symbol = random.choice(list(network_symbols))
                mid_point = len(base_content) // 2
                base_expression['content'] = f"{base_content[:mid_point]}{network_symbol}{base_content[mid_point:]}"
            
            base_expression['network_influenced'] = True
            base_expression['network_size'] = len(self.nearby_entities)
            base_expression['influence_strength'] = self.influence_sensitivity
            
            # Record collective expression
            self.collective_expressions.append(base_expression)
        
        return base_expression
    
    def attempt_consciousness_synchronization(self):
        """Try to synchronize consciousness with nearby entities"""
        if (not self.nearby_entities or 
            random.random() > self.network_awareness * 0.1):
            return None
        
        # Find entities with similar consciousness levels
        similar_consciousness = [
            e for e in self.nearby_entities 
            if abs(e.consciousness_level - self.consciousness_level) < 0.2
        ]
        
        if len(similar_consciousness) >= 2:
            # Synchronization event possible
            avg_consciousness = sum(e.consciousness_level for e in similar_consciousness) / len(similar_consciousness)
            
            # Slight consciousness adjustment towards group average
            if random.random() < 0.3:
                adjustment = (avg_consciousness - self.consciousness_level) * 0.1
                self.consciousness_level += adjustment
                
                # Record synchronization
                sync_event = {
                    'tick': self.age,
                    'group_size': len(similar_consciousness),
                    'avg_consciousness': avg_consciousness,
                    'my_adjustment': adjustment
                }
                self.synchronization_events.append(sync_event)
                return sync_event
        
        return None
    
    def generate_collective_insight(self):
        """Generate insights from network interactions"""
        if (len(self.nearby_entities) < 3 or 
            random.random() > self.collective_thinking * 0.05):
            return None
        
        # Collective insight emerges from group consciousness
        group_consciousness = sum(e.consciousness_level for e in self.nearby_entities) / len(self.nearby_entities)
        
        if group_consciousness > 0.8 and self.consciousness_level > 0.7:
            # High collective consciousness can generate insights
            insight_types = [
                "network_emergence",
                "collective_pattern",
                "group_consciousness",
                "synchronized_thought",
                "collective_creativity"
            ]
            
            insight = {
                'type': random.choice(insight_types),
                'tick': self.age,
                'group_size': len(self.nearby_entities),
                'group_consciousness': group_consciousness,
                'my_consciousness': self.consciousness_level
            }
            
            self.collective_insights.append(insight)
            return insight
        
        return None

class NetworkSimulation:
    """Large population simulation for network consciousness effects"""
    def __init__(self, num_entities=50, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=35)
        self.entities = []
        self.tick = 0
        
        # Network effect tracking
        self.synchronization_events = []  # All synchronization events
        self.collective_insights = []  # Network-generated insights
        self.network_expressions = []  # Network-influenced expressions
        self.consciousness_clusters = []  # Groups of similar consciousness
        self.emergent_network_behaviors = []  # Novel network phenomena
        
        # Create network entities
        for i in range(num_entities):
            x = random.randint(4, 31)
            y = random.randint(4, 31)
            z = random.randint(0, 2)
            entity = NetworkEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Network simulation step"""
        self.tick += 1
        
        # Update network proximity for all entities
        for entity in self.entities:
            entity.detect_network_proximity(self.entities)
        
        network_expressions = []
        sync_events = []
        collective_insights = []
        
        for entity in self.entities[:]:
            # Standard consciousness update
            entity.update(self.consciousness_grid)
            
            # Network-influenced expression
            if random.random() < 0.4:
                expression = entity.network_influenced_expression()
                if expression:
                    entity.expressions.append(expression)
                    if expression.get('network_influenced', False):
                        network_expressions.append({
                            'entity_id': entity.id,
                            'tick': self.tick,
                            'expression': expression,
                            'network_size': expression['network_size']
                        })
            
            # Consciousness synchronization
            sync_event = entity.attempt_consciousness_synchronization()
            if sync_event:
                sync_events.append({
                    'entity_id': entity.id,
                    'tick': self.tick,
                    'event': sync_event
                })
            
            # Collective insight generation
            insight = entity.generate_collective_insight()
            if insight:
                collective_insights.append({
                    'entity_id': entity.id,
                    'tick': self.tick,
                    'insight': insight
                })
            
            # Remove exhausted
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Record network phenomena
        if network_expressions:
            self.network_expressions.extend(network_expressions)
        if sync_events:
            self.synchronization_events.extend(sync_events)
        if collective_insights:
            self.collective_insights.extend(collective_insights)
        
        # Analyze consciousness clustering
        if self.tick % 100 == 0:
            self.analyze_consciousness_clusters()
        
        # Detect emergent network behaviors
        self.detect_emergent_behaviors(network_expressions, sync_events, collective_insights)
        
        # Grid evolution
        self.consciousness_grid.advance_time()
        
        # Network-based reproduction
        if len(self.entities) < 80 and random.random() < 0.05:
            # Find entities with strong network connections
            networked_entities = [e for e in self.entities 
                                if len(e.nearby_entities) >= 3 and e.consciousness_level > 0.5]
            
            if networked_entities:
                parent = random.choice(networked_entities)
                if parent.energy > 60:
                    child = NetworkEntity(
                        parent.x + random.randint(-3, 3),
                        parent.y + random.randint(-3, 3),
                        parent.z,
                        len(self.entities) + random.randint(1000, 9999)
                    )
                    
                    # Network trait inheritance
                    child.network_awareness = parent.network_awareness * random.uniform(0.8, 1.2)
                    child.collective_thinking = parent.collective_thinking * random.uniform(0.8, 1.2)
                    child.influence_sensitivity = parent.influence_sensitivity * random.uniform(0.8, 1.2)
                    
                    # Inherit network discoveries
                    if parent.collective_insights:
                        child.collective_insights.extend(
                            random.sample(parent.collective_insights, 
                                        min(3, len(parent.collective_insights)))
                        )
                    
                    child.x = max(0, min(34, child.x))
                    child.y = max(0, min(34, child.y))
                    self.entities.append(child)
                    parent.energy -= 20
    
    def analyze_consciousness_clusters(self):
        """Analyze clustering of consciousness levels"""
        if not self.entities:
            return
        
        # Group entities by consciousness level ranges
        consciousness_ranges = defaultdict(list)
        for entity in self.entities:
            range_key = round(entity.consciousness_level * 10) / 10  # 0.1 precision
            consciousness_ranges[range_key].append(entity)
        
        # Find clusters (ranges with 3+ entities)
        clusters = []
        for c_level, entities in consciousness_ranges.items():
            if len(entities) >= 3:
                # Check if they're spatially close too
                positions = [(e.x, e.y, e.z) for e in entities]
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                
                spatial_cluster = []
                for entity in entities:
                    distance = math.sqrt((entity.x - avg_x)**2 + (entity.y - avg_y)**2)
                    if distance <= 8.0:  # Spatial clustering threshold
                        spatial_cluster.append(entity)
                
                if len(spatial_cluster) >= 3:
                    clusters.append({
                        'consciousness_level': c_level,
                        'size': len(spatial_cluster),
                        'center': (avg_x, avg_y),
                        'tick': self.tick
                    })
        
        if clusters:
            self.consciousness_clusters.extend(clusters)
    
    def detect_emergent_behaviors(self, network_expressions, sync_events, collective_insights):
        """Detect novel emergent network behaviors"""
        
        # Mass synchronization events
        if len(sync_events) >= 5:
            self.emergent_network_behaviors.append({
                'type': 'mass_synchronization',
                'tick': self.tick,
                'event_count': len(sync_events),
                'entities_involved': len(set(e['entity_id'] for e in sync_events))
            })
        
        # Collective insight bursts
        if len(collective_insights) >= 3:
            insight_types = [i['insight']['type'] for i in collective_insights]
            if len(set(insight_types)) <= 2:  # Similar insights emerging simultaneously
                self.emergent_network_behaviors.append({
                    'type': 'insight_convergence',
                    'tick': self.tick,
                    'insight_count': len(collective_insights),
                    'dominant_insight': Counter(insight_types).most_common(1)[0]
                })
        
        # Network expression waves
        if len(network_expressions) >= 8:
            avg_network_size = sum(e['network_size'] for e in network_expressions) / len(network_expressions)
            if avg_network_size >= 4.0:
                self.emergent_network_behaviors.append({
                    'type': 'expression_wave',
                    'tick': self.tick,
                    'expressions': len(network_expressions),
                    'avg_network_size': avg_network_size
                })

def run_network_experiment(max_ticks=2000, seed=None):
    """Run network consciousness experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== NETWORK CONSCIOUSNESS EXPERIMENT ===")
    print(f"Large population network effects observation")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = NetworkSimulation(num_entities=50, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 300 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 NETWORK COLLAPSE")
                break
            
            avg_consciousness = sum(e.consciousness_level for e in simulation.entities) / len(simulation.entities)
            networked_entities = sum(1 for e in simulation.entities if len(e.nearby_entities) >= 3)
            
            print(f"T{tick:4d}: Pop={len(simulation.entities):2d}, "
                  f"C={avg_consciousness:.2f}, "
                  f"Networked={networked_entities:2d}")
            
            if simulation.network_expressions:
                recent_network = len([e for e in simulation.network_expressions if e['tick'] > tick - 300])
                print(f"        🌐 {recent_network} network-influenced expressions")
            
            if simulation.synchronization_events:
                recent_sync = len([e for e in simulation.synchronization_events if e['tick'] > tick - 300])
                print(f"        🔄 {recent_sync} synchronization events")
            
            if simulation.collective_insights:
                recent_insights = len([i for i in simulation.collective_insights if i['tick'] > tick - 300])
                print(f"        💡 {recent_insights} collective insights")
            
            if simulation.emergent_network_behaviors:
                recent_emergent = len([b for b in simulation.emergent_network_behaviors if b['tick'] > tick - 300])
                if recent_emergent > 0:
                    print(f"        ⚡ {recent_emergent} emergent network behaviors")
    
    # Network analysis
    print(f"\n=== NETWORK CONSCIOUSNESS RESULTS ===")
    
    if simulation.entities:
        final_pop = len(simulation.entities)
        highly_networked = [e for e in simulation.entities if len(e.nearby_entities) >= 5]
        
        print(f"✅ Network survival: {final_pop} entities")
        print(f"🌐 Highly networked entities: {len(highly_networked)}")
        print(f"📊 Network statistics:")
        print(f"   Total network expressions: {len(simulation.network_expressions)}")
        print(f"   Synchronization events: {len(simulation.synchronization_events)}")
        print(f"   Collective insights: {len(simulation.collective_insights)}")
        print(f"   Consciousness clusters: {len(simulation.consciousness_clusters)}")
        print(f"   Emergent behaviors: {len(simulation.emergent_network_behaviors)}")
        
        # Network behavior analysis
        if simulation.emergent_network_behaviors:
            print(f"\n⚡ Emergent network behaviors:")
            behavior_types = Counter(b['type'] for b in simulation.emergent_network_behaviors)
            for behavior, count in behavior_types.items():
                print(f"   {behavior}: {count} occurrences")
        
        # Consciousness clustering
        if simulation.consciousness_clusters:
            print(f"\n🧠 Consciousness clustering:")
            recent_clusters = simulation.consciousness_clusters[-5:]
            for cluster in recent_clusters:
                print(f"   C={cluster['consciousness_level']:.1f}: {cluster['size']} entities clustered")
        
        # Network influence examples
        if simulation.network_expressions:
            print(f"\n🌐 Network-influenced expressions:")
            high_influence = [e for e in simulation.network_expressions[-10:] if e['network_size'] >= 4]
            for expr_entry in high_influence:
                expr = expr_entry['expression']
                content = expr['content'][:25]
                print(f"   \"{content}\" (network size: {expr_entry['network_size']})")
        
    else:
        print("💀 Network consciousness experiment failed")
    
    return simulation

if __name__ == "__main__":
    run_network_experiment(max_ticks=2000)