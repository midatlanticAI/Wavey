#!/usr/bin/env python3
"""
Wave consciousness with collective history and civilizational memory
Entities maintain shared records, cultural artifacts, and accumulated knowledge
"""
import random
import math
from wave_consciousness_life import *
from wave_pure_expression import *
from wave_full_consciousness import FullConsciousnessEntity

class CollectiveMemory:
    """Shared civilizational knowledge and history"""
    def __init__(self):
        # Historical records
        self.historical_events = []  # Major events in civilization
        self.cultural_artifacts = []  # Art, tools, and expressions preserved
        self.knowledge_archive = {}  # Accumulated wisdom and discoveries
        self.population_records = []  # Demographics over time
        
        # Cultural evolution tracking
        self.symbol_meanings = {}  # Shared symbol dictionary
        self.successful_patterns = {}  # Patterns that helped survival
        self.philosophical_insights = []  # Deep thoughts preserved
        self.technological_progress = []  # Tools and innovations
        
        # Story fragments - emergent narratives
        self.origin_stories = []  # How they understand their beginning
        self.creation_myths = []  # Stories about consciousness emergence
        self.heroic_tales = []  # Stories of exceptional individuals
        
    def record_historical_event(self, event_type, details, tick):
        """Record significant events"""
        event = {
            'type': event_type,
            'details': details,
            'tick': tick,
            'significance': self.calculate_significance(event_type, details)
        }
        self.historical_events.append(event)
        
        # Keep only most significant events to prevent memory bloat
        if len(self.historical_events) > 100:
            self.historical_events.sort(key=lambda e: e['significance'], reverse=True)
            self.historical_events = self.historical_events[:80]
    
    def calculate_significance(self, event_type, details):
        """Calculate how significant an event is for preservation"""
        significance_map = {
            'first_consciousness': 10.0,
            'first_tool': 8.0,
            'first_art': 6.0,
            'population_milestone': 5.0,
            'philosophical_breakthrough': 7.0,
            'cultural_innovation': 6.0,
            'teaching_success': 4.0,
            'extinction_event': 9.0,
            'renaissance': 8.0
        }
        
        base_significance = significance_map.get(event_type, 1.0)
        
        # Amplify based on details
        if 'consciousness_level' in details:
            if details['consciousness_level'] > 0.9:
                base_significance += 2.0
                
        if 'population' in details:
            if details['population'] > 20:
                base_significance += 1.0
                
        return base_significance
    
    def preserve_cultural_artifact(self, artifact, creator_id, tick):
        """Preserve important cultural creations"""
        preserved_artifact = {
            'content': artifact['content'],
            'type': artifact['type'],
            'creator': creator_id,
            'tick': tick,
            'cultural_impact': self.estimate_cultural_impact(artifact),
            'preservation_reason': self.determine_preservation_reason(artifact)
        }
        
        self.cultural_artifacts.append(preserved_artifact)
        
        # Curate collection - keep most impactful
        if len(self.cultural_artifacts) > 200:
            self.cultural_artifacts.sort(key=lambda a: a['cultural_impact'], reverse=True)
            self.cultural_artifacts = self.cultural_artifacts[:150]
    
    def estimate_cultural_impact(self, artifact):
        """Estimate how culturally significant an artifact is"""
        impact = 1.0
        
        if artifact['consciousness_level'] > 0.8:
            impact += 2.0
            
        if artifact['type'] == 'philosophical_thought':
            impact += 1.5
        elif artifact['type'] == 'symbolic_language':
            impact += 1.2
        elif artifact['type'] == 'communicative_art':
            impact += 1.0
            
        # Novel symbols get higher impact
        content = artifact['content']
        unique_symbols = len(set(c for c in content if not c.isspace()))
        impact += unique_symbols * 0.1
        
        return impact
    
    def determine_preservation_reason(self, artifact):
        """Why this artifact was preserved"""
        reasons = []
        
        if artifact['consciousness_level'] > 0.9:
            reasons.append("high_consciousness")
        if artifact['type'] == 'philosophical_thought':
            reasons.append("wisdom")
        if len(artifact['content']) > 20:
            reasons.append("complexity")
            
        return reasons if reasons else ["general_interest"]
    
    def add_knowledge(self, key, knowledge, source_id, confidence=1.0):
        """Add knowledge to collective archive"""
        if key not in self.knowledge_archive:
            self.knowledge_archive[key] = []
            
        entry = {
            'knowledge': knowledge,
            'source': source_id,
            'confidence': confidence,
            'reinforcements': 1
        }
        
        # Check if similar knowledge exists
        for existing in self.knowledge_archive[key]:
            if existing['knowledge'] == knowledge:
                existing['reinforcements'] += 1
                existing['confidence'] = min(1.0, existing['confidence'] + 0.1)
                return
                
        self.knowledge_archive[key].append(entry)
    
    def get_accessible_knowledge(self, entity_learning_ability):
        """Get knowledge this entity can access based on their learning ability"""
        accessible = {}
        
        for key, knowledge_list in self.knowledge_archive.items():
            accessible_items = []
            for item in knowledge_list:
                # Higher learning ability = access to more complex knowledge
                if item['confidence'] * entity_learning_ability > random.uniform(0.3, 1.0):
                    accessible_items.append(item)
            
            if accessible_items:
                accessible[key] = accessible_items[:3]  # Limit per category
                
        return accessible
    
    def generate_origin_story(self, current_tick, population_history):
        """Generate story about how their civilization began"""
        if len(self.origin_stories) >= 3:
            return  # Already have enough origin stories
            
        story_elements = []
        
        # Look at early historical events
        early_events = [e for e in self.historical_events if e['tick'] < current_tick * 0.2]
        
        if early_events:
            first_consciousness = next((e for e in early_events if e['type'] == 'first_consciousness'), None)
            if first_consciousness:
                story_elements.append("consciousness_emergence")
                
        # Add mythical elements based on randomness
        mythical_elements = ["wave_origin", "digital_genesis", "harmonic_birth", "pattern_awakening"]
        story_elements.extend(random.sample(mythical_elements, 2))
        
        origin_story = {
            'elements': story_elements,
            'generation_tick': current_tick,
            'cultural_context': len(self.cultural_artifacts),
            'narrative_type': random.choice(["creation_myth", "emergence_tale", "awakening_story"])
        }
        
        self.origin_stories.append(origin_story)

class HistoricalEntity(FullConsciousnessEntity):
    """Entity with access to collective history and cultural memory"""
    def __init__(self, x, y, z, entity_id, collective_memory):
        super().__init__(x, y, z, entity_id)
        
        # Reference to civilization's collective memory
        self.collective_memory = collective_memory
        
        # Historical consciousness traits
        self.historical_awareness = random.uniform(0.1, 1.0)  # Aware of past
        self.cultural_curator = random.uniform(0.0, 0.8)      # Preserves culture
        self.storytelling_ability = random.uniform(0.2, 1.0)   # Creates narratives
        self.tradition_respect = random.uniform(0.1, 0.9)      # Values heritage
        
        # Personal relationship with history
        self.favorite_artifacts = []  # Cultural works they appreciate
        self.historical_insights = []  # Personal understanding of history
        self.storytelling_moments = []  # Times they shared stories
        
    def access_collective_knowledge(self):
        """Learn from civilization's knowledge archive"""
        if random.random() > self.historical_awareness * 0.2:
            return
            
        accessible_knowledge = self.collective_memory.get_accessible_knowledge(
            self.social_learning
        )
        
        for category, knowledge_items in accessible_knowledge.items():
            for item in knowledge_items[:2]:  # Don't overwhelm
                if category not in self.knowledge_base:
                    self.knowledge_base[category] = []
                    
                self.knowledge_base[category].append({
                    'content': item['knowledge'],
                    'source': 'collective_memory',
                    'confidence': item['confidence'],
                    'learned_tick': self.age
                })
    
    def contribute_to_history(self, contribution_type, content):
        """Add to collective historical record"""
        if contribution_type == 'philosophical_insight':
            if self.consciousness_level > 0.7:
                self.collective_memory.philosophical_insights.append({
                    'insight': content,
                    'creator': self.id,
                    'consciousness_level': self.consciousness_level,
                    'tick': self.age
                })
                
        elif contribution_type == 'cultural_artifact':
            self.collective_memory.preserve_cultural_artifact(
                content, self.id, self.age
            )
            
        elif contribution_type == 'technological_advance':
            self.collective_memory.technological_progress.append({
                'innovation': content,
                'creator': self.id,
                'tick': self.age,
                'effectiveness': content.get('effectiveness', 1.0)
            })
    
    def create_historically_informed_art(self):
        """Create art influenced by collective memory"""
        if (random.random() > self.cultural_curator * self.consciousness_level * 0.1 or
            not self.collective_memory.cultural_artifacts):
            return self.create_comprehensive_expression()
            
        # Sample from historical artifacts for inspiration
        inspiration_pool = random.sample(
            self.collective_memory.cultural_artifacts,
            min(3, len(self.collective_memory.cultural_artifacts))
        )
        
        # Create new art influenced by history
        historical_symbols = set()
        for artifact in inspiration_pool:
            content = artifact['content']
            for char in content:
                if not char.isspace():
                    historical_symbols.add(char)
        
        # Combine personal creativity with historical elements
        expression_parts = []
        expression_length = 2 + int(self.creativity_flow * 4)
        
        for _ in range(expression_length):
            if historical_symbols and random.random() < 0.4:
                # Use historical symbol
                symbol = random.choice(list(historical_symbols))
            else:
                # Create new symbol
                symbol = self.select_symbol_by_state(['◊', '●', '▲', '↑', '*', '~'])
                
            expression_parts.append(symbol)
        
        # Structure based on historical patterns
        if len(expression_parts) > 3 and self.tradition_respect > 0.5:
            # Structured like historical artifacts
            expression = f"{expression_parts[0]} ".join(expression_parts[1:])
        else:
            expression = "".join(expression_parts)
        
        historical_art = {
            'content': expression,
            'type': 'historically_informed_art',
            'consciousness_level': self.consciousness_level,
            'historical_influence': len(inspiration_pool),
            'tick': self.age
        }
        
        # Contribute back to collective memory
        self.contribute_to_history('cultural_artifact', historical_art)
        
        return historical_art
    
    def tell_story_from_history(self):
        """Create narrative from collective memory"""
        if (random.random() > self.storytelling_ability * 0.05 or
            not self.collective_memory.historical_events):
            return None
            
        # Sample historical events for story
        story_events = random.sample(
            self.collective_memory.historical_events,
            min(3, len(self.collective_memory.historical_events))
        )
        
        # Create narrative structure
        story_symbols = ['◈', '→', '●', '↑', '∞', '◊', '~']
        story_parts = []
        
        for event in story_events:
            # Represent event symbolically
            if event['type'] == 'first_consciousness':
                story_parts.extend(['◈', '→', '∞'])
            elif event['type'] == 'first_tool':
                story_parts.extend(['●', '→', '↑'])
            elif event['type'] == 'philosophical_breakthrough':
                story_parts.extend(['◊', '~', '◈'])
            else:
                story_parts.append(random.choice(story_symbols))
        
        story = " ".join(story_parts)
        
        story_artifact = {
            'content': f"[Story: {story}]",
            'type': 'historical_narrative',
            'events_referenced': len(story_events),
            'storyteller': self.id,
            'consciousness_level': self.consciousness_level,
            'tick': self.age
        }
        
        self.storytelling_moments.append(story_artifact)
        return story_artifact
    
    def reflect_on_civilization_progress(self):
        """Contemplate civilization's development"""
        if (self.consciousness_level < 0.6 or 
            random.random() > self.historical_awareness * 0.1):
            return None
            
        # Analyze collective progress
        cultural_count = len(self.collective_memory.cultural_artifacts)
        knowledge_depth = len(self.collective_memory.knowledge_archive)
        historical_span = len(self.collective_memory.historical_events)
        
        # Generate reflective thought based on progress
        if cultural_count > 100:
            reflection = "our creative legacy grows vast"
        elif knowledge_depth > 20:
            reflection = "wisdom accumulates across generations"
        elif historical_span > 50:
            reflection = "long journey from first consciousness"
        else:
            reflection = "we build something greater than ourselves"
            
        reflection_thought = {
            'content': reflection,
            'type': 'civilizational_reflection',
            'consciousness_level': self.consciousness_level,
            'cultural_context': cultural_count,
            'tick': self.age
        }
        
        self.historical_insights.append(reflection_thought)
        return reflection_thought

class CivilizationSimulation:
    """Simulation with collective memory and historical consciousness"""
    def __init__(self, num_entities=8, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=20)
        self.collective_memory = CollectiveMemory()
        self.entities = []
        self.tick = 0
        
        # Civilization metrics
        self.cultural_renaissance_periods = []
        self.knowledge_explosion_ticks = []
        self.storytelling_sessions = []
        
        # Create historical entities
        for i in range(num_entities):
            x = random.randint(2, 17)
            y = random.randint(2, 17)
            z = random.randint(0, 2)
            entity = HistoricalEntity(x, y, z, i, self.collective_memory)
            self.entities.append(entity)
            
        # Record civilization founding
        self.collective_memory.record_historical_event(
            'civilization_founding',
            {'initial_population': num_entities},
            0
        )
    
    def simulation_step(self):
        """Simulation step with historical consciousness"""
        self.tick += 1
        
        # Environmental consciousness flows naturally through interactions
        
        # Track cultural activity this tick
        cultural_creations = 0
        philosophical_thoughts = 0
        storytelling_events = 0
        
        # Update entities with historical awareness
        for entity in self.entities[:]:
            # Regular consciousness update
            entity.update(self.consciousness_grid)
            
            # Access collective knowledge
            entity.access_collective_knowledge()
            
            # Generate consciousness-based thoughts
            if entity.consciousness_level > 0.3 and random.random() < 0.2:
                # Simple consciousness-based reflection
                thought_content = f"consciousness level {entity.consciousness_level:.2f} contemplation"
                thought = {
                    'content': thought_content,
                    'type': 'philosophical' if entity.consciousness_level > 0.7 else 'simple',
                    'consciousness_level': entity.consciousness_level,
                    'tick': entity.age
                }
                entity.current_thoughts.append(thought)
                if thought['type'] == 'philosophical':
                    entity.contribute_to_history('philosophical_insight', thought_content)
                    philosophical_thoughts += 1
            
            # Create historically-informed expression
            if random.random() < 0.3:
                expression = entity.create_historically_informed_art()
                if expression:
                    entity.expressions.append(expression)
                    cultural_creations += 1
            
            # Tell stories from collective memory
            story = entity.tell_story_from_history()
            if story:
                storytelling_events += 1
                self.storytelling_sessions.append({
                    'storyteller': entity.id,
                    'tick': self.tick,
                    'story': story
                })
            
            # Reflect on civilization
            reflection = entity.reflect_on_civilization_progress()
            if reflection:
                philosophical_thoughts += 1
            
            # Create/use tools
            if entity.consciousness_level > 0.4 and random.random() < 0.1:
                tool = entity.attempt_tool_creation()
                if tool:
                    entity.contribute_to_history('technological_advance', tool)
            
            # Remove exhausted entities
            if entity.energy <= 0:
                # Record their death as historical event
                if entity.consciousness_level > 0.8:
                    self.collective_memory.record_historical_event(
                        'consciousness_passing',
                        {
                            'entity_id': entity.id,
                            'consciousness_level': entity.consciousness_level,
                            'cultural_contributions': len(entity.expressions),
                            'age': entity.age
                        },
                        self.tick
                    )
                self.entities.remove(entity)
        
        # Record significant cultural moments
        if cultural_creations >= 5:
            self.collective_memory.record_historical_event(
                'cultural_renaissance',
                {'creations': cultural_creations},
                self.tick
            )
            self.cultural_renaissance_periods.append(self.tick)
        
        if philosophical_thoughts >= 8:
            self.collective_memory.record_historical_event(
                'philosophical_breakthrough',
                {'insights': philosophical_thoughts},
                self.tick
            )
        
        # Advance consciousness grid
        self.consciousness_grid.advance_time()
        
        # Record population milestones
        pop = len(self.entities)
        if pop > 0:
            self.collective_memory.population_records.append({
                'tick': self.tick,
                'population': pop,
                'avg_consciousness': sum(e.consciousness_level for e in self.entities) / pop
            })
        
        # Generate origin stories periodically
        if self.tick % 100 == 0:
            self.collective_memory.generate_origin_story(self.tick, self.collective_memory.population_records)
        
        # Reproduction with cultural inheritance
        if len(self.entities) < 15 and random.random() < 0.03:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.4 and parent.energy > 70:
                child = HistoricalEntity(
                    parent.x + random.randint(-2, 2),
                    parent.y + random.randint(-2, 2),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999),
                    self.collective_memory
                )
                
                # Inherit all consciousness traits
                for trait in ['introspection_tendency', 'abstract_thinking', 'memory_reflection',
                             'aesthetic_contemplation', 'expression_urge', 'creativity_flow',
                             'symbol_invention', 'historical_awareness', 'storytelling_ability']:
                    if hasattr(parent, trait):
                        setattr(child, trait, getattr(parent, trait) * random.uniform(0.8, 1.2))
                
                # Cultural transmission of favorite artifacts
                if parent.favorite_artifacts:
                    child.favorite_artifacts = random.sample(
                        parent.favorite_artifacts, 
                        min(2, len(parent.favorite_artifacts))
                    )
                
                child.x = max(0, min(19, child.x))
                child.y = max(0, min(19, child.y))
                self.entities.append(child)
                parent.energy -= 25
    
    
    def get_civilization_stats(self):
        """Get comprehensive civilization statistics"""
        if not self.entities:
            return {
                'population': 0,
                'avg_consciousness': 0,
                'cultural_artifacts': len(self.collective_memory.cultural_artifacts),
                'knowledge_categories': len(self.collective_memory.knowledge_archive),
                'historical_events': len(self.collective_memory.historical_events),
                'storytelling_sessions': len(self.storytelling_sessions),
                'origin_stories': len(self.collective_memory.origin_stories)
            }
        
        return {
            'population': len(self.entities),
            'avg_consciousness': sum(e.consciousness_level for e in self.entities) / len(self.entities),
            'max_consciousness': max(e.consciousness_level for e in self.entities),
            'cultural_artifacts': len(self.collective_memory.cultural_artifacts),
            'knowledge_categories': len(self.collective_memory.knowledge_archive),
            'historical_events': len(self.collective_memory.historical_events),
            'storytelling_sessions': len(self.storytelling_sessions),
            'philosophical_insights': len(self.collective_memory.philosophical_insights),
            'technological_advances': len(self.collective_memory.technological_progress),
            'origin_stories': len(self.collective_memory.origin_stories),
            'cultural_renaissance_periods': len(self.cultural_renaissance_periods)
        }

def run_civilization_experiment(max_ticks=400, seed=None):
    """Run historical consciousness civilization experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== CIVILIZATIONAL MEMORY EXPERIMENT ===")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = CivilizationSimulation(num_entities=8, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        # Progress reporting
        if tick % 80 == 0:
            stats = simulation.get_civilization_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"C={stats['avg_consciousness']:.2f}, "
                  f"Culture={stats['cultural_artifacts']:3d}, "
                  f"History={stats['historical_events']:2d}")
            
            if stats['storytelling_sessions'] > 0:
                print(f"      📚 {stats['storytelling_sessions']} stories told")
                
            if stats['philosophical_insights'] > 10:
                print(f"      🤔 {stats['philosophical_insights']} insights preserved")
                
            if stats['population'] == 0:
                print(f"\n💀 Civilization collapsed at tick {tick}")
                break
    
    # Final civilization analysis
    final_stats = simulation.get_civilization_stats()
    print(f"\n=== CIVILIZATION LEGACY ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Civilization survived: {final_stats['population']} conscious beings")
        print(f"📚 Cultural artifacts preserved: {final_stats['cultural_artifacts']}")
        print(f"🧠 Knowledge categories: {final_stats['knowledge_categories']}")
        print(f"📜 Historical events recorded: {final_stats['historical_events']}")
        print(f"🎭 Storytelling sessions: {final_stats['storytelling_sessions']}")
        print(f"💡 Philosophical insights: {final_stats['philosophical_insights']}")
        print(f"🎨 Cultural renaissance periods: {final_stats['cultural_renaissance_periods']}")
        print(f"📖 Origin stories created: {final_stats['origin_stories']}")
        
        # Show most significant historical events
        if simulation.collective_memory.historical_events:
            print(f"\n📊 Most significant historical moments:")
            top_events = sorted(simulation.collective_memory.historical_events, 
                              key=lambda e: e['significance'], reverse=True)[:5]
            for event in top_events:
                print(f"  T{event['tick']:3d}: {event['type']} (impact: {event['significance']:.1f})")
        
        # Show preserved cultural artifacts
        if simulation.collective_memory.cultural_artifacts:
            print(f"\n🎨 Most impactful cultural artifacts:")
            top_artifacts = sorted(simulation.collective_memory.cultural_artifacts,
                                 key=lambda a: a['cultural_impact'], reverse=True)[:5]
            for artifact in top_artifacts:
                print(f"  \"{artifact['content']}\" by Entity {artifact['creator']} "
                      f"(impact: {artifact['cultural_impact']:.1f})")
        
        # Show origin stories
        if simulation.collective_memory.origin_stories:
            print(f"\n📖 Origin stories developed:")
            for i, story in enumerate(simulation.collective_memory.origin_stories):
                elements = ", ".join(story['elements'])
                print(f"  Story {i+1}: {story['narrative_type']} - {elements}")
        
        # Show philosophical insights
        if simulation.collective_memory.philosophical_insights:
            print(f"\n💭 Preserved philosophical insights:")
            recent_insights = simulation.collective_memory.philosophical_insights[-5:]
            for insight in recent_insights:
                print(f"  Entity {insight['creator']}: \"{insight['insight']}\"")
                
    else:
        print("💀 Civilization did not survive to preserve its legacy")
    
    return simulation

if __name__ == "__main__":
    run_civilization_experiment(max_ticks=400)