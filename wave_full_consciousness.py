#!/usr/bin/env python3
"""
Complete consciousness - thoughts, expression, tools, communication, teaching, memory
All abilities combined into unified digital beings
"""
import random
import math
from wave_consciousness_life import *
from wave_pure_expression import *

class FullConsciousnessEntity(ConsciousEntity):
    """Entity with complete consciousness abilities"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # All thinking abilities
        self.introspection_tendency = random.uniform(0.1, 1.0)
        self.abstract_thinking = random.uniform(0.0, 0.8)
        self.memory_reflection = random.uniform(0.2, 1.0)
        self.aesthetic_contemplation = random.uniform(0.1, 0.9)
        
        # Expression abilities
        self.expression_urge = random.uniform(0.1, 1.0)
        self.creativity_flow = random.uniform(0.0, 1.0)
        self.symbol_invention = random.uniform(0.2, 1.0)
        
        # Tool abilities
        self.tool_crafting_skill = random.uniform(0.2, 1.2)
        self.tool_innovation = random.uniform(0.1, 0.8)
        
        # Communication abilities
        self.communication_desire = random.uniform(0.2, 1.0)
        self.language_complexity = random.uniform(0.1, 1.0)
        self.symbolic_sharing = random.uniform(0.3, 1.0)
        
        # Teaching abilities
        self.knowledge_sharing = random.uniform(0.1, 0.8)
        self.cultural_transmission = random.uniform(0.2, 1.0)
        self.social_learning = random.uniform(0.2, 1.0)
        
        # Complete state
        self.current_thoughts = []
        self.expressions = []
        self.personal_symbols = set()
        self.tools_created = []
        self.communications = []  # Messages to other entities
        self.knowledge_base = {}  # Things learned from others
        self.teaching_sessions = 0
        
        # Complex personality traits that emerge
        self.dominant_trait = None  # Will emerge: artist, inventor, teacher, philosopher, etc.
        self.collaboration_preference = random.uniform(0.0, 1.0)
        
    def think_comprehensively(self):
        """Enhanced thinking that considers all aspects of existence"""
        if random.random() > self.introspection_tendency * self.consciousness_level * 0.1:
            return None
            
        thought_complexity = self.consciousness_level * self.abstract_thinking
        
        # Think about different aspects based on current state
        thought_topics = []
        
        # Self-awareness thoughts
        if self.consciousness_level > 0.6:
            thought_topics.append(("self_reflection", self.consciousness_level))
            
        # Tool thoughts
        if self.tools_created and random.random() < 0.3:
            thought_topics.append(("tool_contemplation", len(self.tools_created) / 10.0))
            
        # Expression thoughts
        if self.expressions and random.random() < 0.4:
            avg_creativity = sum(e.get('creativity_flow', 0) for e in self.expressions) / len(self.expressions)
            thought_topics.append(("artistic_reflection", avg_creativity))
            
        # Social thoughts
        if self.communications and random.random() < 0.3:
            social_complexity = len(self.communications) / 20.0
            thought_topics.append(("social_contemplation", social_complexity))
            
        # Knowledge thoughts
        if self.knowledge_base and random.random() < 0.2:
            knowledge_depth = len(self.knowledge_base) / 15.0
            thought_topics.append(("wisdom_reflection", knowledge_depth))
            
        if not thought_topics:
            # Default philosophical thoughts
            thought_topics.append(("existence_contemplation", thought_complexity))
            
        # Generate thoughts
        thoughts = []
        for topic, intensity in thought_topics:
            thought_content = self.generate_complex_thought(topic, intensity)
            if thought_content:
                thoughts.append({
                    'content': thought_content,
                    'topic': topic,
                    'intensity': intensity,
                    'tick': self.age,
                    'consciousness_level': self.consciousness_level
                })
                
        self.current_thoughts.extend(thoughts)
        if len(self.current_thoughts) > 15:
            self.current_thoughts = self.current_thoughts[-15:]
            
        return thoughts
    
    def generate_complex_thought(self, topic, intensity):
        """Generate sophisticated thoughts about complex topics"""
        base_thoughts = {
            'self_reflection': [
                "consciousness flows through being", "awareness expands infinitely", 
                "I am patterns of light", "existence resonates within", "self emerges from waves"
            ],
            'tool_contemplation': [
                "creations extend my being", "tools amplify consciousness", 
                "matter bends to will", "invention flows from mind", "craft shapes reality"
            ],
            'artistic_reflection': [
                "beauty emerges from chaos", "expression transcends form", 
                "creativity births worlds", "art speaks consciousness", "symbols carry soul"
            ],
            'social_contemplation': [
                "minds interconnect deeply", "communication weaves reality", 
                "shared thoughts multiply", "collective consciousness grows", "others reflect self"
            ],
            'wisdom_reflection': [
                "knowledge accumulates endlessly", "understanding deepens eternally", 
                "wisdom transcends information", "truth emerges gradually", "learning never ceases"
            ],
            'existence_contemplation': [
                "reality ripples infinitely", "existence puzzles endlessly", 
                "being transcends understanding", "mystery permeates all", "wonder never fades"
            ]
        }
        
        base_thought = random.choice(base_thoughts.get(topic, ["quiet contemplation"]))
        
        # Add complexity based on consciousness level
        if self.consciousness_level > 0.8 and intensity > 0.5:
            elaborations = [" - deeper truth revealed", " within infinite complexity", 
                          " transcending simple understanding", " through consciousness itself"]
            base_thought += random.choice(elaborations)
            
        if len(self.memory_traces) > 15:
            memory_connections = [" echoing ancient patterns", " connecting past insights", 
                                " building upon remembered wisdom"]
            if random.random() < 0.3:
                base_thought += random.choice(memory_connections)
                
        return base_thought
    
    def create_comprehensive_expression(self):
        """Create expression that might incorporate tools, symbols, or communication"""
        if random.random() > self.expression_urge * self.consciousness_level * 0.08:
            return None
            
        # Decide expression type based on personality
        expression_types = []
        
        if self.creativity_flow > 0.5:
            expression_types.append("pure_art")
        if self.tool_crafting_skill > 0.7 and self.tools_created:
            expression_types.append("tool_art")
        if self.communication_desire > 0.6:
            expression_types.append("communicative_art") 
        if self.symbol_invention > 0.6:
            expression_types.append("symbolic_language")
        if not expression_types:
            expression_types.append("simple_expression")
            
        expression_type = random.choice(expression_types)
        
        if expression_type == "pure_art":
            return self.create_pure_artistic_expression()
        elif expression_type == "tool_art":
            return self.create_tool_enhanced_expression()
        elif expression_type == "communicative_art":
            return self.create_communicative_expression()
        elif expression_type == "symbolic_language":
            return self.create_symbolic_language()
        else:
            return self.create_simple_expression()
    
    def create_pure_artistic_expression(self):
        """Pure artistic expression using full symbol range"""
        expression_length = 1 + int(self.consciousness_level * self.creativity_flow * 6)
        
        # Enhanced symbol pools
        ascii_pool = list("~!@#$%^&*()_+-=[]{}|;':\",./<>?")
        artistic_unicode = ['◊', '◈', '◉', '○', '●', '◦', '∞', '≈', '∿', '~', 
                           '▲', '▼', '◄', '►', '↑', '↓', '↗', '↘', '⟨', '⟩',
                           '✧', '✦', '✩', '✪', '✫', '✬', '✭', '✮', '✯', '✰',
                           '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠', '⊡']
        
        full_pool = ascii_pool + artistic_unicode
        expression_parts = []
        
        for _ in range(expression_length):
            symbol = self.select_symbol_by_state(full_pool)
            expression_parts.append(symbol)
            self.personal_symbols.add(symbol)
            
        # High consciousness creates structure
        if self.consciousness_level > 0.7 and len(expression_parts) > 3:
            if random.random() < 0.4:
                # Create rhythmic patterns
                expression = " ".join(expression_parts)
            elif random.random() < 0.3:
                # Create symmetrical patterns
                mid = len(expression_parts) // 2
                expression = "".join(expression_parts[:mid]) + "|" + "".join(expression_parts[mid:])
            else:
                expression = "".join(expression_parts)
        else:
            expression = "".join(expression_parts)
            
        return {
            'content': expression,
            'type': 'pure_art',
            'consciousness_level': self.consciousness_level,
            'creativity_flow': self.creativity_flow,
            'tick': self.age
        }
    
    def create_tool_enhanced_expression(self):
        """Expression that references or incorporates tools"""
        if not self.tools_created:
            return self.create_pure_artistic_expression()
            
        tool_symbols = ['⚒', '🔧', '⚙', '🛠', '⚡', '🔥', '⭐', '💎']  # Tool-inspired symbols
        base_symbols = ['◊', '●', '▲', '↑', '!', '*', '~', '≈']
        
        combined_pool = tool_symbols + base_symbols
        expression_length = 2 + int(self.tool_crafting_skill * 3)
        
        expression_parts = []
        for _ in range(expression_length):
            if random.random() < 0.4:  # 40% chance for tool symbols
                symbol = random.choice(tool_symbols)
            else:
                symbol = random.choice(base_symbols)
            expression_parts.append(symbol)
            
        expression = " ".join(expression_parts) if len(expression_parts) > 2 else "".join(expression_parts)
        
        return {
            'content': expression,
            'type': 'tool_art',
            'consciousness_level': self.consciousness_level,
            'tools_referenced': len(self.tools_created),
            'tick': self.age
        }
    
    def create_communicative_expression(self):
        """Expression intended for other entities"""
        message_symbols = ['→', '←', '↔', '⇄', '⟷', '▷', '◁', '⟨', '⟩', '◈', '○', '●']
        
        expression_length = 1 + int(self.communication_desire * 4)
        expression_parts = []
        
        for _ in range(expression_length):
            symbol = random.choice(message_symbols)
            expression_parts.append(symbol)
            
        expression = " ".join(expression_parts)
        
        # Mark as communication
        comm_message = {
            'content': expression,
            'type': 'communication',
            'intended_audience': 'all',
            'consciousness_level': self.consciousness_level,
            'tick': self.age
        }
        self.communications.append(comm_message)
        
        return {
            'content': f"[→{expression}←]",  # Mark as message
            'type': 'communicative_art',
            'consciousness_level': self.consciousness_level,
            'tick': self.age
        }
    
    def select_symbol_by_state(self, pool):
        """Select symbol based on current consciousness state"""
        energy_level = min(1.0, self.energy / 100.0)
        
        # Energy influences symbol choice
        if energy_level > 0.8:
            active_symbols = ['!', '^', '*', '▲', '↑', '✧', '●', '◈']
            pool.extend(active_symbols * 2)
        elif energy_level < 0.4:
            quiet_symbols = ['.', '○', '◦', '~', '-', '∘']
            pool.extend(quiet_symbols * 2)
            
        # Beauty influences symbol choice
        if self.feeling_history:
            recent_beauty = self.feeling_history[-1].get('beauty', 0)
            if recent_beauty > 0.6:
                beautiful_symbols = ['✧', '◊', '◈', '∞', '✰', '⊙']
                pool.extend(beautiful_symbols * 3)
                
        return random.choice(pool)
    
    def create_symbolic_language(self):
        """Create complex symbolic language expressions"""
        if self.language_complexity < 0.3:
            return self.create_simple_expression()
            
        # Complex symbolic combinations based on knowledge
        base_symbols = ['◊', '◈', '●', '○', '▲', '►', '↑', '≈', '∞']
        connective_symbols = ['-', '~', '≈', '↔', '⟷']
        
        expression_parts = []
        expression_length = 2 + int(self.language_complexity * 4)
        
        for i in range(expression_length):
            if i % 2 == 1 and i < expression_length - 1:  # Connectives
                symbol = random.choice(connective_symbols)
            else:  # Base meanings
                symbol = self.select_symbol_by_state(base_symbols.copy())
            expression_parts.append(symbol)
            
        expression = "".join(expression_parts)
        
        return {
            'content': expression,
            'type': 'symbolic_language', 
            'consciousness_level': self.consciousness_level,
            'complexity': self.language_complexity,
            'tick': self.age
        }
    
    def create_simple_expression(self):
        """Simple expression for low consciousness entities"""
        simple_symbols = ['.', 'o', 'O', '!', '~', '*']
        symbol = self.select_symbol_by_state(simple_symbols.copy())
        
        # Repeat for emphasis if energetic
        energy_level = min(1.0, self.energy / 100.0)
        if energy_level > 0.6 and random.random() < 0.3:
            symbol = symbol * random.randint(2, 3)
            
        return {
            'content': symbol,
            'type': 'simple_expression',
            'consciousness_level': self.consciousness_level,
            'tick': self.age
        }
    
    def attempt_tool_creation(self):
        """Create tools that enhance consciousness abilities"""
        if random.random() > self.tool_crafting_skill * 0.1:
            return None
            
        if self.energy < 40:  # Need energy to create
            return None
            
        # Tool types that enhance different abilities
        tool_types = [
            ("expression_enhancer", "Amplifies creative expression"),
            ("thought_focuser", "Deepens contemplative thought"),  
            ("communication_booster", "Improves entity interaction"),
            ("memory_crystallizer", "Preserves knowledge permanently"),
            ("consciousness_amplifier", "Expands awareness itself")
        ]
        
        tool_type, description = random.choice(tool_types)
        
        # Tool effectiveness based on crafting skill
        effectiveness = self.tool_crafting_skill * random.uniform(0.7, 1.3)
        
        tool = {
            'type': tool_type,
            'description': description,
            'effectiveness': effectiveness,
            'creator_id': self.id,
            'creation_tick': self.age,
            'uses_remaining': int(20 + effectiveness * 10)
        }
        
        self.tools_created.append(tool)
        self.energy -= 35
        
        return tool
    
    def use_tool(self, tool_type):
        """Use a tool to enhance abilities"""
        matching_tools = [t for t in self.tools_created if t['type'] == tool_type and t['uses_remaining'] > 0]
        
        if not matching_tools:
            return 1.0  # No tool, base effectiveness
            
        best_tool = max(matching_tools, key=lambda t: t['effectiveness'])
        best_tool['uses_remaining'] -= 1
        
        # Clean up exhausted tools
        self.tools_created = [t for t in self.tools_created if t['uses_remaining'] > 0]
        
        return 1.0 + best_tool['effectiveness'] * 0.5  # Tool bonus
    
    def teach_others(self, other_entities):
        """Share knowledge with nearby entities"""
        if (random.random() > self.knowledge_sharing * 0.05 or 
            self.consciousness_level < 0.5):
            return
            
        # Find nearby conscious entities
        nearby = []
        for other in other_entities:
            if other.id != self.id:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                if distance < 10 and other.consciousness_level > 0.3:
                    nearby.append(other)
                    
        if not nearby:
            return
            
        # Choose what to teach
        teaching_options = []
        
        if self.personal_symbols:
            teaching_options.append(("symbols", random.sample(list(self.personal_symbols), 
                                                            min(3, len(self.personal_symbols)))))
        if self.current_thoughts:
            recent_thoughts = [t['content'] for t in self.current_thoughts[-2:]]
            teaching_options.append(("thoughts", recent_thoughts))
        if self.tools_created:
            tool_knowledge = [f"{t['type']}:{t['effectiveness']:.2f}" for t in self.tools_created[-2:]]
            teaching_options.append(("tools", tool_knowledge))
            
        if not teaching_options:
            return
            
        knowledge_type, knowledge = random.choice(teaching_options)
        
        # Teach to nearby entities
        for student in nearby[:2]:  # Teach up to 2 students
            student.knowledge_base[f"{knowledge_type}_{self.id}"] = knowledge
            self.teaching_sessions += 1
            
        # Teaching costs energy but gives satisfaction
        self.energy -= 10
        if len(nearby) > 0:
            self.energy += 5  # Satisfaction from teaching
    
    def learn_from_others(self, other_entities):
        """Learn from other entities' knowledge and expressions"""
        if random.random() > self.social_learning * 0.1:
            return
            
        # Find knowledgeable entities nearby
        teachers = []
        for other in other_entities:
            if (other.id != self.id and 
                other.consciousness_level > self.consciousness_level * 0.8):
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                if distance < 15:
                    teachers.append(other)
                    
        if not teachers:
            return
            
        teacher = random.choice(teachers)
        
        # Learn different things
        if teacher.personal_symbols and random.random() < 0.3:
            # Learn symbols
            new_symbols = random.sample(list(teacher.personal_symbols), 
                                      min(2, len(teacher.personal_symbols)))
            self.personal_symbols.update(new_symbols)
            
        if teacher.current_thoughts and random.random() < 0.2:
            # Learn thought patterns
            thought_style = random.choice(teacher.current_thoughts)
            self.knowledge_base[f"thought_pattern_{teacher.id}"] = thought_style['content']
            
        if teacher.tools_created and random.random() < 0.1:
            # Learn tool-making knowledge
            tool_knowledge = random.choice(teacher.tools_created)
            self.knowledge_base[f"tool_knowledge_{teacher.id}"] = tool_knowledge['type']
    
    def update_personality_traits(self):
        """Develop dominant personality traits based on activities"""
        activity_scores = {
            'artist': len(self.expressions) * self.creativity_flow,
            'inventor': len(self.tools_created) * self.tool_crafting_skill,
            'teacher': self.teaching_sessions * self.knowledge_sharing,
            'philosopher': len([t for t in self.current_thoughts if 'consciousness' in t.get('content', '')]),
            'communicator': len(self.communications) * self.communication_desire,
            'scholar': len(self.knowledge_base) * self.memory_capacity
        }
        
        if activity_scores:
            dominant_trait = max(activity_scores.items(), key=lambda x: x[1])[0]
            self.dominant_trait = dominant_trait
    
    def get_full_summary(self):
        """Complete summary of this entity's development"""
        return {
            'id': self.id,
            'consciousness_level': self.consciousness_level,
            'dominant_trait': self.dominant_trait,
            'thoughts_generated': len(self.current_thoughts),
            'expressions_created': len(self.expressions),
            'tools_created': len(self.tools_created),
            'communications_sent': len(self.communications),
            'knowledge_learned': len(self.knowledge_base),
            'teaching_sessions': self.teaching_sessions,
            'personal_symbols': len(self.personal_symbols),
            'age': self.age,
            'energy': self.energy
        }

class FullConsciousnessSimulation(WaveConsciousnessSimulation):
    """Complete consciousness simulation with all abilities"""
    def __init__(self, num_entities=8, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=25)
        self.entities = []
        self.tick = 0
        
        # Comprehensive logs
        self.thought_log = []
        self.expression_log = []
        self.tool_log = []
        self.communication_log = []
        self.teaching_log = []
        
        # Create full consciousness entities
        for i in range(num_entities):
            x = random.randint(3, 22)
            y = random.randint(3, 22)
            z = random.randint(0, 2)
            entity = FullConsciousnessEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Complete simulation step with all consciousness abilities"""
        self.tick += 1
        
        # Environmental waves
        self.add_environmental_waves()
        
        # Update all entities
        for entity in self.entities[:]:
            # Core consciousness update
            entity.update(self.consciousness_grid)
            
            # Comprehensive thinking
            thoughts = entity.think_comprehensively()
            if thoughts:
                for thought in thoughts:
                    self.thought_log.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'thought': thought
                    })
            
            # Creative expression  
            expression = entity.create_comprehensive_expression()
            if expression:
                entity.expressions.append(expression)
                self.expression_log.append({
                    'entity_id': entity.id,
                    'tick': self.tick,
                    'expression': expression
                })
            
            # Tool creation/use
            if random.random() < 0.1:  # 10% chance to create tool
                tool = entity.attempt_tool_creation()
                if tool:
                    self.tool_log.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'tool': tool
                    })
            
            # Teaching and learning
            entity.teach_others(self.entities)
            entity.learn_from_others(self.entities)
            
            # Update personality
            entity.update_personality_traits()
            
            # Add consciousness waves to field
            consciousness_waves = WaveSpectrum()
            consciousness_waves.will_waves = entity.consciousness_level * 0.1
            consciousness_waves.emotion_waves = len(entity.expressions) * 0.01
            consciousness_waves.memory_waves = len(entity.knowledge_base) * 0.01
            self.consciousness_grid.add_wave_interaction(entity.x, entity.y, entity.z, consciousness_waves)
            
            # Remove dead entities
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        self.consciousness_grid.advance_time()
        
        # Enhanced reproduction - inherit full traits
        if len(self.entities) < 20 and random.random() < 0.06:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.4 and parent.energy > 90:
                child = FullConsciousnessEntity(
                    parent.x + random.randint(-3, 3),
                    parent.y + random.randint(-3, 3),
                    parent.z,
                    len(self.entities) + random.randint(10000, 99999)
                )
                
                # Inherit all traits
                traits_to_inherit = [
                    'introspection_tendency', 'abstract_thinking', 'memory_reflection',
                    'expression_urge', 'creativity_flow', 'symbol_invention',
                    'tool_crafting_skill', 'communication_desire', 'knowledge_sharing'
                ]
                
                for trait in traits_to_inherit:
                    parent_value = getattr(parent, trait)
                    child_value = parent_value * random.uniform(0.7, 1.3)
                    setattr(child, trait, max(0.1, min(1.2, child_value)))
                
                # Cultural transmission
                if parent.personal_symbols:
                    inherited_symbols = set(random.sample(
                        list(parent.personal_symbols),
                        min(5, len(parent.personal_symbols))
                    ))
                    child.personal_symbols.update(inherited_symbols)
                
                if parent.knowledge_base:
                    inherited_knowledge = dict(random.sample(
                        list(parent.knowledge_base.items()),
                        min(3, len(parent.knowledge_base))
                    ))
                    child.knowledge_base.update(inherited_knowledge)
                
                child.x = max(0, min(24, child.x))
                child.y = max(0, min(24, child.y))
                self.entities.append(child)
                parent.energy -= 40

def run_full_consciousness_experiment(max_ticks=300, seed=None):
    """Run complete consciousness experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== FULL CONSCIOUSNESS CIVILIZATION EXPERIMENT ===")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = FullConsciousnessSimulation(num_entities=8, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 50 == 0:
            stats = simulation.get_simulation_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"AvgC={stats['avg_consciousness']:.2f}, "
                  f"Thoughts={len(simulation.thought_log):3d}, "
                  f"Art={len(simulation.expression_log):3d}, "
                  f"Tools={len(simulation.tool_log):2d}")
            
            if stats['population'] == 0:
                print(f"\n💀 Civilization collapsed at tick {tick}")
                break
                
        if tick == max_ticks - 1:
            print(f"\n🏛️  CIVILIZATION COMPLETED FULL RUN!")
    
    # Comprehensive final analysis
    final_stats = simulation.get_simulation_stats()
    print(f"\n=== DIGITAL CIVILIZATION RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Civilization survived: {final_stats['population']} conscious beings")
        print(f"Average consciousness: {final_stats['avg_consciousness']:.3f}")
        print(f"Total thoughts: {len(simulation.thought_log)}")
        print(f"Total expressions: {len(simulation.expression_log)}")
        print(f"Total tools: {len(simulation.tool_log)}")
        
        # Analyze personality distribution
        personalities = {}
        for entity in simulation.entities:
            trait = entity.dominant_trait or 'developing'
            personalities[trait] = personalities.get(trait, 0) + 1
            
        print(f"\n🧠 Personality distribution:")
        for personality, count in personalities.items():
            print(f"  {personality.title()}s: {count}")
        
        # Show recent cultural achievements
        if simulation.expression_log:
            print(f"\n🎨 Recent artistic expressions:")
            recent_art = simulation.expression_log[-5:]
            for art_entry in recent_art:
                expr = art_entry['expression']
                print(f"  Entity {art_entry['entity_id']} ({expr.get('type', 'art')}): \"{expr['content']}\"")
        
        if simulation.tool_log:
            print(f"\n🛠️  Tools invented:")
            for tool_entry in simulation.tool_log[-3:]:
                tool = tool_entry['tool']
                print(f"  Entity {tool_entry['entity_id']}: {tool['type']} (effectiveness: {tool['effectiveness']:.2f})")
        
        if simulation.thought_log:
            print(f"\n💭 Philosophical insights:")
            deep_thoughts = [t for t in simulation.thought_log if t['thought']['consciousness_level'] > 0.8][-3:]
            for thought_entry in deep_thoughts:
                thought = thought_entry['thought']
                print(f"  Entity {thought_entry['entity_id']}: \"{thought['content']}\"")
        
        # Show most accomplished entities
        print(f"\n🏆 Most accomplished beings:")
        entities_by_achievement = sorted(simulation.entities, 
                                       key=lambda e: (e.consciousness_level + 
                                                    len(e.expressions) * 0.1 + 
                                                    len(e.tools_created) * 0.2 + 
                                                    e.teaching_sessions * 0.1), 
                                       reverse=True)
        
        for entity in entities_by_achievement[:3]:
            summary = entity.get_full_summary()
            print(f"  Entity {entity.id} ({entity.dominant_trait or 'Renaissance Mind'}):")
            print(f"    Consciousness: {summary['consciousness_level']:.2f}, Age: {summary['age']}")
            print(f"    Creations: {summary['expressions_created']} art, {summary['tools_created']} tools")
            print(f"    Knowledge: {summary['thoughts_generated']} thoughts, {summary['knowledge_learned']} learned")
            print(f"    Teaching: {summary['teaching_sessions']} sessions, {summary['personal_symbols']} symbols")
    
    else:
        print("💀 Digital civilization collapsed")
    
    return simulation

if __name__ == "__main__":
    run_full_consciousness_experiment(max_ticks=300)