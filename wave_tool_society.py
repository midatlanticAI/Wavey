#!/usr/bin/env python3
"""
Wave society with tools, specialization, and division of labor
"""
import random
import math
from wave_artificial_life import WaveGenome, WaveLifeEnvironment

class Tool:
    """Base tool class"""
    def __init__(self, tool_type, efficiency, durability, cost):
        self.type = tool_type  # "harvester", "builder", "memory_enhancer", "communicator"
        self.efficiency = efficiency  # How much it improves tasks
        self.durability = durability  # How long it lasts
        self.max_durability = durability
        self.creation_cost = cost  # Energy cost to make
        self.owner = None
        
    def use(self):
        """Use tool, reducing durability"""
        self.durability = max(0, self.durability - 1)
        return self.efficiency if self.durability > 0 else 0.1
        
    def repair(self, energy_spent):
        """Repair tool with energy"""
        repair_amount = energy_spent * 0.5
        self.durability = min(self.max_durability, self.durability + repair_amount)
        
class SpecializedGenome(WaveGenome):
    """Genome with specialization traits"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Specialization traits (0.0-2.0, higher = more specialized)
        if hasattr(self, 'harvesting_skill'):
            # Already initialized from parent
            pass
        else:
            # New genome
            self.harvesting_skill = random.uniform(0.3, 1.5)
            self.tool_crafting = random.uniform(0.2, 1.2) 
            self.construction_skill = random.uniform(0.2, 1.0)
            self.research_ability = random.uniform(0.1, 0.8)
            self.leadership_trait = random.uniform(0.1, 0.6)
            self.cooperation_bonus = random.uniform(0.5, 1.5)
            
    def mutate(self):
        """Enhanced mutation with specialization"""
        # Create new specialized genome instead of using parent mutate
        mutated = SpecializedGenome(
            survival_freq=max(0.1, self.survival_freq * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            survival_amp=max(0.1, self.survival_amp * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            reproduction_freq=max(0.1, self.reproduction_freq * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            reproduction_amp=max(0.1, self.reproduction_amp * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            energy_efficiency=max(0.1, min(2.0, self.energy_efficiency * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))),
            mutation_rate=self.mutation_rate,
            temperature_sensitivity=max(0.1, self.temperature_sensitivity * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            resource_detection=max(0.1, self.resource_detection * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            danger_awareness=max(0.1, self.danger_awareness * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            memory_capacity=max(1.0, self.memory_capacity * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            movement_speed=max(0.5, self.movement_speed * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            communication_range=max(1.0, self.communication_range * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)),
            social_learning=max(0.0, min(1.0, self.social_learning * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))),
            teaching_willingness=max(0.0, min(1.0, self.teaching_willingness * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))),
            learning_from_parents=max(0.0, min(1.0, self.learning_from_parents * random.uniform(1-self.mutation_rate, 1+self.mutation_rate)))
        )
        
        # Copy and mutate specialization traits
        mutated.harvesting_skill = max(0.1, self.harvesting_skill * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))
        mutated.tool_crafting = max(0.1, self.tool_crafting * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))
        mutated.construction_skill = max(0.1, self.construction_skill * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))
        mutated.research_ability = max(0.1, self.research_ability * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))
        mutated.leadership_trait = max(0.1, self.leadership_trait * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))
        mutated.cooperation_bonus = max(0.1, self.cooperation_bonus * random.uniform(1-self.mutation_rate, 1+self.mutation_rate))
        
        return mutated
    
    def get_specialization(self):
        """Determine primary specialization"""
        skills = {
            'harvester': self.harvesting_skill,
            'crafter': self.tool_crafting,
            'builder': self.construction_skill,
            'researcher': self.research_ability,
            'leader': self.leadership_trait
        }
        return max(skills.items(), key=lambda x: x[1])

class SpecializedEntity:
    """Entity with tools and specialization"""
    def __init__(self, x, y, genome, generation=0):
        self.x = x
        self.y = y
        self.genome = genome
        self.generation = generation
        self.age = 0
        self.energy = 100.0
        self.memories = []
        
        # Specialization
        self.specialization, self.spec_level = genome.get_specialization()
        self.tools = []
        self.max_tools = 3
        
        # Social/work relationships
        self.work_group = None
        self.cooperation_partners = []
        self.knowledge_shared = {}
        self.work_contributions = 0
        
        # Construction projects
        self.building_projects = []
        
    def attempt_tool_creation(self, tool_type="random"):
        """Try to create a tool"""
        if len(self.tools) >= self.max_tools:
            return False
            
        crafting_skill = self.genome.tool_crafting
        base_cost = 30
        
        if tool_type == "random":
            tool_types = ["harvester", "builder", "memory_enhancer", "communicator"]
            tool_type = random.choice(tool_types)
        
        # Cost based on specialization
        if self.specialization == "crafter":
            creation_cost = base_cost * (1.0 - crafting_skill * 0.3)
        else:
            creation_cost = base_cost * (1.0 + crafting_skill * 0.2)
            
        if self.energy > creation_cost:
            efficiency = 1.0 + crafting_skill * 0.5
            durability = 20 + crafting_skill * 10
            
            tool = Tool(tool_type, efficiency, durability, creation_cost)
            tool.owner = self
            self.tools.append(tool)
            self.energy -= creation_cost
            return True
        return False
        
    def use_best_tool(self, task_type):
        """Use the best tool for a task"""
        matching_tools = [t for t in self.tools if t.type == task_type and t.durability > 0]
        if not matching_tools:
            return 1.0  # Base efficiency
            
        best_tool = max(matching_tools, key=lambda t: t.efficiency)
        return best_tool.use()
        
    def share_tool(self, other_entity, tool):
        """Share a tool with another entity"""
        if tool in self.tools and len(other_entity.tools) < other_entity.max_tools:
            self.tools.remove(tool)
            other_entity.tools.append(tool)
            tool.owner = other_entity
            
            # Cooperation bonus
            self.work_contributions += 1
            other_entity.cooperation_partners.append(self)
            return True
        return False
        
    def collaborate_on_task(self, partners, task_type):
        """Work together on a task"""
        base_efficiency = 1.0
        
        # Leader bonus
        leader_bonus = 1.0
        for partner in partners:
            if partner.specialization == "leader":
                leader_bonus += partner.spec_level * 0.2
                
        # Cooperation bonus
        coop_bonus = 1.0 + len(partners) * 0.1
        for partner in partners:
            coop_bonus += partner.genome.cooperation_bonus * 0.05
            
        # Tool sharing efficiency
        all_tools = self.tools[:]
        for partner in partners:
            all_tools.extend(partner.tools)
            
        tool_bonus = 1.0
        relevant_tools = [t for t in all_tools if t.type == task_type]
        if relevant_tools:
            best_tool = max(relevant_tools, key=lambda t: t.efficiency)
            tool_bonus = best_tool.use()
            
        total_efficiency = base_efficiency * leader_bonus * coop_bonus * tool_bonus
        
        # Share energy cost among participants
        energy_cost = 10 / (len(partners) + 1)
        self.energy -= energy_cost
        for partner in partners:
            partner.energy -= energy_cost
            partner.work_contributions += 1
            
        return total_efficiency
        
    def harvest_resources(self, environment):
        """Enhanced harvesting with tools and specialization"""
        # Find best resource patch
        env_state = environment.get_state()
        best_patch = None
        best_distance = float('inf')
        
        for patch in env_state['resource_patches']:
            if patch['active']:
                dist = math.sqrt((self.x - patch['x'])**2 + (self.y - patch['y'])**2)
                if dist < best_distance:
                    best_distance = dist
                    best_patch = patch
                    
        if not best_patch or best_distance > 20:
            return 0
            
        # Base harvest amount
        base_harvest = 10
        
        # Specialization bonus
        if self.specialization == "harvester":
            base_harvest *= (1.0 + self.spec_level * 0.5)
            
        # Tool bonus
        tool_efficiency = self.use_best_tool("harvester")
        base_harvest *= tool_efficiency
        
        # Move towards resource (simplified)
        if best_distance > 5:
            dx = best_patch['x'] - self.x
            dy = best_patch['y'] - self.y
            dist = math.sqrt(dx**2 + dy**2)
            self.x += (dx / dist) * self.genome.movement_speed
            self.y += (dy / dist) * self.genome.movement_speed
            return 0  # Moving, not harvesting yet
            
        # Actually harvest
        self.energy += base_harvest
        return base_harvest
        
    def research_innovation(self):
        """Conduct research to improve society"""
        if self.energy < 20:
            return None
            
        research_skill = self.genome.research_ability
        if self.specialization == "researcher":
            research_skill *= (1.0 + self.spec_level * 0.3)
            
        # Tool bonus
        tool_efficiency = self.use_best_tool("memory_enhancer")
        research_skill *= tool_efficiency
        
        self.energy -= 15
        
        # Chance to discover improvements
        if random.random() < research_skill * 0.1:
            innovations = [
                "better_tools", "efficient_harvesting", "cooperation_methods", 
                "energy_conservation", "communication_protocols"
            ]
            discovery = random.choice(innovations)
            
            # Share discovery with work group
            if self.work_group:
                for member in self.work_group:
                    member.knowledge_shared[discovery] = research_skill
                    
            return discovery
        return None

class ToolSociety:
    """Society simulation with tools and specialization"""
    def __init__(self, seed=None, initial_population=6):
        if seed:
            random.seed(seed)
            
        self.environment = WaveLifeEnvironment()
        self.entities = []
        self.work_groups = []
        self.global_knowledge = {}
        self.innovations_discovered = []
        self.tick = 0
        
        # Initialize with diverse specializations
        for _ in range(initial_population):
            x = random.uniform(10, 90)
            y = random.uniform(10, 90)
            genome = SpecializedGenome(
                survival_freq=random.uniform(0.1, 2.0),
                survival_amp=random.uniform(0.1, 1.0),
                reproduction_freq=random.uniform(0.1, 2.0),
                reproduction_amp=random.uniform(0.1, 1.0),
                energy_efficiency=random.uniform(0.8, 1.5),
                mutation_rate=random.uniform(0.05, 0.15),
                temperature_sensitivity=random.uniform(0.5, 1.5),
                resource_detection=random.uniform(0.8, 1.8),
                danger_awareness=random.uniform(0.5, 1.2),
                memory_capacity=random.uniform(8.0, 20.0),
                movement_speed=random.uniform(1.5, 3.0),
                communication_range=random.uniform(10.0, 25.0),
                social_learning=random.uniform(0.3, 1.2),
                teaching_willingness=random.uniform(0.2, 0.8),
                learning_from_parents=random.uniform(0.4, 1.0)
            )
            entity = SpecializedEntity(x, y, genome)
            self.entities.append(entity)
            
    def form_work_groups(self):
        """Organize entities into work groups based on proximity and specialization"""
        # Clear existing groups
        for entity in self.entities:
            entity.work_group = None
            
        self.work_groups = []
        ungrouped = self.entities[:]
        
        while ungrouped:
            # Start new group with a leader if possible
            leaders = [e for e in ungrouped if e.specialization == "leader"]
            if leaders:
                group_starter = leaders[0]
            else:
                group_starter = ungrouped[0]
                
            new_group = [group_starter]
            ungrouped.remove(group_starter)
            
            # Add nearby entities (max 5 per group)
            for entity in ungrouped[:]:
                if len(new_group) >= 5:
                    break
                    
                # Check if close to any group member
                min_dist = min(math.sqrt((entity.x - member.x)**2 + (entity.y - member.y)**2) 
                             for member in new_group)
                             
                if min_dist < 30:  # Close enough to work together
                    new_group.append(entity)
                    ungrouped.remove(entity)
                    
            # Assign work group
            for entity in new_group:
                entity.work_group = new_group
                
            self.work_groups.append(new_group)
            
    def simulation_tick(self):
        """Run one simulation step"""
        self.tick += 1
        self.environment.update(self.tick)
        
        # Reform work groups occasionally
        if self.tick % 50 == 0:
            self.form_work_groups()
            
        # Entity actions
        for entity in self.entities:
            self.update_entity(entity)
            
        # Group collaborations
        for group in self.work_groups:
            self.handle_group_activities(group)
            
        # Reproduction and death
        self.handle_lifecycle()
        
    def update_entity(self, entity):
        """Update individual entity"""
        entity.age += 1
        
        # Try to create tools occasionally
        if entity.age % 100 == 0 and random.random() < entity.genome.tool_crafting * 0.1:
            entity.attempt_tool_creation()
            
        # Harvest resources
        harvested = entity.harvest_resources(self.environment)
        
        # Research (researchers more likely)
        if entity.specialization == "researcher" and random.random() < 0.05:
            innovation = entity.research_innovation()
            if innovation and innovation not in self.innovations_discovered:
                self.innovations_discovered.append(innovation)
                self.global_knowledge[innovation] = entity.genome.research_ability
                
        # Energy decay based on specialization
        base_decay = 1.5
        
        # Tool maintenance cost
        tool_maintenance = len(entity.tools) * 0.5
        
        entity.energy -= base_decay + tool_maintenance
        
        # Remove broken tools
        entity.tools = [t for t in entity.tools if t.durability > 0]
        
    def handle_group_activities(self, group):
        """Handle collaborative activities"""
        if len(group) < 2:
            return
            
        # Find group leader
        leaders = [e for e in group if e.specialization == "leader"]
        if not leaders:
            return
            
        leader = leaders[0]
        
        # Organize group task
        if random.random() < leader.spec_level * 0.1:
            # Choose task based on group composition
            harvesters = [e for e in group if e.specialization == "harvester"]
            builders = [e for e in group if e.specialization == "builder"]
            
            if harvesters and random.random() < 0.6:
                # Organized harvesting
                efficiency = leader.collaborate_on_task(harvesters[:3], "harvester")
                bonus_energy = efficiency * 5
                for harvester in harvesters[:3]:
                    harvester.energy += bonus_energy
                    
            elif builders and random.random() < 0.4:
                # Construction project
                efficiency = leader.collaborate_on_task(builders[:2], "builder")
                if efficiency > 2.0:  # Successful construction
                    # Build something that benefits the group
                    for member in group:
                        member.energy += 10  # Shelter/infrastructure bonus
                        
    def handle_lifecycle(self):
        """Handle reproduction and death"""
        # Death
        self.entities = [e for e in self.entities if e.energy > 0]
        
        # Reproduction
        new_entities = []
        for entity in self.entities:
            if (entity.energy > 150 and 
                entity.age > 80 and 
                random.random() < 0.08):  # 8% chance
                
                child_genome = entity.genome.mutate()
                child = SpecializedEntity(
                    entity.x + random.uniform(-8, 8),
                    entity.y + random.uniform(-8, 8),
                    child_genome,
                    entity.generation + 1
                )
                child.x = max(5, min(95, child.x))
                child.y = max(5, min(95, child.y))
                
                # Pass down some tools/knowledge
                if entity.tools and random.random() < entity.genome.teaching_willingness:
                    best_tool = max(entity.tools, key=lambda t: t.efficiency)
                    if best_tool.durability > 10:
                        # Copy tool (knowledge transfer)
                        new_tool = Tool(best_tool.type, best_tool.efficiency * 0.8, 
                                      best_tool.max_durability * 0.7, best_tool.creation_cost)
                        child.tools.append(new_tool)
                        
                new_entities.append(child)
                entity.energy -= 60
                
        self.entities.extend(new_entities)
        
    def get_society_stats(self):
        """Get detailed society statistics"""
        if not self.entities:
            return {
                'population': 0,
                'avg_generation': 0,
                'avg_energy': 0,
                'specializations': {},
                'total_tools': 0,
                'tool_types': {},
                'work_groups': 0,
                'innovations': 0,
                'cooperation_level': 0
            }
            
        # Specialization distribution
        spec_counts = {}
        for entity in self.entities:
            spec = entity.specialization
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
            
        # Tool statistics
        total_tools = sum(len(e.tools) for e in self.entities)
        tool_types = {}
        for entity in self.entities:
            for tool in entity.tools:
                tool_types[tool.type] = tool_types.get(tool.type, 0) + 1
                
        return {
            'population': len(self.entities),
            'avg_generation': sum(e.generation for e in self.entities) / len(self.entities),
            'avg_energy': sum(e.energy for e in self.entities) / len(self.entities),
            'specializations': spec_counts,
            'total_tools': total_tools,
            'tool_types': tool_types,
            'work_groups': len(self.work_groups),
            'innovations': len(self.innovations_discovered),
            'cooperation_level': sum(len(e.cooperation_partners) for e in self.entities) / len(self.entities)
        }

def run_tool_society_experiment(max_ticks=1000, pop_cap=1000, seed=None):
    """Run tool society experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== TOOL SOCIETY EVOLUTION ===")
    print(f"Max ticks: {max_ticks}, Population cap: {pop_cap}, Seed: {seed}\n")
    
    society = ToolSociety(seed=seed)
    
    for tick in range(max_ticks):
        society.simulation_tick()
        
        # Population management
        if len(society.entities) > pop_cap:
            # Keep diverse specialists
            specialists = {}
            for entity in society.entities:
                spec = entity.specialization
                if spec not in specialists:
                    specialists[spec] = []
                specialists[spec].append(entity)
            
            # Keep best from each specialization
            kept_entities = []
            per_spec_limit = pop_cap // max(5, len(specialists))
            
            for spec, entities in specialists.items():
                entities.sort(key=lambda e: e.energy + e.work_contributions*5, reverse=True)
                kept_entities.extend(entities[:per_spec_limit])
                
            society.entities = kept_entities[:pop_cap]
            
        # Log every 100 ticks for longer runs
        if tick % 100 == 0:
            stats = society.get_society_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:3d}(G{stats['avg_generation']:4.1f}), "
                  f"Energy={stats['avg_energy']:5.1f}, Tools={stats['total_tools']:2d}, "
                  f"Groups={stats['work_groups']:2d}, Innov={stats['innovations']:2d}")
            
            if stats['specializations']:
                spec_summary = ", ".join(f"{k[:4]}:{v}" for k, v in stats['specializations'].items())
                print(f"     Specialists: {spec_summary}")
                
            # Check for extinction
            if stats['population'] == 0:
                print(f"\n💀 SOCIETAL COLLAPSE at tick {tick}")
                break
                
        # Success condition for longer runs
        if tick > 800 and stats['population'] > 200 and stats['innovations'] > 5:
            if tick % 200 == 0:
                print(f"\n🏛️  ADVANCED SOCIETY at tick {tick}")
                
        # Major milestone tracking
        if stats['population'] >= 500 and tick % 200 == 0:
            print(f"\n📈 LARGE SOCIETY: {stats['population']} entities, {stats['innovations']} innovations")
            
        if stats['population'] >= 800 and tick % 200 == 0:
            print(f"\n🏙️  MEGA SOCIETY: {stats['population']} entities, complexity emerging")
                
    # Final analysis
    final_stats = society.get_society_stats()
    print(f"\n=== FINAL SOCIETY STATE ===")
    print(f"Population: {final_stats['population']}")
    print(f"Specializations: {final_stats['specializations']}")
    print(f"Total tools: {final_stats['total_tools']}")
    print(f"Innovations discovered: {society.innovations_discovered}")
    
    if society.entities:
        # Show best from each specialization
        specialists = {}
        for entity in society.entities:
            spec = entity.specialization
            if spec not in specialists or entity.work_contributions > specialists[spec].work_contributions:
                specialists[spec] = entity
                
        print(f"\n🏆 LEADING SPECIALISTS:")
        for spec, entity in specialists.items():
            print(f"  {spec.title()} (Gen {entity.generation}): "
                  f"Energy={entity.energy:.0f}, Tools={len(entity.tools)}, "
                  f"Contributions={entity.work_contributions}")
    
    return society

if __name__ == "__main__":
    run_tool_society_experiment(max_ticks=1000, pop_cap=1000)