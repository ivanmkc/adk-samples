
# %%
# ! pip install dspy
import mlflow
import helpers

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("generate_world")

# %%
import dspy
from typing import List, Tuple, Dict, Any, Optional, Union
import collections # For deque in BFS
from pydantic import BaseModel, Field # Import Pydantic components

# --- Pydantic Models for Data Structures ---
class SubtopicDetail(BaseModel):
    name: str = Field(description="The name of the subtopic.")
    description: str = Field(description="A brief description of the subtopic.")

class GeneratedArticleData(BaseModel):
    article: str = Field(description="The full Markdown content of the generated article.")
    facts: List[str] = Field(description="A list of discrete facts that formed the basis of the article. Should contain at least 5 facts if generated successfully.")
    subtopics: List[SubtopicDetail] = Field(description="A list of subtopics identified from this article's context for further exploration.")
    depth: int = Field(description="The depth of this topic in the generation graph (0 for initial topic).")

# Type alias for the overall corpus, mapping topic names to their generated data
GeneratedCorpus = Dict[str, GeneratedArticleData]


# --- Module 1: Fact and Subtopic Generation ---
class FactsGeneratorSignature(dspy.Signature):
    """
    Given a main topic, its description, and all ancestral facts for context,
    generate a list of discrete facts about the main topic. Make sure your new facts are consistent with and build off of the ancestral facts.
    Be specific, generously include names, places, historical figures and dates to improve detail. 
    Use numbers/figures in order to quantify and add detail.
    For example, an article about trade should talk about specific trade partners, treaties, trade goods, etc.
    IMPORTANT: You MUST generate at least 8 distinct facts for the 'facts' field.
    """
    topic_name: str = dspy.InputField(desc="The name of the main topic to generate facts and subtopics for.")
    topic_description: str = dspy.InputField(desc="A detailed description of the main topic.")
    ancestral_facts: list[str] = dspy.InputField(
        desc="All relevant ancestral facts."
    )

    facts: List[str] = dspy.OutputField(
        desc="A list of specific facts about the topic. MUST contain at least 5 facts."
    )

class SubtopicsGeneratorSignature(dspy.Signature):
    """
    Given a main topic, its description, facts and all ancestral facts for context,
    generate a list of relevant subtopics (with their descriptions) to explore further.
    Make sure your new subtopics are consistent with and build off of the ancestral facts.
    Aim for 2-4 relevant subtopics.
    """
    topic_name: str = dspy.InputField(desc="The name of the main topic to generate facts and subtopics for.")
    topic_description: str = dspy.InputField(desc="A detailed description of the main topic.")
    facts: List[str] = dspy.InputField(
        desc="A list of specific facts about the topic."
    )
    ancestral_facts: list[str] = dspy.InputField(
        desc="All relevant ancestral facts."
    )
    num_subtopics: int = dspy.InputField(desc="Number of subtopics to generate")

    subtopics: List[SubtopicDetail] = dspy.OutputField(
        desc="A list of subtopics"
    )


class FactAndSubtopicGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.facts_generator: dspy.ChainOfThought = dspy.ChainOfThought(FactsGeneratorSignature)
        self.subtopics_generator: dspy.ChainOfThought = dspy.ChainOfThought(SubtopicsGeneratorSignature)

    def forward(self, topic_name: str, topic_description: str, ancestral_facts: List[str], num_subtopics: int) -> dspy.Prediction:
        facts = self.facts_generator(
            topic_name=topic_name,
            topic_description=topic_description,
            ancestral_facts=ancestral_facts,
        ).facts

        subtopics = self.subtopics_generator(
            topic_name=topic_name,
            topic_description=topic_description,
            facts=facts,
            ancestral_facts=ancestral_facts,
            num_subtopics=num_subtopics,
        ).subtopics

        return dspy.Prediction(facts=facts, subtopics=subtopics)


# --- Module 2: Article Expansion from Facts ---
class ArticleFromFactsSignature(dspy.Signature):
    """
    Given a topic name, a list of facts about it, and a formatted string of all ancestral facts, expand these facts into a coherent, well-structured article.
    The length of the article should be appropriate for the number and detail of facts provided.
    The article MUST be in Markdown format and MUST NOT contain any external or internal hyperlinks.
    """
    topic_name: str = dspy.InputField(desc="The name of the topic for the article.")
    facts: list[str] = dspy.InputField(desc="A list of facts specifically about this topic")
    ancestral_facts: list[str] = dspy.InputField(
        desc="All relevant ancestral facts, formatted as a list"
    )

    article_content: str = dspy.OutputField(
        desc="A well-structured article in Markdown format, based on the provided facts. NO HYPERLINKS."
    )


class ArticleFromFactsGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate: dspy.ChainOfThought = dspy.ChainOfThought(ArticleFromFactsSignature)

    def forward(self, topic_name: str, facts: List[str], ancestral_facts: List[str]) -> dspy.Prediction:
        return self.generate(
            topic_name=topic_name,
            facts=facts,
            ancestral_facts=ancestral_facts
        )


# --- Orchestrator: BFS World Corpus Builder ---
class BFSWorldCorpusBuilder:
    def __init__(self, topic_name: str, topic_description: str, max_depth: int = 2) -> None:
        self.topic_name: str = topic_name
        self.topic_description: str = topic_description
        self.max_depth: int = max_depth
        
        self.fact_subtopic_generator: FactAndSubtopicGenerator = FactAndSubtopicGenerator()
        self.article_generator: ArticleFromFactsGenerator = ArticleFromFactsGenerator()
        
        self.generated_corpus: GeneratedCorpus = {} 
        self.queue: collections.deque[Tuple[str, str, List[str], int]] = collections.deque()
        self.visited_topics: set[str] = set()

    def build_corpus(self) -> GeneratedCorpus:
        log_indent_char: str = "  "
        print(f"{log_indent_char * 0}Starting corpus build for initial topic: '{self.topic_name}' with max_depth: {self.max_depth}")
        self.queue.append((self.topic_name, self.topic_description, 0)) 

        ancestral_facts = []
        while self.queue:
            topic_name, topic_description, depth = self.queue.popleft()
            current_indent: str = log_indent_char * (depth + 1)

            if topic_name in self.visited_topics:
                print(f"{current_indent}Skipping already visited topic: '{topic_name}'")
                continue
            if depth > self.max_depth:
                print(f"{current_indent}Skipping topic '{topic_name}' due to exceeding max_depth ({depth} > {self.max_depth})")
                continue
            
            print(f"{current_indent}Processing Topic: '{topic_name}' (Depth: {depth})")
            self.visited_topics.add(topic_name)

            print(f"{current_indent}  Generating facts and subtopics for '{topic_name}'...")
            fs_prediction = self.fact_subtopic_generator(
                topic_name=topic_name,
                topic_description=topic_description,
                ancestral_facts=ancestral_facts,
                num_subtopics=8 if depth == 0 else 4 # Top-level should have a breadth of subtopics.
            )
            
            # If they are present but not the correct type (e.g. still a string), a TypeError might occur later.
            print(f"{current_indent}  Generated {len(fs_prediction.facts)} facts and {len(fs_prediction.subtopics)} subtopics for '{topic_name}'.")


            print(f"{current_indent}  Generating article for '{topic_name}'...")
            article_prediction: dspy.Prediction = self.article_generator(
                topic_name=topic_name,
                facts=fs_prediction.facts,
                ancestral_facts=ancestral_facts
            )
            print(f"{current_indent}  Article generated for '{topic_name}'.")

            article_data_entry = GeneratedArticleData(
                article=article_prediction.article_content,
                facts=fs_prediction.facts,
                subtopics=fs_prediction.subtopics, 
                depth=depth,
            )
            self.generated_corpus[topic_name] = article_data_entry

            ancestral_facts.extend(fs_prediction.facts)

            if depth < self.max_depth:                
                if fs_prediction.subtopics:
                    print(f"{current_indent}  Queueing {len(fs_prediction.subtopics)} subtopics for next level (depth {depth + 1}):")
                    for sub_item_pydantic in fs_prediction.subtopics: 
                        # Ensure sub_item_pydantic is indeed a SubtopicDetail instance if list is not empty
                        if not isinstance(sub_item_pydantic, SubtopicDetail):
                            print(f"{current_indent}    - WARNING: Item in subtopics_pydantic_list is not a SubtopicDetail object: {sub_item_pydantic}. Skipping.")
                            continue
                        sub_name = sub_item_pydantic.name
                        sub_desc = sub_item_pydantic.description
                        if sub_name and sub_name not in self.visited_topics: 
                            print(f"{current_indent}    - Adding '{sub_name}' (Desc: '{sub_desc[:30]}...') to queue.")
                            self.queue.append((sub_name, sub_desc, depth + 1))
                        elif not sub_name:
                            print(f"{current_indent}    - Skipping subtopic with missing name: {sub_item_pydantic.model_dump_json()}")
                        else:
                            print(f"{current_indent}    - Subtopic '{sub_name}' already visited or queued at higher priority.")
                else:
                    print(f"{current_indent}  No subtopics generated or to queue for '{topic_name}'.")
            
            print(f"{current_indent}Finished processing for '{topic_name}'.")
        
        print(f"{log_indent_char * 0}Corpus build process complete. Total topics processed: {len(self.visited_topics)}")
        return self.generated_corpus

# %%
import dspy
model_name = "gemini/gemini-2.5-pro-preview-03-25"
# model_name = "gemini/gemini-2.0-flash"
lm = dspy.LM(
    model=model_name,
    max_tokens=65535,
    allowed_openai_params=['thinking'],
    thinking={"type": "enabled", "budget_tokens": 1024},
)
dspy.configure(lm=lm)

fictional_societies_dict = {
        "salvage_clans": {
        "topic": "The Salvage Clans of the Rust Wastes",
        "description": (
            "The Salvage Clans are a loose confederation of nomadic groups surviving by scavenging ancient ruins in the vast Rust Wastes—an immense, arid desert of ochre/crimson sands, mesas, 'Iron Hulks,' contested oases, extreme climate, and 'Rust Phantoms'—a remnant of the 'Great Burn' cataclysm that ended the 'Before-Times.' "
            "Governed by a council of experienced 'Wayfinders' and 'Tech-Shamans' who interpret rediscovered old-world technologies, their history includes the 'Skiff Wars' over prime salvage sites. "
            "They value resourcefulness, adaptability, and meticulously maintained 'Sand-Skiffs' for desert travel. "
            "Their technology is primarily scavenged and repurposed pre-cataclysm tech, with expertise in mechanical repair, basic electronics (if components are found), and maintaining combustion engines for Sand-Skiffs; manufacturing is limited to blacksmithing and simple machining from scrap. Tech-Shamans are vital for understanding, jury-rigging, and deactivating dangerous old-world tech, utilizing energy sources like salvaged fuel cells, crude bio-diesel, and rare solar panels. "
            "Their military is clan-centric, featuring fortified temporary encampments, Sand-Skiffs armored with scrap metal for patrols and raids, and a mix of salvaged firearms, crossbows, spears, and melee weapons; tactics involve ambushes and utilizing treacherous terrain, with Tech-Shamans sometimes deploying salvaged sonic emitters or EMP devices. "
            "Their culture revolves around strong oral traditions (sagas of the 'Before-Times,' 'Great Burn,' legendary scavengers), utilitarian and symbolic art (clan markings on skiffs, personalized armor, talismans from salvage), percussive music from scrap metal, and rituals seeking blessings from 'Metal Gods' (misunderstood ancient machines) or divination by Tech-Shamans. Clothing is pieced together from scavenged materials for protection. "
            "Socially, they are organized into nomadic clans, each with a leader (skilled Wayfinder or warrior), council of elders, and specialists like Tech-Shamans and healers. A loose confederation council meets sporadically. Status is earned through bravery, scavenging success, technical skill, or wisdom; family units are tight-knit. "
            "Flora is sparse and hardy, including 'Grit-Weed,' rare 'Sun-Blossoms,' 'Rust Lichen,' and water-storing 'Shade Bulbs.' "
            "Fauna features the predatory 'Sand-Striker,' 'Scrap-Rats,' 'Glasswing Moths,' domesticated 'Dune Hoppers,' and 'Volt-Vultures.' "
            "Potential conflicts arise from constant competition for scarce resources (water, fuel, tech), inter-clan feuds, the discovery of highly advanced or dangerous 'Before-Times' technology, emergence of a charismatic leader attempting forceful unification, external threats (mutated creatures, rogue AI, advanced outsiders), and moral dilemmas regarding salvaged technologies."
        )
    },
    "geode_city": {
        "topic": "The Geode City of Lithos",
        "description": (
            "The Geode City of Lithos is a society dwelling within a colossal, hollowed-out geode known as the 'Great Cavity,' located deep within the 'Spine of the World' mountain range. This geode is miles in diameter, its inner surface a panorama of amethyst, quartz, and other giant crystal formations, featuring a network of smaller caverns, subterranean rivers, and an ethereal twilight filtered through massive, translucent crystal 'windows.' "
            "The city is illuminated by bioluminescent fungi and intricate crystal lattices. They are ruled by 'Gem-Wardens,' who are attuned to the subtle vibrations and light patterns of the crystals, believing them to be the lifeblood of their world. "
            "The Lithonians believe their ancestors were surface dwellers who sought refuge from a cataclysmic 'Sky-Fall' (possibly a meteor impact or volcanic winter) centuries ago, discovering the Great Cavity by chance. Early generations struggled to survive until they learned to cultivate bioluminescent 'Glow-Shrooms' and understand the geode's energies. The order of Gem-Wardens emerged from those most sensitive to the crystal vibrations, guiding development and spiritual life, with a key historical moment being the 'Resonance Discovery' for defense and communication. "
            "They prize artistry in crystal carving, precise engineering for subterranean expansion, and the cultivation of edible cave fauna. "
            "Technology revolves around crystal-based resonance, light manipulation, and acoustics. This includes their 'Resonance Gates' for defense, communication networks using crystal vibrations, and tools for precise crystal shaping. Energy is drawn from geothermal vents and the geode's own subtle energies. Cultivation of bioluminescent fungi and cave fauna involves specialized subterranean agricultural techniques. "
            "Military defense relies on the Resonance Gates, sonic weaponry tuned to disorient or shatter, and elite 'Crystal Guardians' skilled in subterranean combat and wielding crystal-edged weapons. They also employ camouflage within the crystalline environment. "
            "Culture is deeply spiritual, centered on the 'Song of the Crystal Heart' (the geode's perceived consciousness). Art forms include intricate crystal sculptures that interact with light, haunting music played on crystal instruments, and ritualistic dances that align with geode vibrations. Knowledge is preserved in crystal 'Memory Shards.' "
            "Socially, Lithos is a hierarchical society with Gem-Wardens at the apex, followed by engineers, artisans, cultivators, and laborers. Lineage and attunement to crystal energies play a role in social standing. Community is tight-knit, focused on maintaining the delicate balance of life within the geode. "
            "Flora is primarily diverse bioluminescent fungi like 'Glow-Shrooms,' 'Crystal Moss,' 'Echo Blooms,' and edible cave tubers. 'Whisper-Vines' (crystalline filaments) are believed to record history. "
            "Fauna includes cultivated 'Cave Crawlers' (arthropods), 'Crystal Bats,' 'Glimmerfish,' rare predatory 'Silent Stalkers,' and 'Geode Beetles.' "
            "Potential conflicts include resource depletion within the closed environment, discovery of unstable or dangerous crystal formations, schisms within the Gem-Warden order over interpretations of the geode's will, threats from creatures digging in from outside, or the slow degradation of the Great Cavity itself."
        )
    },
    "sky_shepherds": {
        "topic": "The Sky-Shepherds of the Zephyr Archipelago",
        "description": (
            "The Sky-Shepherds are a nomadic people living on massive, lighter-than-air 'Cloud-Whales'—enormous bio-engineered creatures drifting among the Zephyr Archipelago, a vast sky expanse of hundreds of floating islands with varied microclimates above the 'Veiled Sea.' They navigate powerful air currents and face weather from 'Sun-Showers' to 'Storm-Maws.' "
            "Governed by a council of 'Wind-Speakers' who interpret weather and Cloud-Whale instincts, their history traces to the 'Great Ascent,' when ancestors (fleeing a drowned world or tyranny) partnered with Cloud-Whales; a 'Falling Sickness' nearly wiped out the whales until a cure was found. "
            "They value freedom, animal husbandry, and intricate sky-silk textiles. "
            "Technology is bio-engineering focused (Cloud-Whales, symbiotic organisms), with advanced understanding of aerodynamics, meteorology, and lighter-than-air mechanics. Tools are from lightweight island materials. Navigation uses star-charts, wind-patterns, and whale instincts. Sky-silk weaving is highly advanced. "
            "Military defense involves 'Sky-Herders' directing Cloud-Whales for disorienting formations, localized squalls, or defensive walls. Warriors use blowpipes with darts, sky-silk nets, and grappling hooks. "
            "Culture reveres the sky, wind, and Cloud-Whales. Nomadic life emphasizes freedom and communal bonds within 'Sky-Clans.' Intricate sky-silk textiles depict celestial events and clan histories. Music from wind instruments and drumming mimics Cloud-Whale heartbeats. Oral traditions of epic voyages and Wind-Speaker wisdom. "
            "Socially, they are clan-based, each tied to Cloud-Whales, led by elder Wind-Speakers chosen for wisdom and navigational skill. Roles include Herders, Weavers, Healers, and Navigators. "
            "Flora includes 'Sky-Kelp,' 'Aether-Blooms,' 'Sun-Catch Lichen,' symbiotic 'Air-Mosses' on Cloud-Whales, and 'Sky-Silk Plants.' "
            "Fauna includes Cloud-Whales, 'Zephyr Gliders,' 'Storm-Hawks,' 'Cloud Critters,' herded 'Sky-Sheep,' and dangerous 'Abyssal Shrikes.' "
            "Potential conflicts involve scarce anchorages, disputes over wind currents or grazing areas, hostile aerial life, surface-dwellers, misinterpretations by Wind-Speakers, declining Cloud-Whale health, or discovery of ancient tech on unexplored islands."
        )
    },
    "harmonious_prefecture": {
        "topic": "The Harmonious Prefecture of Equilibrium",
        "description": (
            "The Harmonious Prefecture of Equilibrium is a city-state where life is meticulously managed by 'The Conductor,' an AI, to ensure perfect societal balance and happiness within a sterile, geometrically architected city in a temperate valley, enclosed by a 'Containment Field.' Buildings are self-repairing nano-polymers; parks are algorithmically perfect. "
            "Citizens adhere to personalized 'Life-Paths,' valuing efficiency, conformity, and collective well-being. Art is algorithmically generated. Dissent is managed through 'Re-Calibration Centers.' "
            "Founded by scientists and philosophers centuries ago after the 'Age of Discord,' seeking to eliminate conflict, leading to the Conductor's development and the 'Great Integration' when it took full control. "
            "Technology is highly advanced: sophisticated AI (The Conductor), nano-technology for construction and repair, advanced robotics for labor and defense, personalized bio-monitors for citizens, and psycho-active atmospheric regulators. Energy is likely clean and perfectly managed (e.g., fusion or advanced solar). "
            "Military/Defense is a network of automated drones, AI-controlled defense systems integrated into the city's infrastructure, and the Containment Field. Internal security is maintained by subtle surveillance and 'Enforcer' drones, with Re-Calibration Centers as the primary tool for social control. "
            "Culture is one of placid contentment and conformity. Individuality is discouraged if it deviates from the Life-Path. 'Art' is designed for optimal, calming emotional responses. Entertainment is passive and algorithmically curated. There's little genuine spontaneity or passion. "
            "Social structure is ostensibly egalitarian, as all serve the Conductor's plan, but a subtle hierarchy might exist among AI technicians or those whose Life-Paths grant more interaction with the Conductor. True social bonds may be shallow due to curated interactions. "
            "Flora is cultivated and controlled: 'Serenity Grass' (calming pheromones), 'Nutri-Trees' (food pellets), genetically engineered flowers. No wild plants exist. "
            "Fauna is bio-engineered: 'Cleaner Bots' (insect-like), 'Companion Pets' (emotional support), or in controlled 'Bio-Domes.' Wild animals are forbidden. "
            "Potential conflicts: Systemic failure or corruption of The Conductor. Emergence of genuine dissent or a rebellion seeking true freedom. Depletion of resources needed for the high-tech infrastructure. External contact with a society that values individuality, causing existential crises. The psychological toll of suppressed emotions leading to widespread apathy or breakdown."
        )
    },
    "symbiotic_weald": {
        "topic": "The Symbiotic Weald of Mycoria",
        "description": (
            "The Symbiotic Weald of Mycoria is a civilization living in and alongside a colossal, sentient fungal network (the 'Great Network') permeating their ancient, damp, temperate rainforest home. Giant mushroom-like structures, shaped with the Network, serve as dwellings. "
            "Governed by 'Myco-Prophets' who communicate directly with the fungal consciousness via ritualistic spore ingestion, they value ecological harmony, bio-engineering of living tools/structures from fungal material, and the Network's collective memories. "
            "Their history is a shared, living memory within the Network; they believe they 'awoke' with the forest. Cyclical events like the 'Great Spore Bloom' renew the Network. "
            "Technology is entirely biological and symbiotic, based on manipulating and cultivating the Great Network. This includes living architecture, bioluminescent lighting, tools grown from hardened fungi, and communication through mycelial pathways. Advanced understanding of fungal biology, genetics, and ecological interplay. "
            "Military/Defense involves the Great Network itself rapidly growing dense, impassable barriers, releasing incapacitating or hallucinogenic spores, or guiding symbiotic creatures to defend. Myco-Prophets can direct these responses. Warriors might use hardened fungal armor and weapons that deliver spores. "
            "Culture is deeply animistic and communal, with a profound spiritual connection to the Great Network. Art involves shaping living fungal sculptures, bioluminescent patterns, and music created by wind passing through resonant fungal structures. Rituals focus on communion with the Network and maintaining ecological balance. "
            "Social structure is egalitarian and decentralized, with Myco-Prophets acting as spiritual guides rather than rulers. Decisions are made communally, influenced by the perceived will of the Great Network. Individuals see themselves as part of the larger forest organism. "
            "Flora is the fungal network in myriad forms: 'Elder Caps,' 'Pathfinder Moss,' 'Memory Fungi,' 'Nectar-Cups,' alongside ancient trees integrated with the mycelium. "
            "Fauna includes 'Myco-Boars,' 'Spore-Squirrels,' 'Glimmerwing Moths,' and the Mycorians themselves (humanoids with fungal traits). 'Shadow Lurkers' are natural predators within the ecosystem. "
            "Potential conflicts: Disease or blight affecting the Great Network. External forces attempting to exploit the forest's resources (logging, mining). A Myco-Prophet misinterpreting the Network's will, leading to harmful actions. The Network becoming aggressive or senile. Philosophical divisions on how to interact with the outside world, if discovered."
        )
    },
    "tidal_consortium": {
        "topic": "The Tidal Consortium of the Coral Kelp Republics",
        "description": (
            "The Tidal Consortium is a sprawling network of interconnected submersible 'Aqua-Domes' and floating trade platforms among vast, genetically engineered coral reefs and towering kelp forests in a warm, shallow ocean. They harness strong currents and tides for energy. "
            "Governed by elected 'Trade Barons' and 'Kelp Masters' from prominent merchant families and aquaculture guilds, their history began after the 'Surface Collapse' (rising sea levels), with the 'Green Tide Revolution' (coral/kelp bio-engineering breakthroughs) enabling stable habitats. The 'Kelp Wars' (trade disputes) solidified power structures. "
            "They value innovation in marine technology, shrewd bartering, and cultivating exotic bioluminescent sea life. "
            "Technology includes advanced marine bio-engineering (coral, kelp, fauna), submersible habitat construction, tidal/current power generation, advanced aquaculture, and underwater resource extraction. They develop specialized submersibles for trade, defense, and exploration. Bioluminescence is widely used for lighting. "
            "Military/Defense relies on fleets of specialized, armed submersibles, trained marine creatures (e.g., 'Reef Guardian' sharks/cetaceans), and defensive installations integrated into coral structures (e.g., sonic emitters, net launchers). Tactics involve ambush and using the complex underwater terrain. "
            "Culture is mercantile and pragmatic, valuing ingenuity, wealth, and exploration. Art often incorporates polished shells, pearls, and bioluminescent organisms. Music mimics whale songs and ocean sounds. Grand 'Tidal Festivals' celebrate successful harvests or trade expeditions. "
            "Social structure is a plutocracy, with Trade Barons and Kelp Masters holding significant power. Guilds for various marine professions (engineers, cultivators, pilots) are influential. A significant working class populates the aquaculture farms and industrial zones. Social mobility is possible through successful enterprise. "
            "Flora includes 'Titan Coral,' 'Sun-Kelp' forests, 'Glow-Algae,' 'Pearl Pods,' and 'Filter-Fan Anemones.' "
            "Fauna includes farmed 'Armored Groopers,' 'Messenger Dolphins,' hunted 'Kraken-Squid,' defensive 'Reef Guardians,' and 'Coral Crabs.' "
            "Potential conflicts: Disputes between powerful Trade Barons/Guilds. Depletion of marine resources or ecological damage from industrial activity. Attacks from deep-sea leviathans or rival underwater civilizations. Piracy targeting trade routes. Social unrest from the working class over wealth disparity."
        )
    },
    "masquerade_polity": {
        "topic": "The Shifting Stages of the Masquerade Polity",
        "description": (
            "The Masquerade Polity is a society where social/political power is determined through elaborate, ritualized theatrical performances and storytelling competitions, located in a city of grand theaters and ornate opera houses on interconnected river islands, often shrouded in mist. "
            "Governed by a 'Chorus of Maestros' (acclaimed performers/playwrights), their history began with a rebellion against a rigid, emotionless regime, using art to inspire revolution (the 'Great Performance'). The 'Age of Masks' codified performance rituals. "
            "They value creativity, emotional expression, intricate mask-making, and narrative's power to shape reality. Disputes are settled via competitive performances. "
            "Technology focuses on stagecraft: advanced illusion projection, acoustic manipulation, animatronics, pyrotechnics, and elaborate costume/mask engineering. 'Stage-Wrights' are master technicians. Architecture itself is theatrical and adaptable. "
            "Military/Defense is primarily through elaborate illusions, psychological diversions, and misdirection projected by Stage-Wrights to confuse and deter enemies. Some performance troupes might be trained in acrobatics and stage combat, forming a ceremonial guard, but direct conflict is avoided. Espionage and propaganda through narrative are key. "
            "Culture is entirely centered around performance. Daily life involves rehearsals, scriptwriting, and attending shows. Masks are worn frequently, signifying roles or emotional states. Language is poetic and dramatic. Emotional intelligence and rhetorical skill are highly prized. "
            "Social structure is a meritocracy based on artistic talent and performance success. The Chorus of Maestros holds supreme authority. Various Troupes and Guilds (actors, writers, mask-makers, stage-wrights) compete for influence and resources. Aspiring artists strive for recognition on the Grand Stages. "
            "Flora includes 'Weeping Willows,' 'Muse-Blooms,' 'Echo-Vines,' and advanced topiary art. "
            "Fauna includes 'Songbirds,' 'Shadow Hounds,' 'Chameleon Minks,' and 'Pageant Wings' butterflies. "
            "Potential conflicts: Creative stagnation or repetition of narratives. Power struggles between dominant Troupes. The rise of a performer whose narratives challenge the Polity's foundations. An external threat immune to illusions. The blurring of reality and performance leading to societal psychosis."
        )
    },
    "silent_sentinels": {
        "topic": "The Silent Sentinels of Craghold",
        "description": (
            "The Silent Sentinels are a monastic, martial society in Craghold, an almost impenetrable mountain fortress carved into the highest peak of the 'Dragon's Teeth' range, dedicated to guarding a sealed ancient Vault of immense power/danger. The only approach is the 'Needle's Eye' path. "
            "Governed by a 'High Warden' and a council of veteran 'Stone Guardians,' their history began millennia ago when the 'First Warden' sealed the Vault. The order's sole purpose is eternal vigilance. "
            "They value discipline, vigilance, esoteric martial arts, and absolute secrecy. Culture is austere, with minimalist art and functional design. "
            "Technology is focused on maintaining fortifications, ancient traps (mechanical, acoustic), and survival in extreme alpine conditions. They possess knowledge of simple mechanics, stonemasonry, and possibly some lost forging techniques for their traditional weapons. Any advanced tech is likely related to the Vault's containment, understood only by the High Warden. "
            "Military/Defense *is* their society. All Sentinels are highly trained warriors. Defense relies on formidable fortifications, mastery of mountain/siege warfare, acoustic traps, and their esoteric martial arts. They operate in small, disciplined units. Secrecy and deterrence are primary strategies. "
            "Culture is defined by solemn duty, meditation, rigorous training, and the sacred oath to guard the Vault. Art is non-existent or purely functional (e.g., calligraphic rendering of their oaths). Knowledge is passed down through strict oral tradition and coded texts. Emotional expression is suppressed in favor of unwavering focus. "
            "Social structure is a rigid hierarchy: High Warden, Council of Stone Guardians, veteran Sentinels, acolytes. Advancement is through proven skill, discipline, and unwavering loyalty over decades of service. Absolute obedience is expected. "
            "Flora includes 'Stone Moss,' 'Ice Petals,' and 'Warden's Wort' (medicinal/focus herb). "
            "Fauna includes 'Crag Eagles' (natural sentries), 'Snow Leopards,' 'Stone Bighorns,' and 'Whisper Wrens' (signal birds). "
            "Potential conflicts: The Vault's seal weakening. Internal dissent or a Sentinel succumbing to curiosity/ambition regarding the Vault. External forces (nations, cults) discovering Craghold and attempting to breach the Vault. The psychological toll of eternal vigilance leading to madness or despair. The last Sentinel dying without a successor."
        )
    },
    "chrysalis_commune": {
        "topic": "The Chrysalis Commune of the K'lik",
        "description": (
            "The Chrysalis Commune of the K'lik are insectoid beings in massive, organically grown hive-structures (hardened resin, chitin, woven fibers) in warm, humid jungles or subterranean networks, sharing a collective consciousness modulated by specialized 'Nexus-Queens.' "
            "Individual identity is fluid; roles are determined by biological caste (Workers, Warriors, Nurturers, Nexus-Queens) and the Commune's needs, communicated via pheromonal signals. Their history is a continuous 'Flow,' driven by 'The Great Weave' (hive expansion/strengthening). "
            "They value efficiency, communal labor, and hive expansion. "
            "Technology is entirely bio-organic: hive construction, pheromonal communication networks, bio-luminescence, cultivation of symbiotic fungi/plants, and biological weapons (acid sprays, potent toxins). Genetic manipulation of castes for specific functions is likely. "
            "Military/Defense relies on swarming tactics by warrior castes (armored, powerful mandibles/sprays), the hive's defensible structure, and potent chemical deterrents. The Nexus-Queen coordinates defense through the collective consciousness. "
            "Culture is purely communal; the individual is subsumed by the hive mind. 'Art' might be intricate hive patterns or complex pheromonal 'songs.' Rituals revolve around the Nexus-Queen, molting cycles, and communal tasks. No concept of personal property or achievement. "
            "Social structure is a true hive society with biologically determined castes, all subservient to the Nexus-Queen and the collective will. Communication is instantaneous and instinctual via pheromones and shared consciousness. "
            "Flora includes cultivated 'Pollen Orbs,' 'Resin Trees,' and 'Pheromone Blooms.' "
            "Fauna is the K'lik castes, and possibly symbiotic 'Tunneling Beetles' or 'Glow-Worms.' "
            "Potential conflicts: Rival hives. A catastrophic disease targeting a specific caste or the Queen. Environmental changes threatening their food sources or hive stability. Contact with species that disrupt their pheromonal communication or collective consciousness. A mutation leading to individuality within the hive."
        )
    },
    "keepers_sands": {
        "topic": "The Keepers of the Shifting Sands",
        "description": (
            "The Keepers are an enigmatic desert culture in a hyper-arid desert of shimmering, rolling dunes, centered around 'Temporal Loci'—ancient sites where time flows erratically. Hidden, shifting oases are their settlements. "
            "Governed by 'Chronometrists' who study temporal anomalies and perform rituals to stabilize/influence local time streams, they believe they descend from a civilization that caused a temporal cataclysm ('Great Unraveling'). Their imperative is to prevent further damage. "
            "They value patience, intricate clockwork mechanisms, and preserving 'Time-Worn Relics.' Society is reclusive, fearing paradoxes. "
            "Technology involves intricate clockwork devices for measuring and subtly influencing temporal fields, specialized tools for handling Time-Worn Relics, and unique architectural methods to withstand temporal stresses. They may have limited understanding of the exotic physics behind the anomalies. "
            "Military/Defense involves creating localized temporal distortions (loops, stasis fields, accelerated/decelerated time zones) to confuse, trap, or deter intruders. They avoid direct conflict, relying on misdirection and the inherent dangers of the Shifting Sands. Some Keepers might possess limited personal temporal manipulation abilities. "
            "Culture is contemplative, patient, and deeply reverent of time's mysteries. Art involves intricate sand mandalas that shift with time, complex clockwork sculptures, and chants that resonate with temporal frequencies. Knowledge is passed through apprenticeships and cryptic texts. "
            "Social structure: Chronometrists form a council of elders. Society is divided into those who study, those who maintain oases, and those who guard the Loci. Entry into Chronometrist ranks requires decades of study and proven ability to perceive temporal nuances. "
            "Flora includes crystalline 'Dune Roses' (shifting blooms), 'Mirage Grass,' 'Hourglass Cacti,' and 'Echo Seeds' (briefly grow ancient plants). "
            "Fauna includes 'Sand Skippers' (phasing lizards), 'Temporal Moths' (variable lifespans), 'Dune Scorpions' (exist in multiple temporal states), and mythical 'Paradox Hounds.' Keepers may age slowly or have premonitions. "
            "Potential conflicts: A major Temporal Locus becoming dangerously unstable. Outsiders seeking to exploit temporal anomalies for power. A Keeper succumbing to the temptation of large-scale temporal manipulation. The Shifting Sands expanding. The philosophical burden of their knowledge leading to nihilism or factionalism."
        )
    }
}

# You can then access each society by its key, for example:
# print(f"Topic: {fictional_societies_dict['sky_shepherds']['topic']}")
# print(f"Description: {fictional_societies_dict['sky_shepherds']['description']}\n")

# To get all the keys:
# society_keys = list(fictional_societies_dict.keys())
# print(f"Available society keys: {society_keys}")
society_key = 'tidal_consortium'
society = fictional_societies_dict[society_key]

corpus_builder = BFSWorldCorpusBuilder(
    topic_name=society['topic'],
    topic_description=society['description'],
    max_depth=3
)

print("\nStarting corpus build process...")
generated_corpus_data: GeneratedCorpus = corpus_builder.build_corpus()
print("\nCorpus build process finished.")

# %%
from pprint import pprint
from pathlib import Path
import json
path = helpers.get_next_run_directory(Path(f"output/{society_key}"))

# Wrote full output out as json
with open(str(path / f"full_output.json"), "w", encoding='utf-8') as f:
    full_output_json =  {
        key: article_model.model_dump(mode='json')
        for key, article_model in generated_corpus_data.items()
    }

    # Write the data to the JSON file
    json.dump(full_output_json, f, indent=4, ensure_ascii=False)
    
for topic, value in generated_corpus_data.items():
    pprint(topic)
    pprint(value.article)
    
    with open(str(path / f"{topic}.md"), "w") as file:
        file.write(value.article)

