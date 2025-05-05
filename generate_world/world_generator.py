import collections
from typing import Dict, List, Tuple

import dspy
from pydantic import BaseModel, Field  # Import Pydantic components


# --- Pydantic Models for Data Structures ---
class SubtopicDetail(BaseModel):
    name: str = Field(description="The name of the subtopic.")
    description: str = Field(description="A brief description of the subtopic.")


class GeneratedArticleData(BaseModel):
    article: str = Field(
        description="The full Markdown content of the generated article."
    )
    facts: List[str] = Field(
        description="A list of discrete facts that formed the basis of the article. Should contain at least 5 facts if generated successfully."
    )
    subtopics: List[SubtopicDetail] = Field(
        description="A list of subtopics identified from this article's context for further exploration."
    )
    depth: int = Field(
        description="The depth of this topic in the generation graph (0 for initial topic)."
    )


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

    topic_name: str = dspy.InputField(
        desc="The name of the main topic to generate facts and subtopics for."
    )
    topic_description: str = dspy.InputField(
        desc="A detailed description of the main topic."
    )
    ancestral_facts: list[str] = dspy.InputField(desc="All relevant ancestral facts.")

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

    topic_name: str = dspy.InputField(
        desc="The name of the main topic to generate facts and subtopics for."
    )
    topic_description: str = dspy.InputField(
        desc="A detailed description of the main topic."
    )
    facts: List[str] = dspy.InputField(desc="A list of specific facts about the topic.")
    ancestral_facts: list[str] = dspy.InputField(desc="All relevant ancestral facts.")
    num_subtopics: int = dspy.InputField(desc="Number of subtopics to generate")

    subtopics: List[SubtopicDetail] = dspy.OutputField(desc="A list of subtopics")


class FactAndSubtopicGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.facts_generator: dspy.ChainOfThought = dspy.ChainOfThought(
            FactsGeneratorSignature
        )
        self.subtopics_generator: dspy.ChainOfThought = dspy.ChainOfThought(
            SubtopicsGeneratorSignature
        )

    def forward(
        self,
        topic_name: str,
        topic_description: str,
        ancestral_facts: List[str],
        num_subtopics: int,
    ) -> dspy.Prediction:
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
    facts: list[str] = dspy.InputField(
        desc="A list of facts specifically about this topic"
    )
    ancestral_facts: list[str] = dspy.InputField(
        desc="All relevant ancestral facts, formatted as a list"
    )

    article_content: str = dspy.OutputField(
        desc="A well-structured article in Markdown format, based on the provided facts. NO HYPERLINKS."
    )


class ArticleFromFactsGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate: dspy.ChainOfThought = dspy.ChainOfThought(
            ArticleFromFactsSignature
        )

    def forward(
        self, topic_name: str, facts: List[str], ancestral_facts: List[str]
    ) -> dspy.Prediction:
        return self.generate(
            topic_name=topic_name, facts=facts, ancestral_facts=ancestral_facts
        )


# --- Orchestrator: World Corpus Builder ---
class WorldCorpusBuilder:
    def __init__(
        self, topic_name: str, topic_description: str, max_depth: int = 2
    ) -> None:
        self.topic_name: str = topic_name
        self.topic_description: str = topic_description
        self.max_depth: int = max_depth

        self.fact_subtopic_generator: FactAndSubtopicGenerator = (
            FactAndSubtopicGenerator()
        )
        self.article_generator: ArticleFromFactsGenerator = ArticleFromFactsGenerator()

        self.generated_corpus: GeneratedCorpus = {}
        self.queue: collections.deque[
            Tuple[str, str, List[str], int]
        ] = collections.deque()
        self.visited_topics: set[str] = set()

    def build_corpus(self) -> GeneratedCorpus:
        log_indent_char: str = "  "
        print(
            f"{log_indent_char * 0}Starting corpus build for initial topic: '{self.topic_name}' with max_depth: {self.max_depth}"
        )
        self.queue.append((self.topic_name, self.topic_description, 0))

        ancestral_facts = []
        while self.queue:
            topic_name, topic_description, depth = self.queue.popleft()
            current_indent: str = log_indent_char * (depth + 1)

            if topic_name in self.visited_topics:
                print(f"{current_indent}Skipping already visited topic: '{topic_name}'")
                continue
            if depth > self.max_depth:
                print(
                    f"{current_indent}Skipping topic '{topic_name}' due to exceeding max_depth ({depth} > {self.max_depth})"
                )
                continue

            print(f"{current_indent}Processing Topic: '{topic_name}' (Depth: {depth})")
            self.visited_topics.add(topic_name)

            print(
                f"{current_indent}  Generating facts and subtopics for '{topic_name}'..."
            )
            fs_prediction = self.fact_subtopic_generator(
                topic_name=topic_name,
                topic_description=topic_description,
                ancestral_facts=ancestral_facts,
                num_subtopics=8
                if depth == 0
                else 4,  # Top-level should have a breadth of subtopics.
            )

            # If they are present but not the correct type (e.g. still a string), a TypeError might occur later.
            print(
                f"{current_indent}  Generated {len(fs_prediction.facts)} facts and {len(fs_prediction.subtopics)} subtopics for '{topic_name}'."
            )

            print(f"{current_indent}  Generating article for '{topic_name}'...")
            article_prediction: dspy.Prediction = self.article_generator(
                topic_name=topic_name,
                facts=fs_prediction.facts,
                ancestral_facts=ancestral_facts,
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
                    print(
                        f"{current_indent}  Queueing {len(fs_prediction.subtopics)} subtopics for next level (depth {depth + 1}):"
                    )
                    for sub_item_pydantic in fs_prediction.subtopics:
                        # Ensure sub_item_pydantic is indeed a SubtopicDetail instance if list is not empty
                        if not isinstance(sub_item_pydantic, SubtopicDetail):
                            print(
                                f"{current_indent}    - WARNING: Item in subtopics_pydantic_list is not a SubtopicDetail object: {sub_item_pydantic}. Skipping."
                            )
                            continue
                        sub_name = sub_item_pydantic.name
                        sub_desc = sub_item_pydantic.description
                        if sub_name and sub_name not in self.visited_topics:
                            print(
                                f"{current_indent}    - Adding '{sub_name}' (Desc: '{sub_desc[:30]}...') to queue."
                            )
                            self.queue.append((sub_name, sub_desc, depth + 1))
                        elif not sub_name:
                            print(
                                f"{current_indent}    - Skipping subtopic with missing name: {sub_item_pydantic.model_dump_json()}"
                            )
                        else:
                            print(
                                f"{current_indent}    - Subtopic '{sub_name}' already visited or queued at higher priority."
                            )
                else:
                    print(
                        f"{current_indent}  No subtopics generated or to queue for '{topic_name}'."
                    )

            print(f"{current_indent}Finished processing for '{topic_name}'.")

        print(
            f"{log_indent_char * 0}Corpus build process complete. Total topics processed: {len(self.visited_topics)}"
        )
        return self.generated_corpus
