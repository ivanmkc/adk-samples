import pathlib
import dataclasses
from typing import Literal

QueryAmbiguity = Literal["straightforward", "ambiguous"]

@dataclasses.dataclass
class Claim:
    query: str = dataclasses.field(metadata={"description": "A query about the source text"})
    claim: str = dataclasses.field(metadata={"description": "An answer to the query"})
    context: str = dataclasses.field(metadata={"description": "The context from the source text that the answer is generated from."})
    source_contains_context: str = dataclasses.field(metadata={"description": "Whether or not the context actually exists in the source text or not."})
    is_supported: bool = dataclasses.field(metadata={"description": "Whether the answer is supported by the corpus or not"})
    is_supported_after_rewriting: bool = dataclasses.field(metadata={"description": "After rewriting, whether the answer is supported by the corpus or not. This should ideally match `is_supported`."})
    reasoning: str = dataclasses.field(metadata={"description": "The reasoning for `is_supported_after_rewriting`."})
    source_file_path: str = dataclasses.field(metadata={"description": "Source file for which the query/answer applies to."})
    ambiguity: QueryAmbiguity
