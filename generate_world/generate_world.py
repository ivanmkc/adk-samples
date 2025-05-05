import mlflow
import helpers
import dspy
import world_generator
from world_templates import templates

from pprint import pprint
from pathlib import Path
import json

mlflow.dspy.autolog()
mlflow.set_experiment("generate_world")

model_name = "gemini/gemini-2.5-pro-preview-03-25"
# model_name = "gemini/gemini-2.0-flash"
lm = dspy.LM(
    model=model_name,
    max_tokens=65535,
    allowed_openai_params=['thinking'],
    thinking={"type": "enabled", "budget_tokens": 1024},
)
dspy.configure(lm=lm)

society_key = 'tidal_consortium'
society = templates[society_key]

corpus_builder = world_generator.WorldCorpusBuilder(
    topic_name=society['topic'],
    topic_description=society['description'],
    max_depth=3
)

print("\nStarting corpus build process...")
generated_corpus_data: world_generator.GeneratedCorpus = corpus_builder.build_corpus()
print("\nCorpus build process finished.")

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

