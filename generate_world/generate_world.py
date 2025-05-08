import json
from pathlib import Path
from pprint import pprint
import time

import dspy
import helpers
import mlflow
import world_generator
import world_templates

mlflow.dspy.autolog()
mlflow.set_experiment("generate_world")

model_name = "gemini/gemini-2.5-pro-preview-03-25"
# model_name = "gemini/gemini-2.0-flash"
lm = dspy.LM(
    model=model_name,
    max_tokens=65535,
    allowed_openai_params=["thinking"],
    thinking={"type": "enabled", "budget_tokens": 1024},
)
dspy.configure(lm=lm)

for template_dictionary in [
    world_templates.alien_societies
    # world_templates.us_companies,
    # world_templates.us_companies_nontech,
]:
    for template_id, template_info in template_dictionary.items():
        # society_key = "silent_sentinels"
        # society = templates[society_key]

        corpus_builder = world_generator.WorldCorpusBuilder(
            topic_name=template_info["topic"],
            topic_description=template_info["description"],
            max_depth=2,
        )

        print("\nStarting corpus build process...")
        start_time = time.time()  # Record time before starting the process
        generated_corpus_data: world_generator.GeneratedCorpus = (
            corpus_builder.build_corpus()
        )
        end_time = time.time()  # Record time after the process is finished
        time_spent = end_time - start_time
        print(f"\nCorpus build process finished: {time_spent:.2f} seconds")

        path = helpers.get_next_run_directory(Path(f"output/{template_id}"))

        # Wrote full output out as json
        with open(str(path / f"full_output.json"), "w", encoding="utf-8") as f:
            full_output_json = {
                key: article_model.model_dump(mode="json")
                for key, article_model in generated_corpus_data.items()
            }

            # Write the data to the JSON file
            json.dump(full_output_json, f, indent=4, ensure_ascii=False)

        for topic, value in generated_corpus_data.items():
            pprint(topic)
            # pprint(value.article)
            topic_filename = helpers.convert_to_linux_filename(original_filename=topic)

            with open(str(path / f"{topic_filename}.md"), "w") as file:
                file.write(value.article)
