from typing import Sequence
import dataclasses
import json
from vertexai import generative_models

import asyncio
from async_lru import alru_cache

GENERATION_AUTORATER_EVAL_PROMPT_WITH_IMAGES = """
You are an evaluation expert. You will be provided with a query, a generated response and a ground truth. Your task is to analyze the generated response for accuracy, completeness, and relevance compared to the ground truth.
You may also be given with images that provided by generated response or ground truth, in this case, you also need to evaluate if the generated response images are relevant to the ground truth results.

Please provide:
A float score from 0 to 5, where 0 is completely inaccurate and 5 is perfectly accurate.
A detailed explanation of why you gave that score, highlighting specific areas where the generated response was correct, incorrect, missing information, or irrelevant.
The output should be in JSON format.

EXAMPLE:

Query: What's the Capital of China
Generated Response: Victoria.
Ground Truth: Beijing

EVALUATION
{
"score" : 0.0,
"reason" : "The generated response is incorrect based on provided ground truth."
}
"""

RESPONSE_SCHEMA_SCORE = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}


@dataclasses.dataclass(frozen=True)
class GenerationEvaluation:
  """The generation evaluation result.

  Attributes:
    score: The score of the generation evaluation.
    reason: The reason of the generation evaluation.
  """
  score: float
  reason: str


def eval_generation(
    autorater_model: generative_models.GenerativeModel,
    question: Sequence[generative_models.Part],
    model_reply: Sequence[generative_models.Part],
    ground_truth: Sequence[generative_models.Part],
) -> GenerationEvaluation:
  """Evaluates the generated response for accuracy, completeness, and relevance compared to the ground truth.

  Args:
    autorater_model: The autorater model to use.
    question: The question to ask the model.
    model_reply: The model's reply to the question.
    ground_truth: The ground truth answer to the question.

  Returns:
    The generation evaluation result.
  """
  part_list = []
  part_list.append(
      generative_models.Part.from_text(
          GENERATION_AUTORATER_EVAL_PROMPT_WITH_IMAGES
      )
  )

  part_list.append(generative_models.Part.from_text("\nQuery: "))
  for part in question:
    part_list.append(part)

  part_list.append(generative_models.Part.from_text("\nGenerated Response: "))
  for part in model_reply:
    part_list.append(part)

  part_list.append(generative_models.Part.from_text("\nGround Truth: "))
  for part in ground_truth:
    part_list.append(part)

  response: generative_models.GenerationResponse = (
      autorater_model.generate_content(
          part_list,
          generation_config=generative_models.GenerationConfig(
              temperature=0,
              response_mime_type="application/json",
              response_schema=RESPONSE_SCHEMA_SCORE,
          ),
      )
  )
  score_eval_response_dict = json.loads(response.candidates[0].text)
  eval_result = GenerationEvaluation(
      score=score_eval_response_dict["score"],
      reason=score_eval_response_dict["reason"],
  )
  return eval_result

@alru_cache(maxsize=None)
async def eval_generation_async(
   model: generative_models.GenerativeModel, 
   question: str, 
   model_reply: str, 
   ground_truth: str, 
   semaphore: asyncio.Semaphore) -> GenerationEvaluation:
    async with semaphore:
        return await asyncio.to_thread(eval_generation,
            autorater_model=model,
            question=[generative_models.Part.from_text(question)],
            model_reply=[generative_models.Part.from_text(model_reply)],
            ground_truth=[generative_models.Part.from_text(ground_truth)]
        )