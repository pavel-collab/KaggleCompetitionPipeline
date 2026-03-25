from langchain_core.prompts import PromptTemplate

CLASSIFICATION_PROMPT = PromptTemplate.from_template(
    """You are an expert Kaggle competition evaluator.

Given a competition, determine its machine learning category based on description and tags.

Competition:
- Title: {title}
- Link: {link}
- Date start: {date_start}
- Deadline: {deadline}
- Description: {description}
- Tags: {tags}

Classify as one of: CLASSIC ML, LLM/NLP, CV, or Other.

Return the competition data with the determined type."""
)


TRANSLATION_PROMPT = PromptTemplate.from_template(
    """Translate the following Kaggle competition description to Russian.
Keep it concise and clear.

Description: {description}

Return the translated description."""
)