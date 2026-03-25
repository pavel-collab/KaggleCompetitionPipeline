"""
LLM processing using LangChain.
Uses ChatOpenAI with structured output for classification and translation.
"""

from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_config import get_logger
from prompts import CLASSIFICATION_PROMPT, TRANSLATION_PROMPT

logger = get_logger("llm")


# ============================================================
# Pydantic models for structured output
# ============================================================


#TODO: monore: move chemas to the separate folder
class CompetitionClassification(BaseModel):
    """Structured output for competition classification."""

    title: str = Field(description="Competition title")
    link: str = Field(description="Competition link")
    date_start: str = Field(description="Start date in YYYY-MM-DD format")
    deadline: str = Field(description="Deadline in YYYY-MM-DD format")
    description: str = Field(description="Competition description")
    type: Literal["CLASSIC ML", "LLM/NLP", "CV", "Other"] = Field(
        description="Competition type: CLASSIC ML, LLM/NLP, CV, or Other"
    )


class TranslatedDescription(BaseModel):
    """Structured output for translated description."""

    description_ru: str = Field(description="Description translated to Russian")


# ============================================================
# LLM setup
# ============================================================

"""
Using the separate function to call llm to be able change the llm configuration in one place
"""
def get_llm() -> ChatOpenAI:
    """Create ChatOpenAI instance with OpenRouter settings."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0,
    )


# ============================================================
# Processing functions
# ============================================================


def classify_competition(
    title: str,
    link: str,
    date_start: str,
    deadline: str,
    description: str,
    tags: str,
) -> CompetitionClassification:
    """Classify competition type using LLM."""
    logger.info(f"Classifying competition: {title}")
    logger.debug(f"Using model: {settings.openai_model}, base_url: {settings.openai_api_base}")

    try:
        llm: ChatOpenAI = get_llm()

        # Use with_structured_output for guaranteed schema
        chain = CLASSIFICATION_PROMPT | llm.with_structured_output(
            CompetitionClassification
        )

        result = chain.invoke(
            {
                "title": title,
                "link": link,
                "date_start": date_start,
                "deadline": deadline,
                "description": description,
                "tags": tags,
            }
        )

        logger.info(f"Classification result: {title} -> type={result.type}")
        return result
    except Exception as e:
        logger.error(f"LLM classification failed for {title}: {type(e).__name__}: {e}")
        raise


def translate_description(description: str) -> str:
    """Translate description to Russian using LLM."""
    logger.info(f"Translating description (length={len(description)})")
    logger.debug(f"Description preview: {description[:100]}...")

    try:
        llm = get_llm()

        chain = TRANSLATION_PROMPT | llm.with_structured_output(TranslatedDescription)

        result = chain.invoke({"description": description})

        logger.info("Translation completed successfully")
        return result.description_ru
    except Exception as e:
        logger.error(f"LLM translation failed: {type(e).__name__}: {e}")
        raise
