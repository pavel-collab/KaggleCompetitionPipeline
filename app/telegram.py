"""
Telegram bot integration for sending competition notifications.
"""

import asyncio

from telegram import Bot

from app.config import settings
from app.logging_config import get_logger

logger = get_logger("telegram")


async def send_message_async(text: str) -> None:
    """Send message to configured Telegram chat (async version)."""
    logger.info(f"Sending message to chat_id={settings.telegram_chat_id}")
    logger.debug(f"Message text (first 100 chars): {text[:100]}...")

    try:
        bot = Bot(token=settings.telegram_bot_token)
        result = await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=text,
            parse_mode="HTML",
        )
        logger.info(f"Message sent successfully, message_id={result.message_id}")
    except Exception as e:
        logger.error(f"Failed to send message: {type(e).__name__}: {e}")
        raise


def send_message(text: str) -> None:
    """Send message to configured Telegram chat (sync wrapper)."""
    logger.debug("Calling send_message_async via asyncio.run")
    asyncio.run(send_message_async(text))


def send_competition_notification(
    title: str,
    link: str,
    deadline: str,
    competition_type: str,
    description_ru: str,
) -> None:
    """Send formatted competition notification."""
    logger.info(f"Sending competition notification: {title}")

    message = f"""<b>Kaggle соревнование:</b>

<b>Title:</b> {title}
<b>Link:</b> {link}
<b>Deadline:</b> {deadline}
<b>Theme:</b> {competition_type}

<b>Описание:</b>
{description_ru}"""

    send_message(message)
    logger.info(f"Competition notification sent: {title}")


def send_error_notification(error_message: str) -> None:
    """Send error notification."""
    logger.warning(f"Sending error notification: {error_message}")
    send_message(f"Error: {error_message}")
