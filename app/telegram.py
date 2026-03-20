"""
Telegram bot integration for sending competition notifications.
"""

import asyncio

from telegram import Bot

from app.config import settings


async def send_message_async(text: str) -> None:
    """Send message to configured Telegram chat (async version)."""
    bot = Bot(token=settings.telegram_bot_token)
    await bot.send_message(
        chat_id=settings.telegram_chat_id,
        text=text,
        parse_mode="HTML",
    )


def send_message(text: str) -> None:
    """Send message to configured Telegram chat (sync wrapper)."""
    asyncio.run(send_message_async(text))


def send_competition_notification(
    title: str,
    link: str,
    deadline: str,
    competition_type: str,
    description_ru: str,
) -> None:
    """Send formatted competition notification."""
    message = f"""<b>Kaggle соревнование:</b>

<b>Title:</b> {title}
<b>Link:</b> {link}
<b>Deadline:</b> {deadline}
<b>Theme:</b> {competition_type}

<b>Описание:</b>
{description_ru}"""

    send_message(message)


def send_error_notification(error_message: str) -> None:
    """Send error notification."""
    send_message(f"Error: {error_message}")
