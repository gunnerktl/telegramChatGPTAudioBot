#!/usr/bin/env python
# pylint: disable=unused-argument, import-error
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import logging
import pathlib

import telegram.ext
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext, ExtBot

from src.chat_bot import generate_text_response
from src.config import config
from src.speech2text import get_transcription
from src.text2speech import get_speech_audio_path


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def get_voice_filename_drive(bot: ExtBot, file_id: str) -> pathlib.Path:
    new_file = await bot.get_file(file_id)
    # download the voice note as a file
    file_name = pathlib.Path(config.audio_file_path) / f"{file_id}.ogg"
    await new_file.download_to_drive(file_name)

    return file_name


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(f"{update.message}")


async def echo_audio(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    file_name = await get_voice_filename_drive(context.bot, update.message.voice.file_id)
    request_text = get_transcription(file_name)
    predicted = generate_text_response(input_text=request_text)

    await update.message.reply_text(f"request:\n{request_text}\n response:\n{predicted}")

    response_audio = get_speech_audio_path(predicted)
    await update.message.reply_voice(response_audio)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(config.telegram_chat_token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(telegram.ext.MessageHandler(filters.VOICE & ~filters.TEXT & ~filters.COMMAND, echo_audio))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
