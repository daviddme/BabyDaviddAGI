from babydaviddagi import BabyDaviddAGI
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import dotenv_values

config = dotenv_values("./.env")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    await update.message.reply_text("Hi! Use /run <user_specs>")


async def run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starts Bot"""
    bda = bda = BabyDaviddAGI(
        openai_api_key = config["OPENAI_API_KEY"],
        pinecone_api_key =  config["PINECONE_API_KEY"],
        pinecone_environment = config["PINECONE_ENVIRONMENT"],
        pinecone_table_name =  config["PINECONE_TABLENAME"]
    )
    bda.start(" ".join(context.args))
    await update.message.reply_text(bda.idea)
    await update.message.reply_text(bda.code)
    await update.message.reply_text(bda.optimizer_result)
    await update.message.reply_text(bda.analyst_result)
    

async def babydaviddagi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start BabyDaviddAGI with detail information"""
    bda = bda = BabyDaviddAGI(
        openai_api_key = config["OPENAI_API_KEY"],
        pinecone_api_key =  config["PINECONE_API_KEY"],
        pinecone_environment = config["PINECONE_ENVIRONMENT"],
        pinecone_table_name = config["PINECONE_TABLENAME"]
    )
    bda.brainstorm_agent(user_specs=" ".join(context.args))
    await update.message.reply_text(bda.idea)

    bda.coder_agent()
    await update.message.reply_text(bda.code)

    bda.backtester_agent()
    await update.message.reply_text(bda.bactest_result)

    bda.optimizer_agent()
    await update.message.reply_text(bda.optimizer_result)

    await update.message.reply_text(bda.analyst_agent())


def main() -> None:
    """Run bot."""
    # Create the Application and pass it your bot's token.
    application = (
        Application.builder()
        .token(config["TELEGRAM_BOT_TOKEN"])
        .build()
    )

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler(["start", "help"], start))
    application.add_handler(CommandHandler("babydaviddagi", babydaviddagi))
    application.add_handler(CommandHandler("run", run))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
