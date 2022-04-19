import Constants as Keys
from telegram.ext import *
import TheraBotTelegramCode as Tb

updater = Updater(Keys.API_KEY, use_context=True)


def start_command(update, context):
    name = update.message.chat.first_name
    update.message.reply_text('Hi ' + name + '!')
    update.message.reply_text('I am TheraBot!')
    update.message.reply_text('Nice to meet you :)')
    update.message.reply_text('You can always type quit to end the conversation!')


def help_command(update, context):
    update.message.reply_text('You can always talk to me! ')


def handle_message(update, context):
    user_id = update.message.chat.id
    text = str(update.message.text).lower()
    response = Tb.responses(text, user_id)
    print(response)
    update.message.reply_text(response)


def error(update, context):
    print(f"Update{update} caused error {context.error}")


def main():

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))

    dp.add_handler(MessageHandler(Filters.text, handle_message))

    dp.add_error_handler(error)

    updater.start_polling()
    updater.idle()


main()