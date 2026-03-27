import os
import asyncio
import discord
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError, APIStatusError

DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "arcee-ai/trinity-mini:free",
    "stepfun/step-3.5-flash:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]

current_model_index = 0

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and concise AI assistant living inside a Discord server. "
    "Keep your responses clear and to the point. Use Discord markdown formatting when helpful "
    "(bold, code blocks, bullet points). Do not use excessive filler phrases."
)

MAX_HISTORY = 10
MAX_RETRIES = len(FREE_MODELS)

ai_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

conversation_history: dict[int, list[dict]] = {}


def get_current_model() -> str:
    return FREE_MODELS[current_model_index]


def get_history(channel_id: int) -> list[dict]:
    return conversation_history.get(channel_id, [])


def add_to_history(channel_id: int, role: str, content: str) -> None:
    if channel_id not in conversation_history:
        conversation_history[channel_id] = []
    conversation_history[channel_id].append({"role": role, "content": content})
    if len(conversation_history[channel_id]) > MAX_HISTORY * 2:
        conversation_history[channel_id] = conversation_history[channel_id][-MAX_HISTORY * 2:]


async def ask_ai(channel_id: int, user_message: str) -> str:
    global current_model_index

    add_to_history(channel_id, "user", user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_history(channel_id)

    for attempt in range(MAX_RETRIES):
        model = get_current_model()
        try:
            response = await ai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
            )
            reply = response.choices[0].message.content or "(no response)"
            add_to_history(channel_id, "assistant", reply)
            return reply

        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status in (429, 503, 502) or "rate" in str(e).lower():
                next_index = (current_model_index + 1) % len(FREE_MODELS)
                print(f"⚠️  Model '{model}' rate-limited, switching to '{FREE_MODELS[next_index]}'")
                current_model_index = next_index
                await asyncio.sleep(1)
                continue
            return f"⚠️ AI error: `{e}`"

        except Exception as e:
            return f"⚠️ Error: `{e}`"

    return "⚠️ All models are currently rate-limited. Please try again in a moment."


@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"🤖 Models (in priority order): {', '.join(FREE_MODELS)}")
    print("📡 Listening for mentions and !ask commands...")
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.listening,
        name=".defnotmdream"
    ))


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    is_mentioned = bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)

    if not (is_mentioned or is_dm):
        return

    if message.content.startswith("!"):
        return

    content = message.content
    for mention in message.mentions:
        content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
    content = content.strip()

    if not content:
        await message.reply("Hey! Ask me anything 😊")
        return

    async with message.channel.typing():
        reply = await ask_ai(message.channel.id, content)

    chunks = [reply[i:i+1990] for i in range(0, len(reply), 1990)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)


@bot.command(name="ask")
async def ask_command(ctx: commands.Context, *, question: str):
    """Ask the AI a question: !ask <your question>"""
    async with ctx.typing():
        reply = await ask_ai(ctx.channel.id, question)

    chunks = [reply[i:i+1990] for i in range(0, len(reply), 1990)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await ctx.reply(chunk)
        else:
            await ctx.send(chunk)


@bot.command(name="clear")
async def clear_command(ctx: commands.Context):
    """Clear the conversation history for this channel."""
    conversation_history.pop(ctx.channel.id, None)
    await ctx.send("🧹 Conversation history cleared!")


@bot.command(name="model")
async def model_command(ctx: commands.Context):
    """Show the current AI model being used."""
    await ctx.send(f"🤖 Currently using: `{get_current_model()}`")


@bot.command(name="bothelp")
async def help_command(ctx: commands.Context):
    """Show bot usage instructions."""
    embed = discord.Embed(
        title="AI Bot Help",
        description="I'm an AI assistant powered by OpenRouter. Here's how to use me:",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="Mention me",
        value=f"@{bot.user.display_name} <your message>",
        inline=False,
    )
    embed.add_field(name="!ask", value="!ask <your question>", inline=False)
    embed.add_field(name="!clear", value="Clear conversation history in this channel", inline=False)
    embed.add_field(name="!model", value="Show the current active model", inline=False)
    embed.add_field(name="!bothelp", value="Show this help message", inline=False)
    await ctx.send(embed=embed)


async def main():
    async with bot:
        await bot.start(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
