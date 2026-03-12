import discord
from discord.abc import Messageable
from settings import ALLOWED_FILETYPE, INPUT_TOKEN_LIMIT
from typing import Iterable 
from agents import Agent, Runner, RunConfig

def split_message(text: str, limit: int = 2000) -> list[str]:
    """
    メッセージを limit 文字以内のパーツに分割する。
    可能な限り改行で分割を試みる。
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        
        # 指定した制限内で最後の改行を探す
        split_pos = text.rfind('\n', 0, limit)
        
        # 改行が見つからない場合は、文字数制限で強制分割
        if split_pos == -1:
            split_pos = limit
            
        chunks.append(text[:split_pos].strip())
        text = text[split_pos:].strip()
        
    return [c for c in chunks if c]

async def message_to_context(m: discord.Message) -> tuple[dict, int]:
    # ユーザーのメッセージ
    text = f"[Message ID{m.id}] {m.created_at} {'【BOT】' if m.author.bot else ''} '{m.author.display_name}' （User ID {m.author.id}）のメッセージ{f' ([{m.reference.message_id}]への返信として)' if m.reference else ''}:\n"

    for embed in m.embeds:
        text+= f"\n<埋め込み>[{embed.title}]\n{embed.description}\n"
        for field in embed.fields:
            text+= f"## {field.name}\n{field.value}\n\n"
        text += f"{embed.footer.text}</埋め込み>\n"

    text+= f"「{m.content}」\n\n"
    text+= f"リアクション: {', '.join([f'{r.emoji}x{r.count}' for r in m.reactions]) or 'なし'}"
    
    # 添付ファイルURLを含める
    attachments_n = 0
    for a in m.attachments:
        if a.content_type in ALLOWED_FILETYPE:
            text += f"\n[添付ファイル: {a.url}]"
            attachments_n += 1

    msg_dict = {
        "role": "user" if not m.author.bot else "user",
        "content": text
    }
    # 簡単なトークン数見積もり
    token_est = len(text) // 2
    return msg_dict, token_est


async def generative_reply(
        core,
        agent: Agent,
        discord_client: discord.Client, 
        interaction: discord.Message | Messageable,
        run_config: RunConfig
        ):
    
    if isinstance(interaction, discord.Message):
        message = interaction
        channel = message.channel
        print(f"{interaction.author.global_name}へのgenerative_replyが呼び出されました。")
    elif isinstance(interaction, Messageable):
        message = None
        channel = interaction
        print(f"{channel}へのgenerative_replyが呼び出されました。")
    else:
        raise TypeError(f"{interaction.__class__} is not supported.")

    context_history = []
    if isinstance(channel, discord.TextChannel):
        await discord_client.fetch_channel(channel.id)
        
    async with channel.typing():
        messages = channel.history()
        messages_n = 0
        total_tokens = 0
        added_context_history = []
        async for m in messages:
            print(f"{messages_n+1}件目を読み込み中")
            msg_dict, token_count = await message_to_context(m)
            added_context_history.append(msg_dict)
            total_tokens += token_count
            messages_n += 1
            
            if total_tokens >= 10000:
                break
            if messages_n > 5 and total_tokens >= INPUT_TOKEN_LIMIT:
                break

        if not added_context_history:
            added_context_history.append({"role": "user", "content": "there is no message."})
            
        added_context_history.reverse()

        try:
            # Runner.run を用いた生成ループの実行
            res = await Runner.run(agent, input=added_context_history, run_config=run_config, context=core)
            
            print("runnerが実行されました。")
            if not res.final_output:
                print("空文字列が返されました")
                return

            reply = str(res.final_output)
            print(res)
            # メッセージの分割
            replies = split_message(reply)
            
            for part in replies:
                if message:
                    await message.reply(part)
                else:
                    await channel.send(part)
        except Exception as e:
            await channel.send(f"[generative_reply] エラー ({agent.name}): {type(e).__name__}: {e}")
            await channel.send(f"{run_config.model}")
            raise e
            return
        
async def broadcast_message(message:str, users:Iterable[discord.User|discord.Member]):
    if message == "":
        return
    dms:set[discord.DMChannel] = set()
    for u in users:
        dms.add(await u.create_dm())
    parted_message = split_message(message)
    for dm in dms:
        try:
            for part in parted_message:
                await dm.send(part)
        except Exception as e:
            print("[broadcast_message] エラー ({agent.name}): {type(e).__name__}: {e}")
            continue