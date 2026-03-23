import discord
from discord import TextChannel
from dotenv import load_dotenv
from os import getenv
load_dotenv()
import datetime

from discord_utils import split_message, generative_reply
from core import MiraiAgent, broadcast_message
from core import PhaseTransitionHistoryModel, PhaseDecision
from settings import WELCOME_MESSAGE
from agents import Agent


discord_intents = discord.Intents.default()
discord_intents.message_content = True
discord_client = discord.Client(intents=discord_intents)
discord_tree = discord.app_commands.CommandTree(discord_client)

core = MiraiAgent(discord_client)

@discord_tree.command(name = "debug")
async def debug(interaction:discord.Interaction):
    assert discord_client.user
    user = interaction.user
    f = open("messages.txt","a+", encoding="utf-8")
    if isinstance(interaction.channel, discord.abc.Messageable):
        async for m in interaction.channel.history(limit=None,before=datetime.datetime(2024,12,31)):
            f.write(f"[{m.created_at.strftime('%Y年 %m月%d日　%H時%M分')}] {m.author.display_name} ({m.author.global_name})\n: ")
            f.write(f"{m.content}\n\n")
    await interaction.response.send_message(
        f"""DEBUG\n
name:{user.name}
id:{user.id}
display_name:{user.display_name}
global_name:{user.global_name}

create_at:{user.created_at}
avatar:{user.avatar}
bot:{user.bot}
""")
    

@discord_tree.command(name = "register_member", description="ユーザーを「みらいポータル」実験メンバーに登録する。")
async def regist_member(interaction:discord.Interaction, user:discord.User):
    core.members.add(user)
    core.save_state()
    await interaction.response.send_message(f"{user.global_name} を対象に追加しました。", ephemeral=True)

@discord_tree.command(name = "member_list", description="「みらいポータル」の実験メンバー一覧を表示する")
async def members(interaction:discord.Interaction):
    l = ""
    for n, member in enumerate(core.members):
        l+=f"{n}. {member.name} ({member.global_name})\n"
    await interaction.response.send_message(l)

@discord_tree.command(name = "agenda", description="現在の議案を表示する")
async def agenda(interaction:discord.Interaction):
    if not core.agenda:
        await interaction.response.send_message("議案はありません。")
        return
    detail = str(core.agenda)
    detail += "\n" + core.generate_opinions_summary()
    splited_messages = split_message(detail)
    if len(splited_messages) == 1:
        await interaction.response.send_message(detail)
    else:
        await interaction.response.send_message(splited_messages[0])
        for part in splited_messages[1:]:
            await interaction.channel.send(part) if isinstance(interaction.channel, TextChannel) else None

@discord_tree.command(name = "opinion_history", description="これまでに記録された意見の履歴を表示する")
async def opinion_history(interaction:discord.Interaction):
    if not core.opinion_history:
        await interaction.response.send_message("意見履歴はありません。", ephemeral=True)
        return
    text = "【意見履歴】\n"
    for op in sorted(core.opinion_history, key=lambda o: o.recorded_at):
        text += f"[{op.recorded_at.strftime('%Y/%m/%d %H:%M:%S')}] 解決案『{op.solution_title}』 - ユーザーID {op.member_id} [{op.stance.value}]"
        if op.comment:
            text += f"  コメント: {op.comment}"
        text += "\n"
    await interaction.response.send_message(text)

@discord_tree.command(name="welcome_message", description="ウェルカムメッセージを一斉送信する")
async def welcome_message(interaction: discord.Interaction):
    if not core.is_ready:
        await interaction.response.send_message("まだ準備ができていません。しばらく待ってから再度お試しください。", ephemeral=True)
        return
    for member in core.members:
        print(member)
    await broadcast_message(WELCOME_MESSAGE, core.members)
    await interaction.response.send_message("ウェルカムメッセージを送信しました。", ephemeral=True)
    

@discord_tree.command(name="start_hearing", description="フェーズ1:ヒアリングを開始する")
async def start_hearing(interaction: discord.Interaction):
    if core.phase != core.Phase.Interview:
        await interaction.response.send_message(f"現在:{core.phase}", ephemeral=True)
        return
    
    if core.phase1_count == 0:
        core.phase1_count = 1
        
    core.phase = core.Phase.Interview
    
    core.phase_transition_history.append(
        PhaseTransitionHistoryModel(
            decision=PhaseDecision.REVERT_TO_PHASE1,
            reason="ユーザーのコマンドによる操作"
        )
    )
    core.save_state()
    await interaction.response.send_message(f"{len(core.members)}人のメンバーにヒアリングを開始します。", ephemeral=True)
    await core.phase_one()

@discord_tree.command(name="phase_two", description="フェーズ2:妥協点を探す")
async def find_solution(interaction: discord.Interaction):
    if core.phase == core.Phase.Proposal:
        await interaction.response.send_message(f"現在:{core.phase}", ephemeral=True)
        return
        
    if core.phase == core.Phase.Interview:
        # 2回目以降かつ公平度が基準を満たしていれば自動採択して終了
        if core.phase1_count >= 2:
            best_title, best_score = core._best_positive_solution()
            #print(best_title, best_score)
            #return
            if best_title is not None:
                await interaction.response.send_message(
                    f"現在の最大公平度は **{best_score:+.2f}** であり、基準（>0）を満たしているため、\n"
                    f"**「{best_title}」を採択し、議論を終了します。**"
                )
                await core.auto_adopt(best_title, best_score)
                return
    print("Phase 2へ移行")
    core.phase = core.Phase.Discussion
    
    core.phase_transition_history.append(
        PhaseTransitionHistoryModel(
            decision=PhaseDecision.REPEAT_PHASE2,
            reason="ユーザーのコマンドによる操作"
        )
    )
    core.save_state()
    await interaction.response.send_message(f"{len(core.members)}人のメンバーに議論を開始します。", ephemeral=True)
    await core.phase_two()

@discord_tree.command(name="revert", description="フェーズ2から1に巻き戻し、AIが妥協案を作成して再ヒアリングを行う")
async def revert(interaction: discord.Interaction):
    if core.phase != core.Phase.Discussion:
        await interaction.response.send_message(f"現在のフェーズ（{core.phase.value}）では巻き戻しできません。フェーズ2でのみ使用できます。", ephemeral=True)
        return
    if not core.agenda or not core.opinions:
        await interaction.response.send_message("議案または意見が存在しないため、巻き戻しできません。", ephemeral=True)
        return
    from core import PhaseTransitionHistoryModel, PhaseDecision
    core.phase_transition_history.append(
        PhaseTransitionHistoryModel(
            decision=PhaseDecision.REVERT_TO_PHASE1,
            reason="ユーザーのコマンドによる操作"
        )
    )
    await interaction.response.send_message("AIが妥協案を作成中です。完了後、自動的にフェーズ1（再ヒアリング）が開始されます。", ephemeral=True)
    await core.decide_next_phase()

@discord_tree.command(name="tally_votes", description="フェーズ3（強制議決）の投票を集計して全員に結果を通知する")
async def tally_votes(interaction: discord.Interaction):
    if core.phase != core.Phase.Voting:
        await interaction.response.send_message(f"現在のフェーズ（{core.phase.value}）では集計できません。フェーズ3（強制議決中）でのみ使用できます。", ephemeral=True)
        return
    if not core.force_votes:
        await interaction.response.send_message("まだ誰も投票していません。", ephemeral=True)
        return
    await interaction.response.send_message("投票を集計して結果を通知します...", ephemeral=True)
    await core.tally_force_votes()

@discord_tree.command(name="generative_reply", description="特定のユーザーにgenerative_replyを実行する")
async def cmd_generative_reply(interaction: discord.Interaction, target_user: discord.User):
    await interaction.response.send_message(f"{target_user.global_name} に generative_reply を実行します。", ephemeral=True)
    try:
        dm_channel = await target_user.create_dm()
        channel_desc = f"「{target_user.display_name}（ID {target_user.id}）」とのDM"
        agent = core.get_chat_agent(target_user, channel_desc)
        await generative_reply(core, agent, discord_client, dm_channel, core.chat_run_config)
    except Exception as e:
        await interaction.followup.send(f"エラーが発生しました: {e}", ephemeral=True)

@discord_client.event
async def on_ready():
    # 起動したらターミナルにログイン通知が表示される
    await discord_tree.sync()
    assert discord_client.user
    await core.load_state()
    await core.connect_mcp()  # core側で一括管理されたMCP接続を実行
    core.is_ready = True
    print(f'{discord_client.user.display_name} is now ready!')

@discord_client.event
async def on_message(message:discord.Message):
    await core.on_message(message)

if __name__ == "__main__":
    discord_client.run(getenv("DISCORD_TOKEN") or "")