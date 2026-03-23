from typing import Optional, Literal
from pydantic import BaseModel, Field
import discord
from discord import Member, User, Client, Message, DMChannel
from discord_utils import generative_reply, broadcast_message
from agents import Agent, Runner, function_tool, OpenAIProvider, RunConfig, RunContextWrapper, Tool, ModelSettings
from openai import AsyncOpenAI
from openai.types import Reasoning
from agents.mcp import MCPUtil
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, MCPServerManager
import json
import os
from enum import Enum
import datetime
from settings import (
    PHASE0_INSTRUCTION, 
    PHASE1_INSTRUCTION, 
    PHASE2_INSTRUCTION, 
    PHASE3_INSTRUCTION, 
    COMMON_RULES, 
    PHASE_TRANSITION_INSTRUCTION,
    REVERT_INSTRUCTION,  
    QUESTION_INSTRUCTION, 
    SUMMARIZE_INSTRUCTION, 
    POSITIVE_VALUE
)
from dotenv import load_dotenv
load_dotenv()

class SolutionModel(BaseModel):
    title: str = Field(description="解決策の名称、端的に。")
    description: str = Field(description="解決策の中身、具体的に。")
    author: str = Field(description="発案したユーザー")

class OpinionStance(Enum):
    """5段階の意思表示。値はラベル文字列（AIに渡しやすい形式）。"""
    STRONGLY_AGAINST = "絶対なし"
    SOMEWHAT_AGAINST = "どちらかというとナシ"
    NEUTRAL          = "中立"
    SOMEWHAT_FOR     = "まあいいと思う"
    STRONGLY_FOR     = "めちゃくちゃいい"

    @property
    def positivity_value(self) -> float:
        """ポジティブ度合い値。強い反対は賛成の10倍の重み（提案書設計準拠）。"""
        values: dict[str, float] = {
            "絶対なし":             POSITIVE_VALUE[0],
            "どちらかというとナシ":  POSITIVE_VALUE[1],
            "中立":                 POSITIVE_VALUE[2],
            "まあいいと思う":        POSITIVE_VALUE[3],
            "めちゃくちゃいい":      POSITIVE_VALUE[4],
        }
        return values[self.value]


class OpinionModel(BaseModel):
    """1ユーザー × 1解決案 に対する意見。MiraiAgentが内部管理する（AIツールには非公開）。"""
    member_id: int
    solution_title: str
    stance: OpinionStance
    comment: str = ""
    recorded_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class ForceVoteStance(Enum):
    """3操作の強制議決スタンス。"""
    FOR     = "賛成"
    NEUTRAL = "中立"
    AGAINST = "反対"


class ForceVoteModel(BaseModel):
    """強制議決フェーズでの1ユーザー × 1解決案の投票。"""
    member_id: int
    solution_title: str
    stance: ForceVoteStance
    recorded_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class AgendaModel(BaseModel):
    """AIが組み立てる議案データ。opinionsは含まない（SDKのスキーマ解析を汚染しないため）。"""
    title: str = Field(description="議題テーマのタイトル。キャッチーで分かりやすく。")
    description: str = Field(description="議題に至る背景と、なぜ議案が必要なのか、そしてどのような議題かについて、わかりやすく網羅的な説明文。")
    solutions: list[SolutionModel] = Field(description="解決案。現状維持を含む最低2択。")

    def __str__(self) -> str:
        detail = f"## **【{self.title}】**\n"
        detail += f"{self.description}\n"
        detail += "### **解決案**\n"
        for s in self.solutions:
            detail += f"【{s.title}】（発案：{s.author}）\n{s.description}\n\n"
        return detail


class PhaseDecision(Enum):
    REVERT_TO_PHASE1 = "A"
    PROCEED_TO_PHASE3 = "B"
    REPEAT_PHASE2 = "C"
    FORCE_VOTE = "D"
    QUESTION_MEMBER = "E"


class PhaseTransitionModel(BaseModel):
    decision: PhaseDecision = Field(description="判断の内容")
    reason: str = Field(description="その判断に至った理由を具体的に")

# フェーズ遷移履歴用モデル（グローバルスコープ）
class PhaseTransitionHistoryModel(BaseModel):
    decision: PhaseDecision
    reason: str = Field(default="ユーザーのコマンドによる操作")
    updated_agenda: Optional[AgendaModel] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)


class MiraiAgent():
    
    class Phase(Enum):
        Proposal = "Phase 0"
        Interview = "Phase 1"
        Discussion = "Phase 2"
        Voting = "Phase 3"

    def __init__(self, discord: Client):
        self.tools:list[Tool] = [
            regist_agenda,
            update_agenda,
            regist_opinion,
        ]
        self.mcp_tools: list[Tool] = []  # MCPサーバーから動的に読み込まれるツール
        self.force_vote_tools = [
            regist_force_vote,
        ]
        self.agenda: Optional[AgendaModel] = None
        self.opinions: list[OpinionModel] = []   # 最新の各ユーザー×解決案ごとの意見
        self.opinion_history: list[OpinionModel] = []  # 過去に記録された全ての意見履歴（時系列）
        self.summarized_opinion_history: list[str] = []  # 各ラウンドごとの要約された意見履歴
        self.force_votes: list[ForceVoteModel] = []  # 強制議決の投票
        self.phase_transition_history: list[PhaseTransitionHistoryModel] = []  # フェーズ遷移履歴
        self.discord: Client = discord
        self.members: set[Member|User] = set()
        self.phase = self.Phase.Proposal
        self.phase1_count: int = 1  # Phase1を実行した回数
        self.is_ready:bool = False
        
        if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
            from agents import set_trace_processors
            from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
            set_trace_processors([OpenAIAgentsTracingProcessor()])

        # 会話用
        self.chat_run_config = RunConfig(
            model_provider = OpenAIProvider(
                openai_client = AsyncOpenAI(
                    api_key=os.environ.get("CHAT_AI_API_KEY"),
                    base_url=os.environ["CHAT_BASE_URL"]
                ),
                use_responses=False
            ),
            model=os.environ["CHAT_AI_MODEL"],
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high"),
                parallel_tool_calls=False
                ),
        )

        # 戦略的判断
        self.strategy_run_config = RunConfig(
            model_provider = OpenAIProvider(
                openai_client = AsyncOpenAI(
                    api_key=os.environ.get("STRATEGY_AI_API_KEY"),
                    base_url=os.environ["STRATEGY_BASE_URL"]
                ),
                use_responses=False  # GeminiではResponses APIを無効化する必要がある
            ),
            model=os.environ["STRATEGY_AI_MODEL"],
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high"),
                parallel_tool_calls=False
                ),
        )

        # サーバー定義
        if os.environ.get("MCP_URL"):
            self.mcp_servers = [
                MCPServerStreamableHttp(
                    MCPServerStreamableHttpParams(
                        url=os.environ["MCP_URL"],
                        timeout=300 # HTTPリクエストのタイムアウトを延長
                    ),
                    client_session_timeout_seconds=300 # MCPセッションのタイムアウトを延長
                )
            ]
        # ライフサイクルマネージャー（タスクアフィニティ問題を解決するため）
        self.mcp_manager = MCPServerManager(self.mcp_servers)

    async def connect_mcp(self) -> None:
        """MCPサーバーへ接続し、ツールをボットのツールリストに統合する。"""
        print(f"DEBUG: Connecting to MCP servers...")
        try:
            # マネージャー経由で全てのサーバーを接続
            active_servers = await self.mcp_manager.connect_all()
            if active_servers:
                print(f"✅ {len(active_servers)} 個の MCP サーバーに接続成功")
                # 各サーバーからツールを取得してボットのツールリストに追加
                for server in active_servers:
                    mcp_raw_tools = await server.list_tools()
                    for t in mcp_raw_tools:
                        # ChatCompletions API (Gemini等) でも動作するように、標準の FunctionTool に変換
                        agent_tool = MCPUtil.to_function_tool(t, server, convert_schemas_to_strict=False)
                        
                        # DeepSeek等のLLMがlistやdictの戻り値を受け付けない問題への対策 (強制文字列化)
                        original_invoke = getattr(agent_tool, "on_invoke_tool", None)
                        if original_invoke:
                            async def patched_invoke(*args, _orig=original_invoke, **kwargs):
                                res = await _orig(*args, **kwargs)
                                if isinstance(res, list):
                                    parts = []
                                    for item in res:
                                        if isinstance(item, dict) and "text" in item:
                                            parts.append(item["text"])
                                        else:
                                            parts.append(str(item))
                                    return "\n".join(parts)
                                elif isinstance(res, dict):
                                    return res.get("text", str(res))
                                return str(res)
                            
                            agent_tool.on_invoke_tool = patched_invoke

                        self.mcp_tools.append(agent_tool)
                    print(f"   - {server.name} から {len(mcp_raw_tools)} 個のツールを読み込みました")
            else:
                print("⚠️ 有効な MCP サーバーが見つかりませんでした")
        except Exception as e:
            print(f"❌ MCP 接続中にエラーが発生しました: {e}")

    # ── メンバー検索 ────────────────────────────────
    def _find_member_by_name(self, name: str) -> Optional[Member | User]:
        """グローバル名・表示名・ユーザー名のいずれかでメンバーを検索する。"""
        for m in self.members:
            if (
                (hasattr(m, "global_name") and m.global_name == name)
                or m.display_name == name
                or m.name == name
            ):
                return m
        return None

    # ── 公平度計算 ────────────────────────────────
    def fairness(self) -> Optional[float]:
        """議案全体の公平度。全解決案の平均ポジティブ度のうち最大値を返す。意見なしはNone。"""
        if self.agenda is None:
            return None
        values: list[float] = []
        for s in self.agenda.solutions:
            relevant = [
                o.stance.positivity_value
                for o in self.opinions
                if o.solution_title == s.title
            ]
            if relevant:
                values.append(sum(relevant) / len(relevant))

        return max(values) if values else None

    def generate_phase_transition_history_text(self) -> str:
        """フェーズ遷移履歴を時系列でテキスト化（ENUM→日本語ラベル変換）"""
        if not self.phase_transition_history:
            return "（フェーズ遷移履歴なし）"
        decision_labels = {
            "REVERT_TO_PHASE1": "フェーズ1へ移行",
            "PROCEED_TO_PHASE3": "フェーズ3へ移行",
            "REPEAT_PHASE2": "フェーズ2へ移行",
            "FORCE_VOTE": "強制議決（フェーズ3）",
            "QUESTION_MEMBER": "メンバーに質問したうえで続行",
        }
        lines = ["【フェーズ遷移履歴】"]
        for h in sorted(self.phase_transition_history, key=lambda x: x.timestamp):
            dt = h.timestamp.strftime('%Y/%m/%d %H:%M:%S')
            agenda_title = h.updated_agenda.title if h.updated_agenda else ""
            label = decision_labels.get(h.decision.name, h.decision.name)
            line = f"[{dt}] {label} 理由: {h.reason}"
            if agenda_title:
                line += f" / 更新議案: {agenda_title}"
            lines.append(line)
        return '\n'.join(lines)

    async def _archive_and_summarize_opinions(self) -> None:
        """現在の意見を要約して履歴に追加し、現在の意見リストをクリアする。"""
        if not self.opinions or not self.agenda:
            return

        summary_lines:list[str] = []
        summary_lines.append(f"議題：{self.agenda.title}")
        summary_lines.append(f"{self.agenda.description}\n\n")
        round_num = len(self.summarized_opinion_history) + 1
        summary_lines.append(f"### 第{round_num}ラウンドの議論要約")
        summary_lines.append(f"議案: {self.agenda.title}")
        
        for solution in self.agenda.solutions:
            relevant = [o for o in self.opinions if o.solution_title == solution.title]
            if not relevant:
                continue
            
            # 公平度を計算
            valid_stances = [o.stance.positivity_value for o in relevant]
            avg_fairness = sum(valid_stances) / len(valid_stances)
            
            summary_lines.append(f"- 解決案: {solution.title} (平均公平度: {avg_fairness:+.2f})")
            for o in relevant:
                comment_part = f" 「{o.comment}」" if o.comment else ""
                summary_lines.append(f"  - ユーザID {o.member_id}: [{o.stance.value}]{comment_part}")

        # 要約リストに追加
        agent = Agent("Opinion Summarizer", instructions=SUMMARIZE_INSTRUCTION)
        res = await Runner.run(agent, input="\n".join(summary_lines), run_config=self.strategy_run_config)
        self.summarized_opinion_history.append(res.final_output)
        
        # 生データを履歴に退避してクリア
        self.opinion_history += self.opinions
        self.opinions = []

    def generate_opinions_summary(self) -> str:
        """集まった意見を、プロンプトにコンテキストとして渡すためのテキストとしてフォーマットする。

        過去の議論は要約された形式で含め、現在の最新のスタンスは詳細に表示する。
        """
        if self.agenda is None and not self.summarized_opinion_history and not self.opinions:
            return "（まだ他のメンバーからの意見はありません）"

        summary = ""
        
        # 過去の要約履歴
        if self.summarized_opinion_history:
            summary += "【過去の議論の要約（アーカイブ）】\n"
            summary += "\n\n".join(self.summarized_opinion_history)
            summary += "\n\n"

        # 現在の最新意見
        if self.opinions and self.agenda:
            summary += "【現在の最新意見】\n"
            for solution in self.agenda.solutions:
                # 公平度を計算
                relevant = [
                    o.stance.positivity_value
                    for o in self.opinions
                    if o.solution_title == solution.title
                ]
                if not relevant:
                    continue
                fairness = sum(relevant) / len(relevant)
                related_opinions = [o for o in self.opinions if o.solution_title == solution.title]
                
                summary += f"\n■ 解決案: {solution.title}（公平度={fairness:+.2f}）\n"
                for op in related_opinions:
                    summary += f" - [{op.stance.value}] "    
                    summary += f"   「{op.comment}」\n"
        
        if not summary:
            return "（まだ他のメンバーからの意見はありません）"

        return summary

    def generate_user_opinion_status(self, member_id: int) -> str:
        """特定ユーザーの意見記録状況をテキストで返す"""
        if self.agenda is None:
            return "（議案未登録）"
        
        lines = []
        for solution in self.agenda.solutions:
            existing = [o for o in self.opinions 
                        if o.member_id == member_id and o.solution_title == solution.title]
            if existing:
                op = existing[0]
                lines.append(f"✅ {solution.title} → {op.stance.value}「{op.comment}」")
            else:
                lines.append(f"⬜ {solution.title} → 未回答")
        
        return "\n".join(lines)

    # ── 状態保存・復元 ─────────────────────────────
    def save_state(self, filepath: str = "state.json") -> None:
        """現在の議案、意見、履歴、フェーズ、メンバーID、フェーズ遷移履歴をJSONに保存する"""
        state = {
            "agenda": self.agenda.model_dump(mode="json") if self.agenda else None,
            "opinions": [o.model_dump(mode="json") for o in self.opinions],
            "opinion_history": [o.model_dump(mode="json") for o in self.opinion_history],
            "summarized_opinion_history": self.summarized_opinion_history,
            "force_votes": [v.model_dump(mode="json") for v in self.force_votes],
            "phase": self.phase.value,
            "members": [m.id for m in self.members],
            "phase1_count": self.phase1_count,
            "phase_transition_history": [h.model_dump(mode="json") for h in self.phase_transition_history],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    async def load_state(self, filepath: str = "state.json") -> None:
        """JSONから状態を復元する"""
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            if state.get("agenda"):
                self.agenda = AgendaModel.model_validate(state["agenda"])
            else:
                self.agenda = None
            # load opinions & history
            self.opinions = [OpinionModel.model_validate(o) for o in state.get("opinions", [])]
            self.opinion_history = [OpinionModel.model_validate(o) for o in state.get("opinion_history", [])]
            self.summarized_opinion_history = state.get("summarized_opinion_history", [])
            self.force_votes = [ForceVoteModel.model_validate(v) for v in state.get("force_votes", [])]
            self.phase_transition_history = [PhaseTransitionHistoryModel.model_validate(h) for h in state.get("phase_transition_history", [])]
            
            phase_val = state.get("phase")
            if phase_val:
                self.phase = self.Phase(phase_val)

            self.phase1_count = state.get("phase1_count", 0)
            self.members = set()
            for member_id in state.get("members", []):
                try:
                    user = await self.discord.fetch_user(member_id)
                    if user:
                        self.members.add(user)
                except Exception as e:
                    print(f"Failed to fetch user {member_id}: {e}")
                    pass
            print(f"[{filepath}] 状態を復元しました（メンバー数: {len(self.members)}）")
        except Exception as e:
            print(f"状態の復元に失敗しました: {e}")

    # ── フェーズ処理 ──────────────────────────────
    def generate_member_vote_status(self, member_id: int) -> str:
        """強制議決フェーズにおける特定ユーザーの投票記録状況をテキストで返す"""
        if self.agenda is None:
            return "（議案未登録）"

        lines: list[str] = []
        for solution in self.agenda.solutions:
            existing = [
                v for v in self.force_votes
                if v.member_id == member_id and v.solution_title == solution.title
            ]
            if existing:
                lines.append(f"✅ {solution.title} → {existing[0].stance.value}")
            else:
                lines.append(f"⬜ {solution.title} → 未投票")
        return "\n".join(lines)

    async def phase_one(self):
        """フェーズ1: 全メンバーにDMで意見収集を開始する。
        
        2回目以降の実行時に、公平度 > 0 の Solution が存在する場合は
        最も公平度の高い解決案を自動決定してフェーズ3に移行する。
        """
        print(f"start phase1 (実行回数: {self.phase1_count})")

        # 2回目以降の実行時に、公平度 > 0 の Solution が存在する場合は
        # 最も公平度の高い解決案を自動決定して終了する。
        if self.phase1_count >= 2:
            best_title, best_score = self._best_positive_solution()
            if best_title is not None:
                await self.auto_adopt(best_title, best_score)
                return

        for member in self.members:
            print(member)
            dm_channel = await member.create_dm()

            common_rules = COMMON_RULES.format(
                now=datetime.datetime.now().strftime("%Y/%m/%d %H時%M分%S秒"),
                channel=f"「{member.display_name}（ID {member.id}）」とのDM",
                history_text=self.generate_phase_transition_history_text()
            )
            system_instruction = PHASE1_INSTRUCTION.format(
                common_rules=common_rules,
                agenda=self.agenda or "（まだ議案は登録されていません）",
                user_opinion_status=self.generate_user_opinion_status(member.id)
            )
            
            agent = Agent(name="Phase1_Agent", instructions=system_instruction, tools=self.tools+self.mcp_tools)
            await generative_reply(self, agent, self.discord, dm_channel, self.chat_run_config)

    async def phase_two(self):
        """フェーズ2: 現在集まっている意見をもとに、全員に妥協点を探る議論を開始する"""
        if self.phase != self.Phase.Discussion:
            self.phase = self.Phase.Discussion
            
        opinions_summary = self.generate_opinions_summary()
        
        for member in self.members:
            dm_channel = await member.create_dm()

            common_rules = COMMON_RULES.format(
                now=datetime.datetime.now().strftime("%Y/%m/%d %H時%M分%S秒"),
                channel=f"「{member.display_name}（ID {member.id}）」とのDM",
                history_text=self.generate_phase_transition_history_text()
            )
            system_instruction = PHASE2_INSTRUCTION.format(
                common_rules=common_rules,
                agenda=self.agenda or "（まだ議案は登録されていません）",
                opinions_summary=opinions_summary,
            )
            # フェーズ2開始のきっかけとなるAIからの第一声を生成させるため、空のメッセージを投げる
            agent = Agent(name="Phase2_Agent", instructions=system_instruction, tools=list(self.tools + self.mcp_tools))
            await generative_reply(self, agent, self.discord, dm_channel, self.chat_run_config)

    def _best_positive_solution(self) -> tuple[Optional[str], float]:
        """公平度 > 0 の解決案のうち、最も公平度が高いものを返す。
        
        Returns:
            (解決案タイトル, スコア) または (None, 0.0)
        """
        if self.agenda is None:
            return (None, 0.0)
        
        best_title: Optional[str] = None
        best_score: float = 0.0
        for s in self.agenda.solutions:
            relevant = [
                o.stance.positivity_value
                for o in self.opinions
                if o.solution_title == s.title
            ]
            if not relevant:
                continue
            score = sum(relevant) / len(relevant)
            if score > 0.0 and score > best_score:
                best_score = score
                best_title = s.title
        return (best_title, best_score)

    def _reset_state(self) -> None:
        """議論終了時の共通リセット処理。フェーズ・議案・意見・投票・カウントをすべてクリアする。"""
        self.phase = self.Phase.Proposal
        self.agenda = None
        self.opinions = []
        self.opinion_history = []
        self.summarized_opinion_history = []
        self.force_votes = []
        self.phase1_count = 1
        self.phase_transition_history = []
        self.save_state()

    async def auto_adopt(self, best_title: str, best_score: float) -> None:
        """解決案を自動確定し、全メンバーに通知して議案を終了する。"""
        if self.agenda is None:
            return

        print(f"[AutoAdopt] '{best_title}'（スコア {best_score:+.2f}）を採択し終了します。")
        
        notice = (
            f"📢 **解決案採択のお知らせ**\n"
            f"議論の結果、**「{best_title}」**（公平度スコア: {best_score:+.2f}）が\n"
            f"最も公平度の高い解決案として正式に採択されました。\n"
            f"ご協力ありがとうございました。"
        )

        for member in self.members:
            try:
                dm_channel = await member.create_dm()
                await dm_channel.send(notice)
            except Exception as e:
                print(f"[AutoAdopt] {member.id} への通知に失敗: {e}")

        self._reset_state()

    async def phase_three_force_vote(self) -> None:
        """フェーズ3（強制議決）: 各メンバーにDMで賛成・中立・反対AI投票を行わせる。"""
        if self.agenda is None:
            print("[ForceVote] 議案が存在しないため強制議決を実行できません")
            return

        self.phase = self.Phase.Voting
        self.force_votes = []  # 投票リセット
        self.save_state()
        print("[ForceVote] 強制議決（フェーズ3）を開始します— 全メンバーにDM投票を依頼")

        for member in self.members:
            dm_channel = await member.create_dm()

            common_rules = COMMON_RULES.format(
                now=datetime.datetime.now().strftime("%Y/%m/%d %H時%M分%S秒"),
                channel=f"「{member.display_name}（ID {member.id}）」とのDM",
                history_text=self.generate_phase_transition_history_text()
            )
            system_instruction = PHASE3_INSTRUCTION.format(
                common_rules=common_rules,
                agenda=self.agenda,
                vote_status=self.generate_member_vote_status(member.id),
            )
            agent = Agent(name="Phase3_Agent", instructions=system_instruction, tools=list(self.force_vote_tools))
            await generative_reply(self, agent, self.discord, dm_channel, self.chat_run_config)

    async def tally_force_votes(self) -> None:
        """強制議決の集計を行い、全メンバーに結果をDMで通知する。

        採択の優先順位: 賛成票数 → 中立票数（同数時のタイブレーク）
        
        もし賛成・中立票ともに同率の案が複数存在した場合は自動的に
        フェーズ2（議論）へ戻し、再度議論フェーズを継続する。
        """
        if self.agenda is None:
            print("[Tally] 議案が存在しない")
            return

        results: list[dict] = []
        for s in self.agenda.solutions:
            relevant = [v for v in self.force_votes if v.solution_title == s.title]
            for_count     = sum(1 for v in relevant if v.stance == ForceVoteStance.FOR)
            neutral_count = sum(1 for v in relevant if v.stance == ForceVoteStance.NEUTRAL)
            against_count = sum(1 for v in relevant if v.stance == ForceVoteStance.AGAINST)
            results.append({
                "title": s.title,
                "for": for_count,
                "neutral": neutral_count,
                "against": against_count,
            })
            print(f"  [{s.title}] 賛成:{for_count} 中立:{neutral_count} 反対:{against_count}")

        # determine winner with tie detection
        # 賛成 → 中立 で順位付けし、同率の場合はリストに残す
        if results:
            # find max for count
            max_for = max(r["for"] for r in results)
            candidates = [r for r in results if r["for"] == max_for]
            if len(candidates) > 1:
                max_neutral = max(r["neutral"] for r in candidates)
                candidates = [r for r in candidates if r["neutral"] == max_neutral]

            if len(candidates) > 1:
                # 完全に同率になったのでフェーズ2に戻す
                print("[Tally] 同率が発生したためフェーズ2（議論）に戻します")
                notice = (
                    "📢 強制議決の集計で賛成・中立ともに同率の解決案が複数存在したため、"
                    "フェーズ2に戻り再度議論を行います。"
                )
                await broadcast_message(notice, self.members)

                # フェーズだけ戻して状態は保持（議論は継続）
                self.phase = self.Phase.Discussion
                self.save_state()
                return

            winner = candidates[0]
            print(f"[Tally] 採択: 「{winner['title']}」")
        else:
            print("[Tally] 結果がありません。")
            return

        summary_lines = ["📊 **強制議決 結果**\n"]
        for r in results:
            marker = "✅ **採択**" if r["title"] == winner["title"] else "　"
            summary_lines.append(
                f"{marker} 「{r['title']}」 — 賛成:{r['for']} / 中立:{r['neutral']} / 反対:{r['against']}"
            )
        summary_lines.append(f"\n🏆 **「{winner['title']}」** が採択されました。")
        summary_text = "\n".join(summary_lines)

        for member in self.members:
            try:
                dm_channel = await member.create_dm()
                await dm_channel.send(summary_text)
            except Exception as e:
                print(f"[Tally] {member.id} への通知に失敗: {e}")
        
        self._reset_state()

    async def decide_next_phase(self) -> None:
        """フェーズ2から、進退をエージェンティックに判断する。"""
        
        if self.agenda is None or not self.opinions:
            return

        prompt: str = """
# 現在の議案
{agenda}

# メンバーから集まった意見
{opinions_summary}
""".format(
            agenda=str(self.agenda),
            opinions_summary=self.generate_opinions_summary()
        )

        try:
            revert_agent = Agent(
                name="RevertToPhase1Agent",
                instructions=REVERT_INSTRUCTION,
                output_type=PhaseTransitionModel,
                tools=[
                    update_agenda,
                ],
            )

            main_agent = Agent(
                name="PhaseTransitionDecision",
                instructions=PHASE_TRANSITION_INSTRUCTION,
                output_type=PhaseTransitionModel,
                handoffs=[
                    revert_agent
                ],
                tools=self.mcp_tools + [question_member]
            )
            res = await Runner.run(main_agent, input=prompt, run_config=self.strategy_run_config)

            if not res.final_output:
                print("revert_phase: AIから空の応答が返されました")
                return

            transition: PhaseTransitionModel = res.final_output
            
            print(f"Decision: {transition.decision.value}, Reason: {transition.reason}")
            
            if transition.decision == PhaseDecision.REVERT_TO_PHASE1:
                # Handoffされたので、ここでは処理しない
                if self.phase == self.Phase.Interview:
                    pass
                else:
                    print("Aが選ばれましたが、ハンドオフされずに何も実行されませんでした。")
            
            elif transition.decision == PhaseDecision.PROCEED_TO_PHASE3:
                # フェーズ3へ進む（十分な理解が得られた場合）
                self.phase_transition_history.append(
                    PhaseTransitionHistoryModel(
                        decision=transition.decision,
                        reason=transition.reason
                    )
                )
                # 公平度 > 0 の案があれば自動採択して終了。なければ強制議決（現状維持的な救済）
                best_title, best_score = self._best_positive_solution()
                if best_title is not None:
                    print(f"[decide_next_phase] フェーズ3移行を選択: 公平案があるため自動採択を実行")
                    await self.auto_adopt(best_title, best_score)
                else:
                    print(f"[decide_next_phase] フェーズ3移行を選択: 公平案がないため強制議決を開始")
                    notice = f"📢 **議論を終了し、最終的な強制議決（フェーズ3）に移行します。**\n\n**【理由】**\n{transition.reason}"
                    await broadcast_message(notice, self.members)
                    await self.phase_three_force_vote()
            
            elif transition.decision == PhaseDecision.REPEAT_PHASE2:
                # フェーズ2を再度行う
                self.phase_transition_history.append(
                    PhaseTransitionHistoryModel(
                        decision=transition.decision,
                        reason=transition.reason
                    )
                )
                notice = f"📢 **現在集まっている意見をもとに、さらなる妥協点を探るため議論（フェーズ2）を継続します。**\n\n**【理由】**\n{transition.reason}"
                await broadcast_message(notice, self.members)
                await self.phase_two()
            
            elif transition.decision == PhaseDecision.FORCE_VOTE:
                # 強制議決（フェーズ3）を実行
                self.phase_transition_history.append(
                    PhaseTransitionHistoryModel(
                        decision=transition.decision,
                        reason=transition.reason
                    )
                )
                print(f"[decide_next_phase] 強制議決を選択。理由: {transition.reason}")
                notice = f"📢 **議論の結果、強制議決（フェーズ3）に移行します。**\n\n**【理由】**\n{transition.reason}"
                await broadcast_message(notice, self.members)
                await self.phase_three_force_vote()

            elif transition.decision == PhaseDecision.QUESTION_MEMBER:
                print("[decide_next_phase] メンバーに質問したうえで議論を続行。")
                # 質問後、フェーズ2（議論）へ自動移行
                self.phase = self.Phase.Discussion
                await self.phase_two()

        except Exception as e:
            print(f"revert_phase: エラーが発生しました {e}:{e.args}")
    
    async def on_message(self, message: Message):
        # メンションされてないなら終了
        if message.author == self.discord.user:
            return

        is_dm = message.channel.__class__ == DMChannel

        if not self.discord.user in message.mentions:
            if not is_dm:
                return

        # DMの場合、登録済みメンバー以外は弾く
        if is_dm:
            member_ids = {m.id for m in self.members}
            dm_channel = await message.author.create_dm()
            if not self.is_ready:
                await dm_channel.send("申し訳ありません、只今起動中です。")
                return
            elif message.author.id not in member_ids:
                await dm_channel.send("申し訳ありません、あなたはこの議論の参加者として登録されていないため、対応できません。")
                return

        channel_desc = f"「{message.guild.name}」サーバー / 「{message.channel.name}」チャンネル" if message.guild and isinstance(message.channel, discord.abc.GuildChannel) else f"「{message.author.display_name}（ID {message.author.id}）」とのDM"
        agent = self.get_chat_agent(message.author, channel_desc)
        await generative_reply(self, agent, self.discord, message, self.chat_run_config)

    def get_chat_agent(self, user: discord.User | discord.Member, channel_desc: str) -> Agent:
        """指定したユーザーおよびチャンネル状況に応じた適切なAgentを取得する"""
        if self.phase == self.Phase.Proposal:
            template = PHASE0_INSTRUCTION
        elif self.phase == self.Phase.Interview:
            template = PHASE1_INSTRUCTION
        elif self.phase == self.Phase.Discussion:
            template = PHASE2_INSTRUCTION
        elif self.phase == self.Phase.Voting:
            template = PHASE3_INSTRUCTION
        else:
            template = PHASE0_INSTRUCTION

        common_rules = COMMON_RULES.format(
            now=datetime.datetime.now().strftime("%Y/%m/%d %H時%M分%S秒"),
            channel=channel_desc,
            history_text=self.generate_phase_transition_history_text()
        )
        format_kwargs: dict = {
            "common_rules": common_rules,
            "agenda": self.agenda or "（まだ議案は登録されていません）",
        }
        if self.phase == self.Phase.Interview:
            format_kwargs["user_opinion_status"] = self.generate_user_opinion_status(user.id)
        if self.phase == self.Phase.Discussion:
            format_kwargs["opinions_summary"] = self.generate_opinions_summary()
        if self.phase == self.Phase.Voting:
            format_kwargs["vote_status"] = self.generate_member_vote_status(user.id)

        system_instruction = template.format(**format_kwargs)
        tools = self.force_vote_tools if self.phase == self.Phase.Voting else self.tools + self.mcp_tools
        return Agent(
            name="Chat_Agent",
            instructions=system_instruction,
            tools=list(tools)
        )


# ── AIツール ──────────────────────────────────
@function_tool
def regist_agenda(ctx:RunContextWrapper[MiraiAgent], agenda: AgendaModel) -> str:
    """新しい議題を登録する。"""
    ctx.context.agenda = agenda
    ctx.context.opinions = []  # 議案が変わったら意見もリセット
    ctx.context.opinion_history = []
    ctx.context.summarized_opinion_history = []
    ctx.context.phase1_count = 1  # カウントもリセット
    ctx.context.phase = ctx.context.Phase.Interview
    ctx.context.save_state()
    
    return f"「{agenda.title}」議案を登録しました。"

@function_tool
async def update_agenda(ctx:RunContextWrapper[MiraiAgent], agenda: AgendaModel, reason: str) -> str:
    """議題とその選択肢を上書き更新する。"""
    ctx.context.agenda = agenda
    ctx.context.phase_transition_history.append(
        PhaseTransitionHistoryModel(
            decision=PhaseDecision.REVERT_TO_PHASE1,
            reason=reason,
            updated_agenda=agenda
        )
    )
    await ctx.context._archive_and_summarize_opinions()
    ctx.context.phase = ctx.context.Phase.Interview
    ctx.context.phase1_count += 1
    ctx.context.save_state()
    
    # 通知送信とフェーズ1開始
    notice = f"📢 **議論の状況を踏まえ、フェーズ1（個別ヒアリング）に戻ります。**\n\n**【理由】**\n{reason}"    
    notice += f"\n\n**【議案の更新】**\n議案内容が修正されました。各解決案への意見を再度確認・調整します。"
    await broadcast_message(notice, ctx.context.members)
    await ctx.context.phase_one()
    
    return f"「{agenda.title}」議案を上書き更新しました。"

@function_tool
async def question_member(ctx:RunContextWrapper[MiraiAgent], member_global_name: str, question_objective: str):
    """質問エージェントが、特定のユーザーに質問する。フェーズ遷移の判断に必要な情報が不足している場合などに、AIがこのツールを呼び出して質問することが想定される。"""

    member = ctx.context._find_member_by_name(member_global_name)
    assert member is not None, f"ユーザー「{member_global_name}」が見つかりませんでした。"
    channel = await member.create_dm()

    # 質問内容をreasonから生成
    common_rules = COMMON_RULES.format(
        now=datetime.datetime.now().strftime("%Y/%m/%d %H時%M分%S秒"),
        channel=f"「{member.display_name}（ID {member.id}）」とのDM",
        history_text=ctx.context.generate_phase_transition_history_text()
    )

    agent = Agent(
        name="QuestionMember_Agent",
        instructions=QUESTION_INSTRUCTION.format(
            common_rules=common_rules,
            reason=question_objective
        )
    )

    await generative_reply(ctx.context, agent, ctx.context.discord, channel, ctx.context.strategy_run_config)

@function_tool
def regist_opinion(
    ctx:RunContextWrapper[MiraiAgent],
    member_global_name: str,
    solution_title: str,
    stance: str,
    comment: str = "",
) -> str:
    """ユーザーの意見を登録（上書き更新）する。同一ユーザー×解決案の意見は上書き更新。
    ヒアリング時にユーザーの意見が変わった、もしくは新しく増えた場合、逐一このツールで登録すること。
    **上書きされるので、付け加える際は、元にあった意見も含める！！**
    - member_global_name: 対象ユーザーのグローバル名（display_name / global_name / name のいずれか）
    - solution_title: 対象の解決案タイトル（AgendaModelのsolutionsに存在するものと完全一致）
    - stance: 「絶対なし」「どちらかというとナシ」「中立」「まあいいと思う」「めちゃくちゃいい」のいずれか
    - comment: スタンスの根拠・懸念（任意）
    """
    if ctx.context.agenda is None:
        return "議案が登録されていないため、意見を記録できません。"

    member = ctx.context._find_member_by_name(member_global_name)
    if member is None:
        member_names = [m.display_name for m in ctx.context.members]
        return f"「{member_global_name}」というメンバーは見つかりません。有効なメンバー: {member_names}"
    member_id = member.id

    titles = [s.title for s in ctx.context.agenda.solutions]
    if solution_title not in titles:
        return f"「{solution_title}」という解決案は存在しません。有効な解決案: {titles}"

    try:
        enum_stance = OpinionStance(stance)
    except ValueError:
        valid_stances = [e.value for e in OpinionStance]
        return f"エラー: '{stance}' は正しいスタンスではありません。{valid_stances} のいずれかを指定してください。"

    opinion = OpinionModel(
        member_id=member_id,
        solution_title=solution_title,
        stance=enum_stance,
        comment=comment,
    )

    # 同一ユーザー×同一解決案は上書き
    ctx.context.opinions = [
        o for o in ctx.context.opinions
        if not (o.member_id == member_id and o.solution_title == solution_title)
    ]
    ctx.context.opinions.append(opinion)

    fairness = ctx.context.fairness()
    fairness_str = f"{fairness:+.2f}" if fairness is not None else "集計中"
    
    ctx.context.save_state()

    return (
        f"「{solution_title}」への意見を記録しました。\n"
        f"スタンス: {enum_stance.value}（{enum_stance.positivity_value:+.1f}点）\n"
        f"議案の現在の公平度: {fairness_str}"
    )

@function_tool
def regist_force_vote(
    ctx:RunContextWrapper[MiraiAgent],
    member_global_name: str,
    solution_title: str,
    stance: str,
) -> str:
    """強制議決フェーズでユーザーの投票を記録する。同一ユーザー×解決案の投票は上書き。

    - member_global_name: 対象ユーザーのグローバル名（display_name / global_name / name のいずれか）
    - solution_title: 対象の解決案タイトル（AgendaModelのsolutionsに存在するものと完全一致）
    - stance: 「賛成」「中立」「反対」のいずれか
    """
    print(ctx.context.__class__)
    
    if ctx.context.agenda is None:
        return "議案が登録されていないため、投票を記録できません。"

    member = ctx.context._find_member_by_name(member_global_name)
    if member is None:
        member_names = [m.display_name for m in ctx.context.members]
        return f"「{member_global_name}」というメンバーは見つかりません。有効なメンバー: {member_names}"
    member_id = member.id

    titles = [s.title for s in ctx.context.agenda.solutions]
    if solution_title not in titles:
        return f"「{solution_title}」という解決案は存在しません。有効な解決案: {titles}"

    try:
        enum_stance = ForceVoteStance(stance)
    except ValueError:
        valid = [e.value for e in ForceVoteStance]
        return f"エラー: '{stance}' は正しいスタンスではありません。{valid} のいずれかを指定してください。"

    vote = ForceVoteModel(
        member_id=member_id,
        solution_title=solution_title,
        stance=enum_stance,
    )
    # 同一ユーザー×同一解決案は上書き
    ctx.context.force_votes = [
        v for v in ctx.context.force_votes
        if not (v.member_id == member_id and v.solution_title == solution_title)
    ]
    ctx.context.force_votes.append(vote)
    ctx.context.save_state()

    return f"「{solution_title}」への投票を記録しました。スタンス: {enum_stance.value}"
