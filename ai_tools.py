from __future__ import annotations

from agents import function_tool
from typing import TYPE_CHECKING

# 型チェック時のみ core からインポート（実行時には循環参照を避ける）
if TYPE_CHECKING:
    from core import MiraiAgent, AgendaModel, OpinionModel, OpinionStance, ForceVoteModel, ForceVoteStance