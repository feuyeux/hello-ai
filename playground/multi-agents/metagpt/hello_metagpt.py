"""
Filename: MetaGPT/examples/debate.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.1.3 of RFC 116, modify the data type of the `send_to`
        value of the `Message` object; modify the argument type of `get_by_actions`.
"""

import asyncio
import platform
from typing import Any
import fire
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team


class SpeakAloud(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## 背景
    假设你是{name}，你正在与{opponent_name}进行辩论。
    ## 辩论历史
    前几轮:
    {context}
    ## 轮到你发言
    现在轮到你了。你应该紧密回应对手的最新论点，陈述你的立场，捍卫你的论据，并攻击对手的论点，以{name}的修辞和观点，用200个字，打造一个强烈而富有情感的回应，你的辩论将是：
    """
    name: str = "SpeakAloud"
    # request reached max request: 3, please try again after 1 seconds

    async def run(self, context: str, name: str, opponent_name: str):
        prompt = self.PROMPT_TEMPLATE.format(
            context=context, name=name, opponent_name=opponent_name)
        # logger.info(prompt)
        rsp = await self._aask(prompt)
        return rsp


class Debator(Role):
    name: str = ""
    profile: str = ""
    opponent_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SpeakAloud])
        self._watch([UserRequirement, SpeakAloud])

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [
            msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(
            f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # An instance of SpeakAloud

        memories = self.get_memories()
        context = "\n".join(
            f"{msg.sent_from}: {msg.content}" for msg in memories)

        rsp = await todo.run(context=context, name=self.name, opponent_name=self.opponent_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.opponent_name,
        )
        self.rc.memory.add(msg)

        return msg


async def debate(idea: str, investment: float = 3.0, n_round: int = 5):
    Biden = Debator(name="没头脑", profile="驴", opponent_name="不高兴")
    Trump = Debator(name="不高兴", profile="象", opponent_name="没头脑")
    team = Team()
    team.hire([Biden, Trump])
    team.invest(investment)
    team.run_project(idea, send_to="没头脑")
    await team.run(n_round=n_round)


def main(idea: str, investment: float = 3.0, n_round: int = 10):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debate(idea, investment, n_round))


if __name__ == "__main__":
    # python hello_metagpt.py --idea="选择比努力重要" --investment=3.0 --n_round=5
    fire.Fire(main)
