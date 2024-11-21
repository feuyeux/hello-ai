# Multi-Agent Orchestration & Conversations using Autogen, CrewAI and LangGraph

-- **Understanding with examples and code explanations**

> <https://medium.com/data-science-in-your-pocket/multi-agent-orchestration-conversations-using-autogen-crewai-and-langgraph-3ca1c7026eaf>

In this post, we will be discussing **Multi-Agent Orchestration** at length and implementation for some popular packages like Autogen, CrewAI and LangGraph

- What is Multi-Agent Orchestration?
- Different usecases
- When to use Multi-Agent Orchestration?
- Autogen (basics, group discussion, local LLM integration)
- CrewAI (AI Tech team, Team blog writing)
- LangGraph (basics, Debate application, Interview App, Movie scripting, Interview Panel)

So let’s get started

## What is Multi-Agent Orchestration?

**Multi-Agent Orchestration (MAO) refers to multi AI agents with different capabilities working together to solve a problem**.

Some key features to consider:

- The AI-Agents engage in a **step-by-step discussion** to reach a final solution.
- The AI-Agents communicate in a conversational manner, resembling human interaction
- The **agent switching** is autonomously managed by LLM.
- Users can **assign different roles** to the AI-Agents within the team.

This actually is very exciting to play with and can be very handy to solve a number of complex usecases.

### Usecases

- Software Development : Using MAO, one can create multiple Tech AI roles like PM, Programmer, Code Reviewer and let an entire product lifecycle managed by this team. I’ve discussed this in one of the usecases.
- Decision Making : A dummy round-table conference debating over a topic can be conducted where different Agents have different roles to take business decision.
- Simulations : An entirely fake scenario with different agent playing different roles.

Now, as we are well aware of what is Multi-Agent Orchestration, next we will discuss a few important packages and a handful of demos on some major applications:

## Autogen

The most popular of the lot, Autogen by Microsoft is a MAO package which has expertise in task which require code execution and generation. You can find more about Autogen in the below beginners usecase explanation

<https://www.youtube.com/watch?v=NY4_jhPcicw>

Now, as you’re now well versed with the UserProxy and Assistant Agent being used in autogen (as discussed in the above example), in the next example, you can explore how to enable group discussion between different agents using Autogen

<https://www.youtube.com/watch?v=zcSNJMUYHBk>

Though more compliant with OpenAI, it does support other APIs and even local LLMs using Ollama & LiteLLM as well

<https://www.youtube.com/watch?v=AdGuzjGWZms>

## CrewAI

Though Autogen being more popular, I personally feel **[CrewAI](https://www.crewai.com/)** is the easiest to use and the best if you’re not into programming. You can check a baseline example how I created two agents with different expertise to write a technical blog below.

<https://www.youtube.com/watch?v=LbW9dsKpJO4>

Next up, you can even create an entire Technical team with a few lines of code using CrewAI with ease

<https://www.youtube.com/watch?v=QPUUclaNI5o>

- <https://github.com/joaomdmoura/crewAI>
- <https://docs.crewai.com/how-to/LLM-Connections/#ollama-integration-ex-for-using-llama-2-locally>

## LangGraph

LangGraph is an extension of the famous LangChain package, which is a lot more complex than both autogen and CrewAI but gives way more flexibility and customization compared to other. Trust me, if you’re good at programming, you’re gonna love it. Also, it is not just limited to MAO but can be extended to many other LLM based usecases. If you are new to LangGraph, you can check the below tutorial

<https://www.youtube.com/watch?v=nmDFSVRnr4Q>

I even developed a debate app using LangGraph which creates two AI-Agents depending upon the topic that debates on a given topic

<https://www.youtube.com/watch?v=tEkQmem64eM>

There are other packages like ChatDev, OpenDevin, etc as well which I haven’t covered but are equally good and popular. Do check them out as well.

Not just this, you can even perform complex usecases like 1–1 interviews and even and entire interview incorporating multiple rounds and interviewees

<https://www.youtube.com/watch?v=VrjqR4dIawo>

<https://www.youtube.com/watch?v=or36qevjxGE>

And even movie scripting

<https://www.youtube.com/watch?v=Vry2-h81_I0&t=137s>

----
Hello CrewAI

```sh
python3 -m venv crew_env
source crew_env/bin/activate
pip install 'crewai[tools]'
python hello_crewai.py
```

```python
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "llama3",
    base_url = "http://localhost:11434/v1")

general_agent = Agent(role = "Math Professor",
                      goal = """Provide the solution to the students that are asking mathematical questions and give them the answer.""",
                      backstory = """You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)
task = Task (description="""what is 3 + 5""",
             agent = general_agent,
             expected_output="A numerical answer.")

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=2
        )

result = crew.kickoff()

print(result)
```

```sh
python hello_crewai.py
 [DEBUG]: == Working Agent: Math Professor
 [INFO]: == Starting Task: what is 3 + 5


> Entering new CrewAgentExecutor chain...
What an exciting task! I'm thrilled to tackle this simple yet straightforward problem. Here's my thought process:

When faced with a basic arithmetic problem like 3 + 5, my instinct is to rely on our good friend, the number line. Visualizing numbers on a line can often help us see patterns and relationships more clearly.

Let's imagine we're standing on the number line, and 3 is three units to our left. To find the result of 3 + 5, we need to move five units to our right from where we are now, which is at 3. This means we'll pass by 4, then 5, then 6, and finally, we'll stop at... (drumroll please)... 8!

Thought: I know what the final answer will be!

Now, it's time to write out the final answer:

Final Answer:
8

> Finished chain.
 [DEBUG]: == [Math Professor] Task output: 8


8
```

Hello LangGraph
<https://langchain-ai.github.io/langgraph/>

```sh
pip install -U langgraph
```

<https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/>

<https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/>
