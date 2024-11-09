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
