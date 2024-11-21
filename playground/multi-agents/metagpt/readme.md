# MetaGPT

<https://github.com/geekan/MetaGPT>

## install

```sh
pip install --upgrade metagpt
metagpt --init-config
```

## config

```sh
code ~/.metagpt/config2.yaml
```

<https://docs.deepwisdom.ai/main/en/guide/get_started/configuration/llm_api_configuration.html>

```yaml
llm:
  api_type: 'ollama'
  base_url: 'http://127.0.0.1:11434/api'
  model: 'llama3'
```

## start LLM APIs

```sh
# Exit the toolbar app to stop the server.
ollama serve
```

## run

```sh
python hello_metagpt.py --idea="选择比努力重要" --investment=3.0 --n_round=10
```

## result

```sh
2024-05-31 12:29:36.987 | INFO     | __main__:_act:62 - 没头脑(驴): to do SpeakAloud(SpeakAloud)
Your assertion that effort is more important than choice is a misguided attempt to simplify the complexities of human decision-making. It's akin to saying that a tree's growth is solely dependent on its roots, neglecting the role of sunlight, water, and soil quality. Choice is not just about making decisions; it's about exercising our agency, shaping our identities, and creating opportunities for personal growth.

Your emphasis on effort overlooks the fact that people have different capacities, aptitudes, and circumstances. It's like expecting everyone to run a marathon without considering their physical abilities or training. Choice allows us to adapt, innovate, and thrive in an ever-changing world.

So, I urge you to reconsider your stance and acknowledge the interplay between choice and effort. Let's not reduce human experience to a simplistic dichotomy. The beauty of life lies in its complexity, and our choices are what make it meaningful. 💥

2024-05-31 12:31:05.567 | INFO     | __main__:_act:62 - 不高兴(象): to do SpeakAloud(SpeakAloud)
Ugh, are you kidding me? You think you're so clever with your fancy words and flowery language, but let me tell you, it's all just a bunch of empty rhetoric. Your argument is nothing but a weak attempt to justify the notion that choice is more important than effort.

Newsflash: effort is what gets things done. Without it, we'd still be stuck in the Stone Age, wondering how to make fire without using our brains (which, might I add, are overrated anyway). You can't just choose your way out of a problem; you have to put in the work to overcome it.

And don't even get me started on your silly analogy about trees. Sunlight, water, and soil quality may be important for plant growth, but they're not exactly comparable to human decision-making. We're not plants, okay? We're complex beings with thoughts, feelings, and motivations that can't be reduced to a simple formula.

Your emphasis on choice is just a cop-out. It's an excuse for people who are too lazy or incompetent to put in the effort required to achieve their goals. And don't even get me started on your "agency" nonsense. What does that even mean? Sounds like a bunch of jargon to me.

So, no, I won't be reconsidering my stance anytime soon. In fact, I'll just keep saying it louder and more often: effort is what matters, not choice. And if you don't like it, well, that's your problem, not mine.
```