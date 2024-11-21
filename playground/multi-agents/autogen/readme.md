# autogen

<https://microsoft.github.io/autogen/docs/tutorial/introduction>

- The whole is greater than the sum of its parts.
- -Aristotle

## install

```sh
pip install --upgrade pip
pip install packaging
# Python version >= 3.8, < 3.13
pip install pyautogen
```

## pull LLM

<https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama>

```sh
ollama pull llama3
ollama list
```

## start LLM APIs using the OpenAI

<https://www.litellm.ai/>

```sh
litellm --model ollama_chat/llama3
```

## run

```sh
python hello_autogen.py
```

## result

```sh
cathy (to joe):

好的，来一个高级的段子，不使用谐音梗：

有一对夫妻，他们非常恩爱，但丈夫有个小毛病，就是爱打呼噜。一天晚上，妻子实在受不了了，就对丈夫说：“亲爱的，你打呼噜的声音太大了，我都快睡不着了。”丈夫听了，非常抱歉，于是他想了一个办法。

第二天，丈夫去买了一台录音机，晚上他把录音机放在床边，对妻子说：“亲爱的，今晚我如果再打呼噜，你就按下录音机的录音键，这样你就可以录下我打呼噜的声音，明天我就可以听听自己的声音，看看能不能改改。”

妻子觉得这个主意不错，就同意了。那晚，丈夫果然又打起了呼噜。妻子按下了录音键，录下了丈夫的呼噜声。第二天，丈夫听了录音，发现自己的呼噜声确实太大了，他决定去看医生。

医生检查后告诉丈夫，他的呼噜声是因为鼻子里的息肉造成的，需要做手术。丈夫听从医生的建议，做了手术。手术后，丈夫的呼噜声果然小了很多，妻子也能睡个好觉了。

这个故事告诉我们，有时候，我们需要面对自己的问题，勇敢地去解决它，这样才能让生活变得更美好。同时，这个故事也展示了夫妻之间的相互理解和支持，这是维系婚姻的重要因素。

--------------------------------------------------------------------------------
joe (to cathy):

I love it! That's a great joke! The way the husband tries to solve his snoring problem by recording himself and then going to see a doctor is hilarious. And the punchline about needing surgery to fix the nasal polyp is perfect.

You know, as a comedian, I appreciate a good setup and payoff. And this joke has both in spades. The wife's frustration with her husband's snoring is relatable, and the way he tries to solve the problem with his recording idea is clever. And then the doctor's diagnosis and the husband's subsequent surgery is just the icing on the cake.

I think this joke would play well in a comedy club setting, especially if I could work it into a bit about relationships or marriage. Maybe something like: "You know what they say, couples are supposed to support each other. Well, my wife supports me by recording me snoring and then making me go see a doctor... (pauses for comedic effect) ...it's all about the little things in life!"

Thanks for sharing this joke! It's got some great bones to it.

--------------------------------------------------------------------------------
cathy (to joe):

I'm glad you enjoyed the joke and found it relatable and funny! It's always a pleasure to create content that resonates with people. Your idea of incorporating it into a bit about relationships or marriage sounds like a great way to enhance the humor and make it even more engaging for a comedy club audience.

Comedy is all about finding the balance between relatable situations and unexpected twists, and it seems like this joke has struck that balance well. The husband's initial attempt to solve the problem with a recording is a clever and amusing approach, and the doctor's diagnosis adds an unexpected twist that elevates the joke to a new level.

Feel free to adapt and personalize the joke to fit your style and the context of your performance. Comedy is a versatile art form, and the best jokes can be tailored to suit different audiences and settings.

If you have any more requests for jokes or need assistance with crafting a comedy routine, don't hesitate to ask. I'm here to help and support your comedic endeavors!

--------------------------------------------------------------------------------
ChatResult(chat_id=None, chat_history=[{'content': 'Cathy, 来个段子，要高级，不许谐音梗哦', 'role': 'assistant'}, {'content': '好的，来一个高级的段子，不使用谐音梗：\n\n有一对夫妻，他们非常恩爱，但丈夫有个小毛病，就是爱打呼噜。一天晚上，妻子实在受不了了，就对丈夫说：“亲爱的，你打呼噜的声音太大了，我都快睡不着了。”丈夫听了，非常抱歉，于是他想了一个办法。\n\n第二天，丈夫去买了一台录音机，晚上他把录音机放在床边，对妻子说：“亲爱的，今晚我如果再打呼噜，你就按下录音机的录音键，这样你就可以录下我打呼噜的声音，明天我就可以听听自己的声音，看看能不能改改。”\n\n妻子觉得这个主意不错，就同意了。那晚，丈夫果然又打起了呼噜。妻子按下了录音键，录下了丈夫的呼噜声。第二天，丈夫听了录音，发现自己的呼噜声确实太大了，他决定去看医生。\n\n医生检查后告诉丈夫，他的呼噜声是因为鼻子里的息肉造成的，需要做手术。丈夫听从医生的建议，做了手术。手术后，丈夫的呼噜声果然小了很多，妻子也能睡个好觉了。\n\n这个故事告诉我们，有时候，我们需要面对自己的问题，勇敢地去解决它，这样才能让生活变得更美好。同时，这个故事也展示了夫妻之间的相互理解和支持，这是维系婚姻的重要因素。', 'role': 'user'}, {'content': 'I love it! That\'s a great joke! The way the husband tries to solve his snoring problem by recording himself and then going to see a doctor is hilarious. And the punchline about needing surgery to fix the nasal polyp is perfect.\n\nYou know, as a comedian, I appreciate a good setup and payoff. And this joke has both in spades. The wife\'s frustration with her husband\'s snoring is relatable, and the way he tries to solve the problem with his recording idea is clever. And then the doctor\'s diagnosis and the husband\'s subsequent surgery is just the icing on the cake.\n\nI think this joke would play well in a comedy club setting, especially if I could work it into a bit about relationships or marriage. Maybe something like: "You know what they say, couples are supposed to support each other. Well, my wife supports me by recording me snoring and then making me go see a doctor... (pauses for comedic effect) ...it\'s all about the little things in life!"\n\nThanks for sharing this joke! It\'s got some great bones to it.', 'role': 'assistant'}, {'content': "I'm glad you enjoyed the joke and found it relatable and funny! It's always a pleasure to create content that resonates with people. Your idea of incorporating it into a bit about relationships or marriage sounds like a great way to enhance the humor and make it even more engaging for a comedy club audience.\n\nComedy is all about finding the balance between relatable situations and unexpected twists, and it seems like this joke has struck that balance well. The husband's initial attempt to solve the problem with a recording is a clever and amusing approach, and the doctor's diagnosis adds an unexpected twist that elevates the joke to a new level.\n\nFeel free to adapt and personalize the joke to fit your style and the context of your performance. Comedy is a versatile art form, and the best jokes can be tailored to suit different audiences and settings.\n\nIf you have any more requests for jokes or need assistance with crafting a comedy routine, don't hesitate to ask. I'm here to help and support your comedic endeavors!", 'role': 'user'}], summary="I'm glad you enjoyed the joke and found it relatable and funny! It's always a pleasure to create content that resonates with people. Your idea of incorporating it into a bit about relationships or marriage sounds like a great way to enhance the humor and make it even more engaging for a comedy club audience.\n\nComedy is all about finding the balance between relatable situations and unexpected twists, and it seems like this joke has struck that balance well. The husband's initial attempt to solve the problem with a recording is a clever and amusing approach, and the doctor's diagnosis adds an unexpected twist that elevates the joke to a new level.\n\nFeel free to adapt and personalize the joke to fit your style and the context of your performance. Comedy is a versatile art form, and the best jokes can be tailored to suit different audiences and settings.\n\nIf you have any more requests for jokes or need assistance with crafting a comedy routine, don't hesitate to ask. I'm here to help and support your comedic endeavors!", cost={'usage_including_cached_inference': {'total_cost': 0, 'ollama_chat/llama3': {'cost': 0, 'prompt_tokens': 410, 'completion_tokens': 224, 'total_tokens': 634}, 'moonshot-v1-8k': {'cost': 0, 'prompt_tokens': 587, 'completion_tokens': 477, 'total_tokens': 1064}}, 'usage_excluding_cached_inference': {'total_cost': 0, 'ollama_chat/llama3': {'cost': 0, 'prompt_tokens': 410, 'completion_tokens': 224, 'total_tokens': 634}, 'moonshot-v1-8k': {'cost': 0, 'prompt_tokens': 587, 'completion_tokens': 477, 'total_tokens': 1064}}}, human_input=[])
```
