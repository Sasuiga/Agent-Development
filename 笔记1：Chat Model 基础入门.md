
在LangChain中，`Chat Model`是提供大模型调用能力的关键组件，它为我们使用各式各样的模型提供了统一的接口，让我们无须事先学习不同厂家的模型API调用方式，就能简单地通过几行代码去使用它们。

## Chat Model 基本使用方法

### Step 1.前置准备工作

#### （1）安装所需的模型集成依赖

首先，安装LangChain：**pip install langchain**。

然后，根据你的任务需求，确定具体想要使用的模型。

在LangChain中，使用特定模型的最基本方式，就是利用现成的第三方模型集成（third-party integrations）。你可以在LangChain官网上轻松找到你所需要的模型集成的安装方法。

以DeepSeek为例，只需要：**pip install -qU "langchain-deepseek"**

#### （2）获取并配置LLM API

安装完需要的模型集成依赖，我们就可以进行API key的获取与配置。

在你的项目文件根目录下创建一个txt文件，命名为`.env`，其内容就是你要使用的模型及其对应的密钥。

我们将在代码中使用`load_dotenv`来导入密钥。


### Step 2. 初始化Chat Model

在具体使用模型前，首先需要初始化模型，根据代号确定具体要使用的模型，并对模型温度、最大token数等参数进行确定。

#### （1）基本初始化

你可以根据你需要使用的模型，查阅官方文档，使用对应包中的Chat Model子类来进行初始化。

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
```

#### （2）使用init_chat_model函数初始化

对于某些供应商的模型，我们还可以利用`init_chat_model`函数来初始化Chat Model，具体代码如下：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek-chat", model_provider="deepseek")
```

可以看到，与基本初始化相比，使用`init_chat_model`函数除了传入**模型名称**（model）参数，还需要传入**模型供应商**（model_provider)参数（当然temperature之类的常规参数也可以传）。

另外，对于model参数，还允许直接传入结构为`{model_provider}:{model}`的键值对来实现初始化，这样，配合一个模型字典，可以让模型切换变得更加简洁。

#### （3）暂没有集成的模型供应商的使用

虽然LangChain目前已经集成了数量可观的模型服务集成，但有些我们比较常用的供应商仍然没有独立的包，比如openrouter、硅基流动。

这时，我们可以看看想要使用的供应商是否兼容OpenAI API，如果是，那么就可以在LangChain中利用`ChatOpenAI`来调用。以硅基流动为例：

首先，在终端中输入指令安装openai的包：**pip install -qU langchain-openai**

然后，在进行实例化的时候，填入硅基流动的`base_url`。这里有个细节，因为用的是openai的包，所以前面在系统变量里存入API Key的时候也要用`“OPENAI_API_KEY”`的表述。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="xxxxxxxx",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=""
    base_url="https://api.siliconflow.cn/",
)
```


### Step3. 使用Chat Model

初始化完成后，我们就可以使用Chat  Model的各种方法（“方法”是python语言的术语，你可以简单理解成“功能”）来使用大模型，其中比较常用的方法如下：

- **invoke**：与Chat Model交互的最基本方式，就是聊天本身，它接收一段消息列表作为输入，并返回一段消息列表作为输出。
- **stream**：允许Chat Model在生成输出时采用流式输出的方式。
- **batch**：允许用户将多个请求批量传入Chat Model以提高处理效率。
- **bind_tools**：允许用户将一个工具绑定给Chat Model，使模型能在执行上下文时使用这个工具。
- **with_structured_output**：针对支持结构化输出的模型，这个方法能够自动完成将模式绑定到模型并按给定模式解析输出的过程。

以最常用的invoke方法为例，括号内输入我们需要传给模型的信息，这个过程，就相当于我们在DeepSeek网页端的对话栏中，输入“Hello”。

```python
model.invoke("Hello")
```

上述代码将生成一个`AIMessage`对象，这个对象包含了一个`content`属性，它的值就是模型对我们回复的内容。