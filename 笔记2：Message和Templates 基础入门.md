## Message 概述

在LangChain中，`Message`是`Chat Model`中进行信息交流的基本单位，它用于表示聊天模型的输入与输出，以及与对话可能相关的任何额外上下文或元数据。

事实上，Chat Model的定义就是一种以一个`消息（Message）列表`作为输入，并将一段`消息`（如AIMessage）作为输出的接口。

## Message的类型

Message对象主要包括三个要素：

- **角色（Role）**：定义Message的类型，常见角色如系统、用户；
- **内容（Content）**：代表Message的实际内容，可以是字符串、图片、视频、文档等等；
- **元数据（Metadata）**：可选项，包括根据聊天模型提供商而异的各种额外元数据，如messsage IDs、token用量等等。

```python
【Role】Message(content="消息的内容",其他数据)
```

根据role的不同，Message分又为了4种主要类型，包括：

- **SystemMessage**：对应的角色是系统，用来设定模型的基本身份、行为模式，属于一种系统Prompt。
- **HumanMessage**：对应的角色是用户，代表用户输入的信息。
- **AIMessage**：对应的角色是助手（assistant），代表模型回复的消息。
- **ToolMessage**：对应的角色是工具，它包含的内容主要是工具的调用。

## 三种等价的Prompt格式

除了前述标准格式，LangChain也支持**直接以字符串的形式输入（text prompts）**，或者**用OpenAI格式输入（Dictionary format）**。这意味着下列三种消息输入方式是等价的：

```python
model.invoke("Hello")

model.invoke([{"role": "user", "content": "Hello"}])

model.invoke([HumanMessage("Hello")])
```

## Message的基本使用方法

使用Message的最简单方法，就是将不同role的message放到一个列表中，然后将这个列表传递给model进行invoke。参考代码如下：

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="将用户输入翻译为中文"),
    HumanMessage(content="hello world!"),
]

model.invoke(messages)
```

换言之，LangChain中，如果你向Chat Model传入一个包含SystemMessage和HumanMessage的列表，Chat Model就能够明白SystemMessage是在设定模型的基本行为，而HumanMessage是用户的具体输入。

## 关于Chat Prompt Templates

事实上，熟悉Prompt编写的朋友可能会意识到，把SystemMessage和HumanMessage的组合起来，其实就能构成一套比较完整的Prompt。

于是，在LangChain中，为了让这个作为Prompt的message列表的生成过程更加灵活、便捷，可以使用`ChatPromptTemplates`组件对Prompt的关键要素进行预设，形成Prompt模板。

这样，当我们向模版输入数据（这些数据可以是用户的原始输入，也可以是程序的查询结果等等），这些数据能够与Prompt模版相结合（比如为用户输入增加一个SystemMessage，或者将用户输入整理为预设的格式），从而自动构造出最终真正传递给模型的消息列表。
## Chat Prompt Templates的基本用法

### Step 1.设置Prompt 模板

这里我使用LangChain0.3版本官方文档中的参考代码为例，完整内容如下：

```python
from langchain_core.prompts import ChatPromptTemplate

system_template = "将用户输入翻译为 {language}"

prompt_template = ChatPromptTemplate(
    [
	    ("system", system_template), 
	    ("user", "{text}")
	]
)
```

我们直接看第三段代码，这是构造Prompt模版的核心：

```python
prompt_template = ChatPromptTemplate(
    [
	    ("system", system_template), 
	    ("user", "{text}")
	]
)
```

可以看到，构造模板的方法非常简单，就是向`ChatPromptTemplate`传递一个列表类型的参数，该列表中的每个元素是一个元组（即方括号`[]`内的两个圆括号`()`包裹的数据），而每个元组内包含的元素又依次为`角色（role）`和`对应的内容模板`。

`system`角色对应的模板内容，又引用了一个变量`system_template`，根据第二段代码，这个变量的值为字符串`"将用户输入翻译为 {language}"`，其中` {language}`是一个占位符，将在实际使用时被具体的语言名称替换（例如“英语”、“法语”）。同理，`user`角色对应的模板内容`{text}`也是一个占位符。
### Step 2.使用Prompt 模版

我们使用`ChatPromptTemplate`构造出的模板对象`prompt_template`，也可以使用`invoke`方法来生成最终用来传给大模型的提示词，代码如下：

```python
prompt = prompt_template.invoke({"language": "中文", "text": "hello world!"})
```

其中，向invoke方法传入的是一个字典结构的数据，这个字典中包含了两个键值对，其“键”分别对应着我们前面在设置模版时提前占位的{language}和{text}。

这时，我们如果打印`prompt`，就会发现它其实就是一个message对象的列表。

换言之，我们使用`prompt_template.invoke()`，可以将一个与我们提示词模板相对应的字典结构的数据，转化为标准的LangChain消息格式，以作为提示词传递给大模型。
