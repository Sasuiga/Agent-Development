大家好，在上一期文章中，我给大家展示了如何在我们的基本Graph中加入Conditional Edge（条件边），以实现对workflow的更灵活的控制。

今天，我们来升级一下，利用条件边来构建Loop（循环），并引入LLM来实现一个简单的Chat Bot！另外，为了实现一个简单的memory效果，我还会给大家简要介绍一下Reducer函数的玩法。

今天这篇是相当呕心沥血了，篇幅较长但绝对不水，希望各位老爷觉得有帮助的，一键三连多多支持~

话不多说，直接上干货！
## 初始化LLM

正如本系列文章开头几篇的铺垫内容所介绍的，我们可以直接使用LanggChain中的Chat Model组件来调用大模型。不熟悉的同学请一定查阅前期内容补课，这里我就不再重复细讲了。

下面带着大家简单过一遍如何初始化LLM。

### 1.配置API Key

与前期所讲的通过`getpass`来导入LLM的API Key不同，这里我用了一个新的工具，即`load_dotenv`来导入，这样我们就不用每次运行程序的时候都手动输入密钥了。

```python
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv`的用法非常简单。首先，在你的项目文件根目录下创建一个txt文件，命名为`.env`，其内容就是你要使用的模型及其对应的密钥。这里我打算使用DeepSeek，所以写法就是：`DEEPSEEK_API_KEY=密钥`。

![[Pasted image 20251103214727.png]]

然后在你的项目代码中使用`load_dotenv()`，这个`.env`文件里保存的密钥，就能被自动加载了。
### 2.初始化LLM

这里我使用`init_chat_model`来帮我们完成初始化，具体代码非常简单：

```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("deepseek-chat", model_provider="deepseek")
```

写完以上四行代码，我们的LLM就初始化完成，可以直接使用了。我在notebook里invoke它一下看看：

![[Pasted image 20251103215732.png]]

看到AIMessage了吧，这就是DeepSeek对我的回复。

## 构建Grpah

### 1.构建state

接下来，我们需要创建GraphState。本期我们先暂时放一放我们的法师state（主要我现在还没想好怎么融合Orz），专注一下对话类应用的State的典型写法。

这里我们刚好可以再展开一下关于**state**这个概念的学习，感受下什么是**state**的**schema**和**Reducer**。

#### （1）Schema：一串Message序列

所谓State的Schema，其实就是对传入组成Graph的Nodes、Edges的数据的样式的定义。

正如我在前期文章中给大家分享的，在LangChain中，Chat Model是以一个**消息（Message）列表**作为输入的接口。现在，我们要在Graph中引入Chat Model来调用LLM，那不难猜到我们应当在state中构造一个键，并将它的数据类型定义为一个**消息（Message）列表**。参考代码如下：

```python
from langchain_core.messages import AnyMessage
from typing import TypedDict,List

class ChatState(TypedDict):
    messages:List[AnyMessage]
```

上述代码构造了一个名为`ChatState`的state，它带有一个叫`messages`的键，这个键的值是一个由`AnyMessage`类型的对象组成的`List`，也就是我们会向LLM的invoke方法传入的对象。

这里解释一下为什么是`AnyMessage`。如前期文章所属，LangChain中的Message对象，根据**role**的不同，可以被分类为**HumanMessage**、**AIMessage**、**SystemMessage**等等，以便使LLM在接收后能够更好地理解使用者的意图。

而`AnyMessage`，就意味着List中的元素可以是任意`role`的Message。

这点很重要，为什么？假设你有一个桶，别人问你，苹果（HumanMessage）可不可以放？可以。梨子（AIMessage）可不可以放？可以。橙子（SystemMessage）可不可以放？可以，blablabla。那么，与其说这个桶是一个可以放苹果、梨子和橙子的桶，不如说它是一个可以放水果（AnyMessage）的桶来得更方便。

You feel me？

需要说明的是，理论上这里用`BaseMessage`这个基类也是可以实现同样的效果，但官方文档里面是推荐使用`AnyMessage`的，具体的原因大家感兴趣的可以去自行探索。

#### （2）Reducer：add_message

确定了state的schema，我们的工作只完成了一半。我们现在还需要指定state的Reducer函数，以确定state的更新方式。

##### 1）关于memory

我们现在想搭建的是一个带memory的Chat Bot。所谓memory，从效果上来看就是，如果之前你告诉过LLM你最喜欢猫，那当你几轮对话后再问它你最喜欢的动物是什么时，它会告诉你是猫。如果不带memory的话，你和LLM的每一次对话都是一个全新的开始，上一句刚说的东西，下次再说就忘记了。

从具体实现方式来看，最简单的方法就是将前序对话（老的消息列表）保存下来，再与本次输入（新的消息列表）拼接起来，作为一个完整的context（上下文）一起传给LLM，这样LLM就能够始终“记得”前序的对话。

##### 2）指定Reducer

而正如前期有提到过，如果不指定state的Reducer函数，那么state的更新模式就是overwrite（覆写），即新的state的值会完全替换原来的state的值。所以，为了通过拼接新旧消息列表实现memory，我们需要指定Reducer函数以避免旧消息列表被overwrite。

现在我们对前面的state代码进行升级：

```python
from langchain_core.messages import AnyMessage
from typing import TypedDict,List,Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
	messages:Annotated[List[AnyMessage],add_message]
```

可以看到，我们引入了两个新玩意儿：

- `Annotated`：就是我们实现为state指定Reducer的工具。`Annotated`后面的方括号中有两个参数，前一个参数是我们定义的schema（即一个由message组成的list），而后一个参数，就是我们要指定的Reducer函数。
- `add_message`：就是我们具体选择的Reducer函数。它的效果是将新消息与旧消息拼接到一起，同时，如果新消息的ID与旧消息相同，则直接更新已有ID的旧消息。

我再用官网文档中的例子来展示一下`add_message`的效果：

```python
msgs1 = [HumanMessage(content="Hello", id="1")]
msgs2 = [AIMessage(content="Hi there!", id="2")]
add_messages(msgs1, msgs2)

>>> [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]

msgs1 = [HumanMessage(content="Hello", id="1")]
msgs2 = [HumanMessage(content="Hello again", id="1")]
add_messages(msgs1, msgs2)

>>> [HumanMessage(content='Hello again', id='1')]
```
可以看到，第一个例子中，两个`message`的id是不同的，所以`add_message`的执行效果是把他们两个放到一个列表里，类似于一个`append`。

而第二个例子中，两个`message`的id相同，所以`add_message`执行的是更新消息内容，而不再是放到一起。这就是`add_message`比`append`更灵活的地方

OK，完成了schema和Reducer的构造，我们的state就算完成了。

### 2.构建Node的Function

#### （1）业务逻辑梳理

接下来，我们就要开始为Graph的构建做准备了。首先我们得想清楚我们的业务逻辑、工作流，以确定需要哪些节点，怎么连接他们，然后再去构造节点和边的函数。

我们先闭上眼想一下，如果你和别人聊天，这个过程大致是什么样的：

你：小姐，请问一下有没有卖《半岛铁盒》？
小姐：有啊，你从前面右转的第二排架子上就有了。
你：哦好的，谢谢。
小姐：不会。

以上对话揭示了Chat的工作流的核心要素，主要包括
- 1.**开始需要一个用户输入来开启话题。**
- 2.**整个流程以”用户输入，Agent反馈“的模式，循环往复，直到用户达到自己的目的。**
- 3.**结束需要用户来发起**，即“谢谢”暗示了对话结束，小姐姐说不说“不会”并不重要。

于是，我们可以考虑构建一个这样的Graph，如下图所示：

![[Pasted image 20251107095328.png]]

- 首先，`get_user_input`用来获取用户输入；
- 然后马上接条件判断，看用户输入内容是否要求终止对话（说谢谢了）：
	- 如是，流转到`END`节点；
	- 如果不是，将用户输入传给`chat_bot`节点，由LLM处理生成回复，工作流再返回用户输入节点，等待用户的下一步指示。

因此，我们需要构造两个节点，一个条件边，先来搞节点的函数。

#### （2）构建节点的函数

##### 1）用户输入获取节点

```python
def get_user_input(state:ChatState) -> ChatState:
    """这个节点获取用户输入并添加到消息历史中"""
    user_input = input("输入: ")
    return {"messages":[HumanMessage(content=user_input)]}
```

第一个节点的代码非常简单，主要逻辑就是：
- 1.将用户的输入保存到`user_input`变量；
- 2.然后将这个变量作为`content`构造一个`HumanMessage`对象，放到列表里（即用`[]`包起来）；之所以需要这么做，是因为我们在定义state的时候明确了，`messages`的值应该是一个**由消息对象组成的List**；
- 3.然后用`return`，把这个消息列表更新成state的`messages`键的值。

这里有一个新知识点可以提一下，就是`return`语句的写法。与之前的`return state`不同，这里的写法意味着，我们直接对`messages`键的值进行修改。

##### 2） 机器人节点

接下来写机器人节点的逻辑，其实就是引入LLM，也非常简单：

```python
def Chat_Bot(state:ChatState) -> ChatState:
    """这个节点将使用大语言模型对用户的输入进行反馈"""
    response = llm.invoke(state["messages"])
    print(f"\nAI:{response.content}")
    return {"messages":[AIMessage(content=response.content)]}
```

下面解释一下代码的主要部分：

```python
	response = llm.invoke(state["messages"])
	print(f"\nAI:{response.content}")
```

这里是简单的调用已初始化的llm的invoke方法，不熟悉的同学请看本系列第一期内容。这里我们将state的messages键的值作为对llm的输入。

第3行是将AI的回应打印出来，其中使用了`.content`方法以只显示response中的主要内容。什么意思？因为`llm.invoke`会返回一个`AIMessage`对象，这种Message对象的内部结构中除了content还有其他一些杂七杂八的内容，通过使用`.content`，我们可以只获取我们想要的消息内容本身

```python
	return {"messages":[AIMessage(content=response.content)]}
```

与第一个节点相同，这里直接指定`return`按state定义构造的一个字典。因为`llm.invoke`的结果是一个`AIMessage`，所以`[response]`也符合**消息列表**的定义。

这里有两个值得说明的地方：

首先，大家在第一个节点的时候，看到我们是将`[HumanMessage]`传给了state，而这里，我们又将`[AIMessage]`传给了state，这下大家理解为什么我们在定义state的时候要用`[AnyMessage]`了吧，就是为了省事。

然后重点来了：**由于我们在构建state的时候指定了`add_message`这个Reducer函数，于是在通过`return`去更新`messages`的值的时候，并不是简单地用`response`去覆盖之前的内容，而是把它与前面已储存的值（比如用户已输入的`[HumanMessage]`拼接成一个列表。这样，我们就实现了历史内容的保存，也就是memory。** 

还有个细节，为了节省token，我们使用`.content`把LLM原始的输出内容中的主要信息提取出来，然后重新包装成`AIMessage`再传回给state。

### 3.利用条件边创建循环

完成了节点逻辑的构建，我们接下来需要确定条件边的逻辑（不熟悉的同学请一定要复习上期内容）。回忆一下，我们设定的业务逻辑中，消息的流动顺序是这样的：

- 用户输入（HumanMessage）--> 模型反馈（AIMessage） --> 用户输入（HumanMessage）--> 模型反馈（AIMessage）......

也就是说，只要用户不明确对话结束，这个反复的过程就会一直循环下去。那么显然，**我们的重点就是先判断用户有没有明确说结束对话（通过设定关键词）。** 只要用户没说关键词，循环就会继续。

于是可以写代码如下：

```python
def should_continue(state:ChatState) -> str:
    """根据用户输入决定是否继续对话"""
    # 获取最后一条消息
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content.lower() == "结束对话":
        return "END"
    return "Chat_Bot"
```

如前所述，由于`add_message`这个Reducer函数的存在，凡是向state传入的内容（`AnyMesaage`)，都会与state中既存的内容拼接起来，形成一个形如`[HumanMessage(),AIMessage(),HumanMessage(),AIMessage(),HumanMessage(),AIMessage()]`的历史消息列表。

因此，上述代码中，我们首先用 `state["messages"][-1]`来获取当前保存在state中的`message`列表中的最后一个元素。

然后，设定判断逻辑：

```python
	if isinstance(last_message, HumanMessage) and last_message.content == "结束对话":
        return "END"
    return "Chat_Bot"
```

其中：
- `isinstance(last_message, HumanMessage)`:判断列表的最后一个元素是否为`HumanMessage`；
- `last_message.content == "结束对话"`:是判断列表的最后一个元素的内容是不是“结束对话”（你可以任意设定）；
- 条件中间用`and`连接，意味着需要同时满足以上两个条件，才会触发`END`分支的路线，否则就返回`Chat_Bot`

### 4.将Node连成Graph

接下来要做的，就是把我们创建好的Node函数添加进Node，然后连接起来了。方法非常简单，就是按照我们设定好的工作流来连接他们，具体代码的写法原理请参见上期内容，这里就不再水字数了。

```python
from langgraph.graph import StateGraph,START,END

graph = StateGraph(ChatState)

graph.add_node("get_user_input", get_user_input)
graph.add_node("Chat_Bot", Chat_Bot)

graph.add_edge(START, "get_user_input")
graph.add_conditional_edges(
    "get_user_input",
    should_continue,
    {
        "Chat_Bot": "Chat_Bot",
        "END": END
    }
)
graph.add_edge("Chat_Bot", "get_user_input")

app = graph.compile()
```

完成了前述工作，我们就可以使用invoke方法来使用我们的Graph了。由于需要一个初始的state，所以我们可以传一个空列表来完成app的启动。

```python
app.invoke({"messages": []})
```
