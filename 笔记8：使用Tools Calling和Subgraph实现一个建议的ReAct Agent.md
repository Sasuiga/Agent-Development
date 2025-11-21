大家好，在上一期文章中，我带着大家构建了一个简单的带记忆的Chat Bot，实现了基本的对话循环功能。今天，我们要在此基础上更进一步，引入两个非常重要的概念：**工具调用（Tool Calling）** 和 **子图（Subgraph）**，从而实现一个简单的**ReAct Agent**。

在具体讲解代码实现之前，我们需要对ReAct Agent、Tool Calling、Subgraph这些概念进行一个基本的阐述。

## 关于ReAct Agent

###  什么是ReAct Agent

在介绍具体实现之前，我们首先必须理解什么是ReAct Agent。

ReAct Agent 是一种目前非常流行的Agent框架，它由 Princeton 和 Google 的研究团队在论文《ReAct: Synergizing Reasoning and Acting in Language Models》（2022年）中提出，核心思想是让模型在解决问题时动态交替进行逻辑推理和环境交互，模仿人类的决策过程。

ReAct Agent的核心工作流程包括以下三个步骤：

1. **Reasoning（推理）**：LLM分析用户的输入，思考需要做什么，决定是否需要调用工具来解决问题
2. **Acting（行动）**：如果需要工具，LLM会生成工具调用请求，然后执行相应的工具
3. **Observing（观察）**：获取工具执行的结果，观察结果是否符合预期

这三个步骤会形成一个循环：推理 → 行动 → 观察 → 再次推理 → 再次行动...直到LLM认为已经收集到足够的信息，可以给出最终答案为止。

### 为什么我们需要ReAct

ReAct模式的出现主要是为了解决传统AI Agent的几个关键问题：

1. **能力边界问题**：纯LLM虽然知识丰富，但无法获取实时信息、执行具体操作或访问外部系统。ReAct模式能够通过工具调用扩展LLM的能力边界。这个思路与我们前期文章中介绍智能体的插件、MCP的作用是完全一致的。

2. **决策智能性**：能调用工具的Agent其实很早就出现了，但简单的工具调用Agent可能会盲目调用所有可用工具，而ReAct模式让Agent能够根据具体情境去智能判断是否需要调用工具，以及调用哪个工具。

3. **复杂问题处理**：对于需要多步骤推理的复杂问题，ReAct的循环机制允许Agent进行多次推理-行动-观察，逐步收集信息并完善答案。

正是这些优势，使得ReAct模式成为构建智能AI Agent的重要范式。

## 什么是工具调用（Tool Calling）

从前述对ReAct Agent的介绍不难看出，其实现的核心机制其实在于**工具**的调用。所谓**工具调用（Tool Calling）**，是一种让AI Agent能够使用外部**工具**的机制。

在我前期介绍**工具**创建的文章中，其实已经跟大家解释了工具的含义及作用。某种意义上，Tools跟Message、Template一样，本质上都是根据不同的目的，对向模型输入的信息进行标准化、规范化的方式。

在LangGraph中，工具调用的流程通常是：
- LLM分析用户输入，决定是否需要调用工具
- 如果需要，LLM会生成工具调用请求（tool_calls）
- 工具节点（ToolNode）执行工具调用
- 工具执行结果返回给LLM
- LLM根据工具结果生成最终回复

后面我们会从代码的角度来讲解如何实现工具的调用，这里就不再赘述了。

## 关于子图（Subgraph）

### 什么是子图

ReAct Agent本身，跟子图（Subgraph）这个概念其实没有必然联系，之所以这里一起讲完全是因为我目前只能利用子图来实现我想要的效果。先给大家科普下什么是子图（Subgraph）

在之前的文章中，我们构建的Graph都是单一层级的结构，所有的节点都在同一个Graph中。但在实际应用中，我们经常会遇到这样的情况：某个功能模块需要多个步骤才能完成，而这些步骤本身又可以形成一个完整的子工作流。

**子图（Subgraph）** 是优化这种工作流的一种方案。简单来说，子图本身是一个独立的Graph，但它又能作为一个节点（Node）嵌入到另一个Graph（父图）中。这样做的好处在于：

1. **模块化设计**：将复杂的功能拆分成独立的子图，可以使代码结构更清晰
2. **代码复用**：子图可以在多个主图中重复使用
3. **层次化管理**：通过嵌套的Graph结构，我们可以更好地组织和管理复杂的Agent逻辑

打个比方，如果主Graph是一个公司的整体架构，那么子图就是公司里的各个部门。每个部门都有自己的工作流程，但最终都会作为一个整体参与到公司的运营中。

### 为什么我要引入子图

一开始我的想法很简单：只是想给上一期那个带memory的Chat Bot加入工具调用的功能，让它能够执行一些实际的操作，比如获取当前时间、查询信息等等。

但当我开始动手写代码的时候，却发现了一个问题：**我没法在一个Graph里同时实现对话的循环与工具调用的循环**。

如果我们把这两个循环都放在同一个Graph中，代码结构会变得非常混乱。比如，当LLM需要调用工具时，我们需要从对话流程跳转到工具调用流程，工具执行完后又要跳回对话流程，但如果LLM在工具结果的基础上还需要再次调用工具，我们又需要再次进入工具调用流程...这样的跳转逻辑会让整个Graph变得难以理解和维护。

于是，我想到了**子图（Subgraph）** 。既然工具调用的逻辑本身就是一个完整的子工作流，那我为什么不把它封装成一个独立的子图呢？

这样，主图就只需要负责管理对话的循环，而工具调用的循环则完全在子图中处理。当主图处理用户输入时发现需要使用工具，它只需要调用子图，子图会自己处理完所有的工具调用逻辑，然后返回最终结果给主图，主图再回复用户即可。

好了，下面我们就来看看如何具体实现一个简单的ReAct Agent~

## 简易ReAct Agent的代码实现

如前所述，我们的这个ReAct Agent将由两个部分组成：
1. 一个子图，负责处理工具调用循环
2. 一个主图，负责管理整个对话循环

站在主图视角来看，结构与上期的Chat Bot非常相似，只是这次我们把LLM节点放到了`sub_graph`代表的子图节点中。

![[output.png]]

从具体信息的传递过程来看，整体工作流如下所示：

1. **用户输入**："现在几点了？"
2. **主图**：将用户输入传递给子图
3. **子图 - Chat_Bot**：LLM分析用户问题，决定需要调用`get_current_time`工具
4. **子图 - 条件判断**：检测到`tool_calls`，路由到`tool_node`
5. **子图 - tool_node**：执行`get_current_time`工具，获取当前时间
6. **子图 - Chat_Bot**：LLM根据工具返回的时间信息，生成最终回复
7. **子图 - 条件判断**：没有新的`tool_calls`，结束子图执行
8. **主图**：显示AI回复，然后返回获取下一个用户输入

接下来，我们先来完成子图的构建。

### 1.构建Subgraph

#### （1）定义子图的State

首先，我们需要给子图一个独立的State定义。大家可以根据自己workflow的具体需求来确定子图state该如何定义。由于我们这里主要处理的还是用户与LLM的对话，所以主图、子图的state结构是完全相同的，只是为了区分，我们给它起个不同的名字，即`SubAgentState`：

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages

class SubAgentState(TypedDict):
	messages: Annotated[Sequence[AnyMessage], add_messages]
```

可以看到，这个State定义与之前带记忆的Agent中的State完全一致，使用`add_messages`作为Reducer函数来实现消息的累积。

#### （2）初始化LLM

这一步的过程与上期完全相同，这里就展示下代码，不做过多解释：

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
llm = init_chat_model("deepseek-chat", model_provider="deepseek")
```
#### （3）创建工具并绑定LLM

接下来，我们定义一个获取当前时间的工具：

```python
from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_time() -> str:
	"""获取当前的日期和时间，返回格式化的时间字符串。当用户询问时间相关问题时使用此工具。"""
	now = datetime.now()
	# 格式化日期和时间
	date_str = now.strftime("%Y年%m月%d日")
	time_str = now.strftime("%H:%M:%S")
	# 获取星期几
	weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
	weekday_str = weekdays[now.weekday()]
	# 判断上午/下午
	hour = now.hour
	if hour < 12:
		period = "上午"
	elif hour < 18:
		period = "下午"
	else:
		period = "晚上"
	# 返回友好的时间描述
	return f"当前时间是：{date_str} {weekday_str} {period} {time_str}"
```

这个工具使用了`@tool`装饰器（不熟悉的同学请查阅前期关于Tools的文章），它会返回一个格式化的时间字符串。当然，这里我的函数体内容看起来很复杂，其实本质上就是个获取当前时间的普通Python代码，理解有难度的同学可以直接复制给AI，让AI 给你逐行讲解

创建好工工具之后，我们还需要将工具绑定到LLM：

```python
tools = [get_current_time]
Agent = llm.bind_tools(tools)
```

- `tools = [get_current_time]`：首先，我们需要把建好的工具放到一个列表里。虽然我们这里只有一个工具，但大家完全可以创建任意多的工具供自己的Agent使用，这样想象空间是非常大的。
- `llm.bind_tools(tools)`：然后，需要将工具列表绑定给LLM，这样LLM就知道有哪些工具可以使用
- `ToolNode(tools)`：这里我们还


#### （4） 构建子图的节点函数及条件边的路由函数

我们的子图结构也非常简单，包含两个主要节点：Chat_Bot节点和tool_node节点。以及一个条件边。

##### 1）Chat_Bot节点

这个节点负责调用LLM生成回复，并根据回复决定是否需要调用工具：

```python
def Chat_Bot(state:SubAgentState) -> SubAgentState:
    """这个节点将使用大语言模型对用户的输入进行反馈"""
    system_prompt = SystemMessage(content="你是一个助手，请根据用户输入选择合适的工具（如有）来回复。")
    messages = [system_prompt] + state["messages"]
    response = Agent.invoke(messages)
    if response.content:
        print(f"\nAI:{response.content}\n")
    return {"messages":[response]}
```

这里有几个要点：
- 这里我们添加了一个`SystemMessage`来指导LLM的行为，为了优化效果，在SystemPromp中明确要求LLM要注意根据用户输入来选择工具。
- 其余代码与前期无本质差异，这里使用`Agent.invoke(messages)`调用LLM，由于Agent已经绑定了工具，LLM可能会在回复中包含工具调用请求。而只要返回的response中包含了`tool_calls`，就将在后续的条件边中触发相关的路径选择。

##### 2）Tools节点

  Tools节点的搭建非常简单，只需要向`ToolNode()`函数传入我们前面定义好的工具列表`tools`即可。  Tools节点会自动执行LLM请求的工具调用

```python
tool_node = ToolNode(tools)
```

##### 3）条件路由函数

最后，我们需要一个函数来判断LLM的回复是否需要调用工具：

```python
def should_continue_1(state: SubAgentState) -> str:
    """根据AI响应决定是否需要使用工具（如有）"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "END"
    return "tool_node"
```

这个函数检查最后一条消息是否包含`tool_calls`：
- 如果没有工具调用，返回"END"，结束子图的执行
- 如果有工具调用，返回"tool_node"，继续执行工具调用

#### （5）构建子图

现在我们可以构建子图了，整个过程与前期已讲解的内容没有本质差异。

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

sub_graph = StateGraph(SubAgentState)
sub_graph.add_node("Chat_Bot", Chat_Bot)
sub_graph.add_node("tool_node", tool_node)

sub_graph.add_edge(START, "Chat_Bot")
sub_graph.add_conditional_edges(
    "Chat_Bot",
    should_continue_1,
    {
        "tool_node": "tool_node",
        "END": END
    }
)
sub_graph.add_edge("tool_node", "Chat_Bot")

subgraph = sub_graph.compile()
```

子图的工作流程如下：
1. 从START开始，进入Chat_Bot节点
2. Chat_Bot调用LLM生成回复
3. 通过条件边判断是否需要调用工具
4. 如果需要，进入tool_node执行工具调用
5. tool_node执行完后，返回Chat_Bot节点（形成循环）
6. 如果不需要工具，直接结束，Chat_Bot直接将回复传出。

这个循环机制很重要：如果LLM在一次工具调用后还需要再次调用工具，或者需要根据工具结果生成最终回复，这个循环就能保证流程的完整性。

### 2. 构建主图

主图负责管理整个对话流程，包括获取用户输入和调用子图。主图的代码内容与之前带记忆的Chat Bot基本一致，唯一的区别就是主图中的`Chat_Bot`节点被替换为了`subgraph`节点，而`subgraph`节点其实就是我们刚才编译好的子图，它作为一个节点被嵌入到主图中

具体代码如下，细节不再赘述：

```python
class AgentState(TypedDict):
	messages: Annotated[Sequence[AnyMessage], add_messages]

def get_user_input(state: AgentState) -> AgentState:
    """这个节点获取用户输入并添加到消息历史中"""
    user_input = input("输入: ")
    return {"messages": [HumanMessage(content=user_input)]}

def should_continue_2(state: AgentState) -> str:
    """根据用户输入决定是否继续对话"""
    # 获取最后一条消息
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content == "结束对话":
        return "END"
    return "subgraph"

# 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("get_user_input", get_user_input)
graph.add_node("subgraph", subgraph)

# 设置图的边和条件路由
graph.add_edge(START, "get_user_input")

# 用户输入后，判断是否结束对话
graph.add_conditional_edges(
    "get_user_input",
    should_continue_2,
    {
        "subgraph": "subgraph",
        "END": END
    }
)

graph.add_edge("subgraph", "get_user_input")

app = graph.compile()
```

主图的工作流程如下：
1. 从START开始，获取用户输入
2. 判断用户是否要结束对话
3. 如果不结束，将用户输入传给子图处理
4. 子图处理完成后，返回主图，继续获取下一个用户输入
5. 如果用户说"结束对话"，则结束整个流程

### 3. 启动对话

最后，我们使用invoke方法来启动整个Agent：

```python
app.invoke({"messages": []})
```
## 完整代码

由于本期内容相对复杂一点，为了方便大家理解和复现，我把完整的代码整理如下：

```python
from typing import Annotated,Sequence,TypedDict
from langchain_core.messages import HumanMessage,AIMessage,AnyMessage,ToolMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from datetime import datetime
from dotenv import load_dotenv



class SubAgentState(TypedDict):
	messages:Annotated[Sequence[AnyMessage],add_messages]

load_dotenv()

from langchain.chat_models import init_chat_model
llm = init_chat_model("deepseek-chat", model_provider="deepseek")

# 定义获取当前时间的工具
@tool
def get_current_time() -> str:
	"""获取当前的日期和时间，返回格式化的时间字符串。当用户询问时间相关问题时使用此工具。"""
	now = datetime.now()
	# 格式化日期和时间
	date_str = now.strftime("%Y年%m月%d日")
	time_str = now.strftime("%H:%M:%S")
	# 获取星期几
	weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
	weekday_str = weekdays[now.weekday()]
	# 判断上午/下午
	hour = now.hour
	if hour < 12:
		period = "上午"
	elif hour < 18:
		period = "下午"
	else:
		period = "晚上"
	# 返回友好的时间描述
	return f"当前时间是：{date_str} {weekday_str} {period} {time_str}"

tools = [get_current_time]

def Chat_Bot(state:SubAgentState) -> SubAgentState:
    """这个节点将使用大语言模型对用户的输入进行反馈"""
    system_promt = SystemMessage(content="你是一个助手，请根据用户输入选择合适的工具（如有）来回复。")
    messages = [system_promt] + state["messages"]
    response = Agent.invoke(messages)
    if response.content:
        print(f"\nAI:{response.content}\n")
    return {"messages":[response]}

Agent = llm.bind_tools(tools)

tool_node = ToolNode(tools)

def should_continue_1(state:SubAgentState) -> str:
    """根据AI响应决定是否需要使用工具（如有）"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "END"
    return "tool_node"


sub_graph = StateGraph(SubAgentState)
sub_graph.add_node("Chat_Bot", Chat_Bot)
sub_graph.add_node("tool_node", tool_node)

sub_graph.add_edge(START, "Chat_Bot")
sub_graph.add_conditional_edges(
    "Chat_Bot",
    should_continue_1,
    {
        "tool_node": "tool_node",
        "END": END
    }
)
sub_graph.add_edge("tool_node", "Chat_Bot")


subgraph = sub_graph.compile()


class AgentState(TypedDict):
	messages:Annotated[Sequence[AnyMessage],add_messages]


# 定义节点函数
def get_user_input(state:AgentState) -> AgentState:
    """这个节点获取用户输入并添加到消息历史中"""
    user_input = input("输入: ")
    return {"messages":[HumanMessage(content=user_input)]}


def should_continue_2(state:AgentState) -> str:
    """根据用户输入决定是否继续对话"""
    # 获取最后一条消息
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content == "结束对话":
        return "END"
    return "subgraph"


# 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("get_user_input", get_user_input)
graph.add_node("subgraph", subgraph)

# 设置图的边和条件路由
graph.add_edge(START, "get_user_input")

# 用户输入后，判断是否结束对话
graph.add_conditional_edges(
    "get_user_input",
    should_continue_2,
    {
        "subgraph": "subgraph",
        "END": END
    }
)

graph.add_edge("subgraph", "get_user_input")

app = graph.compile()

# 启动对话
app.invoke({"messages": []})
```

## 总结

通过今天的学习，我们掌握了两个重要的概念：

1. **子图（Subgraph）**：可以将复杂的逻辑封装成独立的Graph，作为节点嵌入到主图中，实现模块化设计
2. **工具调用（Tool Calling）**：让Agent能够使用外部工具，扩展LLM的能力边界

将这两个功能结合起来，我们可以构建出功能强大、结构清晰的Agent系统。在实际应用中，你可以：
- 为不同的功能模块创建不同的子图
- 在子图中实现复杂的工具调用逻辑
- 通过主图统一管理整个对话流程

希望今天的分享对大家有帮助！下一期，我会继续带大家探索LangGraph的更多高级功能~
