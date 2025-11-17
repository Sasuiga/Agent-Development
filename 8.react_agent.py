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
llm = init_chat_model("deepseek-reasoner", model_provider="deepseek")

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
    system_promt = SystemMessage(content="你是一个助手，请根据用户输入选择合适的工具（如有）来查询相关信息，然后根据工具返回的信息回答用户的问题。")
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