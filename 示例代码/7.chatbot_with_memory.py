from dotenv import load_dotenv
from typing import TypedDict,List,Annotated,Literal
from langchain_core.messages import HumanMessage,AIMessage,AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END

load_dotenv()

from langchain.chat_models import init_chat_model
llm = init_chat_model("deepseek-chat", model_provider="deepseek")

class ChatState(TypedDict):
	messages:Annotated[List[AnyMessage],add_messages]

def get_user_input(state:ChatState) -> ChatState:
    """这个节点获取用户输入并添加到消息历史中"""
    user_input = input("输入: ")
    return {"messages":[HumanMessage(content=user_input)]}

def Chat_Bot(state:ChatState) -> ChatState:
    """这个节点将使用大语言模型对用户的输入进行反馈"""
    response = llm.invoke(state["messages"])
    print(f"\nAI:{response.content}\n")
    return {"messages":[AIMessage(content=response.content)]}

def should_continue(state:ChatState) -> Literal["Chat_Bot","END"]:
    """根据用户输入决定是否继续对话"""
    # 获取最后一条消息
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content == "结束对话":
        return "END"
    return "Chat_Bot"

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

# 启动对话
app.invoke({"messages": []})