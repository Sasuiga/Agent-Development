
前面已经陆续给大家分享了一些利用LangGraph的学习笔记，相信大家已经开始大开脑洞搭建自己的AI Agent了。本期我想放缓一下脚步，再回到一些比较重要的概念上来加深一下学习，以提升对LangGraph这个工具的掌控感。

要说LangGraph中重要的概念，那我第一个想到的就是State了。**无论我们定义什么样的节点、边、图，核心目的都是希望通过对Graph的State的操作，以达到我们想要的效果。**

本期我将带着大家结合官方文档，从state的**定义、更新与管理**三个维度，来温故知新。

## 关于State的定义

LangGraph中的State，可以用`TypedDict`，`Pydantic model`，或者`dataclass`来定义，其中前两种比较常用。

### TypedDict：轻量级的选择

`TypedDict`主打一个**轻量级**和**简单直接**。它本质上就是Python标准库`typing`模块中的一个类型提示工具，不需要安装额外的依赖，使用起来非常方便。

正如我在前期文章中给大家展示的，用`TypedDict`定义State的典型写法如下：

```python
from typing import TypedDict

class AppState(TypedDict):
    Key_1: str
    Key_2: int
```

需要注意的是，由于`typing`本质上是个类型提示工具，它只在开发阶段（通过IDE或类型检查工具）发挥提示作用，而并不会在应用实际运行时对其中的参数、键值进行强制的类型验证。举个例子，虽然我们已经设定Key_1的数据类型为str，但如果我们向它传入一个int，程序也不会报错（但可能会导致后续逻辑错误）。因此。如果你需要更严格的运行时验证，那就需要考虑使用`Pydantic model`了。

### Pydantic Model：更强大的选择

`Pydantic`是一个强大的数据验证库，它可以在项目运行时对数据进行验证，确保数据的类型和格式符合预期。这对于构建健壮的AI Agent来说是非常重要的。

下面我们来看一个使用`Pydantic model`定义State的示例：

```python
from pydantic import BaseModel, Field

class AppState(BaseModel):
    Key_1: str
    Key_2: int = Field(ge=0, le=100)  # 限制Key_2的值必须在0到100之间
    Key_3: str = "默认值"  # 如果没传这个字段，就自动使用"默认值"
```

可以看到，与`TypedDict`相比，`Pydantic model`提供了几个非常实用的功能：

1. **自动验证数据**：当你传入的数据不符合要求时，Pydantic会在程序运行时立即报错。比如上面例子中，如果你传入的`Key_2`是负数或者大于100，程序就会抛出异常，告诉你数据有问题。

2. **设置默认值**：如果某个字段是可选的，你可以给它设置一个默认值。比如上面的`Key_3`，如果调用时没有传这个字段，它就会自动使用"默认值"。这样就不需要每次都手动检查字段是否存在了。

3. **限制取值范围**：通过`Field`可以给数值类型的字段设置范围限制。比如`ge=0`表示"大于等于0"（ge是greater or equal的缩写），`le=100`表示"小于等于100"（le是less or equal的缩写）。这样就能确保数据在合理的范围内，避免出现负数等级、超过上限的血量这种不合理的情况。

当然`Pydantic`还提供了其他一些更复杂的功能，比如：

- **自定义验证器**：可以写自己的验证函数，对字段值进行更复杂的检查，比如验证邮箱格式、密码强度等
- **字段间的依赖关系**：可以让一个字段的值依赖于另一个字段，比如总价 = 单价 × 数量
- **数据转换**：可以在验证时自动转换数据类型，比如把字符串"123"自动转换成整数123
- **嵌套模型**：可以在一个模型中包含另一个模型，实现复杂的数据结构

这些功能在构建复杂应用时非常有用。如果你想深入了解，可以查看Pydantic的官方文档：https://docs.pydantic.dev/

那么在实际开发中，我们应该如何选择呢？通常的建议是：

- `TypedDict`：如果你的项目比较简单，不需要复杂的验证逻辑，或者你希望保持代码的轻量级，那么`TypedDict`是个不错的选择
- `Pydantic model`：如果你需要严格的数据验证，或者你的State结构比较复杂，需要字段间的依赖关系，那么`Pydantic model`会更合适

## 关于State的更新

在LangGraph中，State的更新其实是一个**两阶段**的过程：

1. **第一阶段：节点函数处理** - Node函数接收当前的state，根据业务逻辑进行处理，计算出需要更新的新值。
2. **第二阶段：Reducer决定更新方式** - Reducer函数决定如何将节点返回的新值更新到state中，是覆盖（overwrite）还是拼接（append）等

理解这两个阶段的工作机制，对于构建功能完善的Agent至关重要。

### 第一阶段：节点函数处理

节点函数是State更新的第一步。它接收当前的state作为输入，根据业务逻辑进行处理，然后返回一个字典，包含需要更新的字段和新值。

回顾一下我们在【】中的例子：

```python
def dodge_check_node(state: AppState) -> AppState:
    """检查闪避结果并处理HP变化"""
    dice_roll = random.randint(1, 6)
    
    if dice_roll > 3:
        print("你成功闪避了伤害")
        return {}  # 没有变化，返回空字典
    else:
        new_HP = state['char_HP'] - dice_roll
        print(f"闪避失败！受到{dice_roll}点伤害")
        return {"char_HP": new_HP}  # 返回需要更新的字段和新值
```

这里，节点函数计算出了新的HP值，并返回`{"char_HP": new_HP}`。但这时候，我们还无法判断这个新值将如何对state产生影响，因为这是由定义state时设置的Reducer来决定的。

### 第二阶段：Reducer决定更新方式

Reducer函数决定了节点返回的新值如何更新到state中，每一个键都可以有其独立的Reducer。这是State更新的关键环节，不同的Reducer会产生不同的更新效果。

#### Reducer的基本使用方法

在前期文章中，我向大家介绍了一种常用的Reducer，`add_messages`，的使用方法：

```python
from typing import Annotated, List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
```

这里的关键是`Annotated`的使用。`Annotated`是Python的类型注解工具，它允许我们为类型添加额外的元数据。在LangGraph中，我们用它来指定Reducer函数。

`add_messages`的作用是：
- 将新消息追加到现有消息列表的末尾
- 如果新消息的ID与旧消息相同，则更新旧消息而不是追加

除了`add_messages`，LangGraph中还有其他一些内置的Reducer，比如：

1. **`operator.add`**：用于数值累加，将新值加到旧值上。比如统计总分、计数等场景：

```python
from typing import Annotated, TypedDict
from operator import add

class ScoreState(TypedDict):
    total_score: Annotated[int, add]  # 每次返回的分数会累加到总分上
```

2. **`operator.extend`**：用于列表扩展，将新列表的元素追加到当前列表中。比如收集多个节点的结果：

```python
from typing import Annotated, List, TypedDict
from operator import extend

class ResultState(TypedDict):
    results: Annotated[List[str], extend]  # 新列表的元素会追加到现有列表中
```

除了这些内置Reducer，LangGraph也允许我们自定义Reducer来实现想要的效果。

#### 自定义Reducer

当内置的Reducer无法满足需求时，我们可以自定义Reducer函数实现对state更新过程的控制。自定义Reducer的过程非常简单，只需要定义一个函数即可。

**自定义Reducer的基本要求：**

1. **参数**：Reducer必须接收两个参数，依次从左到右分别为：
   - `old_value`：state中该字段的当前值
   - `new_value`：节点函数返回的新值
2. **返回值**：返回合并后的结果，类型应该与字段类型一致

下面我我给大家一个简单的例子。我们定义一个作用是取新旧值中的较大者的Reducer，并将它设定在 state中。

```python
from typing import Annotated, TypedDict

# 定义一个自定义Reducer函数
def take_max(old_value: int, new_value: int) -> int:
    """自定义Reducer：取新旧值中的较大者"""
    return max(old_value, new_value)

# 在State定义中使用自定义Reducer
class AppState(TypedDict):
    max_score: Annotated[int, take_max]  # 使用自定义的take_max作为Reducer
```

然后，我们假设一个更新分数的节点：

```python
def update_score_node(state: AppState) -> AppState:
    """更新分数"""
    return {"max_score": 85}  # 返回新的分数值
```

上述代码意味着，根据这个节点函数的计算操作（被简化），输出max_score的值为85，但由于我们设定了Reducer，所以这个85不会直接对state进行更新，而是需要与经过这个节点之前的state中保存的值（这才是当前的state！）进行比较：

如果当前max_score是80，节点返回85，Reducer会取max(80, 85) = 85
如果当前max_score是90，节点返回85，Reducer会取max(90, 85) = 90

是不是很酷炫？
#### Overwrite：强制覆盖机制

有时候，即使我们已经为某个字段设定了Reducer（比如`add_messages`），但在某些特殊场景下，我们可能希望忽略这个Reducer，直接用新值覆盖state中的旧值。这时候，我们就可以使用`Overwrite`机制。

`Overwrite`是LangGraph提供的一种特殊机制，它允许我们在节点返回时，强制忽略已设定的Reducer，直接覆盖state中的值。

LangGraph提供了两种方式使用`Overwrite`的方式：

1. **使用`Overwrite`类型包装值**：

```python
from langgraph.graph import Overwrite

def reset_node(state: ChatState) -> ChatState:
    """重置消息列表，忽略add_messages的Reducer"""
    return {"messages": Overwrite([])}  # 直接覆盖为空列表，忽略add_messages的拼接逻辑
```

2. **使用`__overwrite__`键**：

```python
def reset_node(state: ChatState) -> ChatState:
    """重置消息列表，忽略add_messages的Reducer"""
    return {"messages": {"__overwrite__": []}}  # 效果与上面相同
```

`Overwrite`在以下场景特别有用：
- **重置状态**：需要清空累积的数据（比如重置消息历史）
- **强制替换**：需要完全替换某个字段的值，而不考虑之前的累积结果
- **特殊处理**：某些节点需要绕过Reducer的常规逻辑

需要注意的是，如果不指定Reducer，LangGraph默认就会直接覆盖（相当于自动使用overwrite），所以`Overwrite`主要用于**在已设定Reducer的情况下强制覆盖**的场景。

## 关于State的管理

在实际应用中，我们有时候需要更精细地管理State的可见性和访问权限。LangGraph提供了几种机制来实现这个目标，包括**input schema、output schema**和**private schema**。

### input schema和output schema

在某些场景下，我们可能希望Graph对外暴露的接口与内部使用的State结构有所不同。比如，我们可能希望用户只需要传入简单的参数，而不需要了解Graph内部复杂的State结构。

这时候，我们就可以使用**input schema**和**output schema**来定义Graph的输入和输出格式。

#### 基本用法

假设我们有一个内部State，结构比较复杂：

```python
from typing import TypedDict, List

class InternalState(TypedDict):
    user_name: str
    user_age: int
    messages: List[str]
	result: str
    status: str
    internal_counter: int
    debug_info: dict
```

但我们希望用户调用Graph时，只需要传入用户名和年龄，而不需要关心内部的`messages`、`internal_counter`、`debug_info`等字段。同时，我们也希望Graph返回给用户的只是处理结果，而不是整个内部State。

这时候，我们可以分别定义input和output schema：

```python
from typing import TypedDict

class InputSchema(TypedDict):
    """用户输入格式"""
    user_name: str
    user_age: int

class OutputSchema(TypedDict):
    """用户输出格式"""
    result: str
    status: str
```

然后在创建Graph时指定这些schema：

```python
graph = StateGraph(InternalState,input_schema=InputSchema,
    output_schema=OutputSchema)
```

#### 效果说明

使用input/output schema后，Graph的行为会发生以下变化：

**1. 输入转换：**

用户调用时，只需要传入InputSchema格式的数据

```python
result = app.invoke({"user_name": "张三", "user_age": 25})
```

 LangGraph会自动将InputSchema转换为InternalState，未输入的地方会自动初始化为空值。 转换后的内部state如下：
 
 ```python
{
	"user_name": "张三",
	"user_age": 25,
	"messages": [],  # 自动初始化为空列表
	"internal_counter": 0,  # 自动初始化为0
	"debug_info": {}  # 自动初始化为空字典
}
 ```
 
**2. 输出转换：**

Graph执行完成后，内部state可能是：

```python
{
	"user_name": "张三",
	"user_age": 25,
	"messages": ["欢迎张三！"],
	"result": "处理完成",
	"status": "success"
	"internal_counter": 1,
	"debug_info": {"processed": True}
}
```

这时，OutputSchema就像一个筛选器一样，只会向用户返回在其中定义的键值。在我们的示例中，实际返回给用户的sate可能是：

```python
{
	"result": "处理完成",
	"status": "success"
}
```

**关键点：**
- input schema定义了用户调用`invoke()`时需要传入的字段格式
- output schema定义了Graph返回给用户的字段格式
- 内部State的完整结构仍然存在，只是在输入输出时进行了转换
- 如果output schema中的字段在内部State中不存在，需要在节点中返回这些字段，或者使用转换函数

### private schema

有时候，我们希望在Graph的执行过程中使用一些中间状态，这些状态只在节点之间传递，但不应该出现在Graph的最终输出中。比如：

1. **中间计算结果**：某些节点需要计算中间值，这些值会被后续节点使用，但不需要对外暴露；
2. **临时数据存储**：在Graph执行过程中需要临时存储一些数据，用于节点间的信息传递；
3. **内部状态管理**：需要跟踪Graph内部的执行状态，但这些状态对用户来说是不必要的。

LangGraph提供了**private state**机制来实现这个需求。
#### 基本用法

private state的使用方法非常简单，就是是定义一个独立的`PrivateState`类型，然后在节点函数中使用它。关键点在于：

- 定义独立的`PrivateState`类型，包含需要在节点间传递但不对外暴露的字段
- 节点函数可以接收`PrivateState`作为输入，也可以返回`PrivateState`作为输出
- `PrivateState`中的字段会在节点间正常传递，但不会出现在Graph的最终输出中

下面我们通过一个示例来理解private state的使用：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义Graph的整体状态：包含公有字段和私有字段
class OverallState(TypedDict):
    user_input: str      # 公有字段，会出现在最终输出中
    graph_output: str    # 公有字段，会出现在最终输出中
    foo: str            # 公有字段，会出现在最终输出中

# 定义私有状态：只在节点间传递，不会出现在最终输出中
class PrivateState(TypedDict):
    bar: str            # 私有字段，只在节点间传递

# 节点1：处理用户输入，写入OverallState
def node_1(state: OverallState) -> OverallState:
    # 处理用户输入，生成中间结果
    return {"foo": state["user_input"] + " name"}

# 节点2：读取OverallState，写入PrivateState
def node_2(state: OverallState) -> PrivateState:
    # 从OverallState读取foo，计算中间值并存入PrivateState
    return {"bar": state["foo"] + " is"}

# 节点3：读取PrivateState，写入OverallState
def node_3(state: PrivateState) -> OverallState:
    # 从PrivateState读取bar，生成最终输出
    return {"graph_output": state["bar"] + " Lance"}

# 创建Graph，只使用OverallState作为主状态
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()

# 调用Graph
result = graph.invoke({"user_input": "My"})
print(result)
```

这段代码的运行结果将是：`{'user_input': 'My', 'graph_output': 'My name is Lance', 'foo': 'My name'}`。
#### 效果说明

**1. 私有状态在节点间正常传递：**

在上面的示例中，`PrivateState`中的`bar`字段在`node_2`中被设置，然后在`node_3`中被读取使用。这说明private state在Graph内部的节点之间是完全可用的，可以正常传递数据。

**2. 私有状态不会出现在最终输出中：**

虽然`PrivateState`中的`bar`字段在Graph执行过程中被使用（在`node_2`中写入，在`node_3`中读取），但最终的输出结果中只包含`OverallState`中定义的字段（`user_input`、`graph_output`、`foo`），`bar`字段被自动过滤掉了。
## 总结

本期我们从三个维度深入了解了LangGraph中的State机制：

**State的定义**：可以使用`TypedDict`或`Pydantic model`来定义State。`TypedDict`轻量级但只有类型提示，`Pydantic model`提供运行时验证、默认值设置等更强大的功能。

**State的更新**：State更新分为两个阶段——节点函数处理业务逻辑并返回新值，Reducer决定如何将新值合并到State中。LangGraph提供了`add_messages`、`operator.add`等内置Reducer，也支持自定义Reducer。当需要强制覆盖时，可以使用`Overwrite`机制。

**State的管理**：通过`input_schema`和`output_schema`可以控制Graph的输入输出格式，简化对外接口。通过`private state`可以定义只在节点间传递但不对外暴露的中间状态，保持内部实现细节的封装性。

理解这些机制有助于我们更好地设计和管理Agent的状态流转，构建更清晰、更易维护的Graph结构。

好了，以上就是本期的主要内容，希望对大家有帮助，喜欢的朋友别忘了点赞、收藏、转发~祝大家玩的开心~