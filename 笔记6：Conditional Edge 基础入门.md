大家好，在上一期文章中，我带着大家搭建了一个基础版的Graph。今天我们要在此基础上，引入Conditional Edge（条件边）的概念，让我们的Graph更加好玩一点。

## 什么是Conditional Edge（条件边）

Conditional Edge（条件边）顾名思义，就是带条件判断的边，即根据条件是否满足，决定走哪条边。

如何大家有任何理解上的困惑，非常建议去看看我之前有篇写Coze的选择器节点的文【】，这两个东西本质上都是对编程中“if-elif-else”的条件判断控制的实现。

我们先从图形上来感受一下条件边：

![[Pasted image 20251029221151.png]]

可以看到，与上期的简单线条结构相比，今天我们要创建的图，多了一个**router**节点，然后从它出发，又分出了防御和闪避两个分支。也就是说，根据**router**设定的逻辑，我们的角色要么选择防御，要么选择闪避。

## 构造条件边的基本原理

条件边的构造方法非常简单，主要分为三个步骤：

### 1.构造routing_function

所谓的routing_function（路由函数），就是指对数据到底如何流动进行定义的函数。而这种定义利用的就是python中非常基础的“if-elif-else“的条件判断。

一个基本的routing_function的结构如下：

```python
def routing_function(state:AgentState) -> AgentState:
	"""This node will select the next node of the graph"""
	if state["KEY"] == "condition":
		return "edge_1"
	else:
		return "edge_2"
```

以上函数的意思就是，如果目前Graph的state中的某个`KEY`的值为`condition`，那么数据就会沿着名为`edge_1`的边流动。否则，就沿着名为`edge_2`的边流动。当然，你也可以用`elif`增加更多种的条件分支。

这里有几个要点：
- 1、一定注意加docstring（就是被“”“包裹的那团注释），因为当引入LLM之后，LLM会根据我们的注释来操作数据。
- 2、routing_function中的判断条件设置非常灵活，这里虽然是以判断state的key的内容为例，但实际上你完全可以自定义条件。
- 3、注意：return返回的是“边”的名字，它是一个字符串。

### 2.添加条件Node

如上期所讲，当我们完成State的初始化之后，首先需要添加Node。添加方法如下：

```python
graph.add_node("节点名",节点函数)
```

同样的，条件Node，如我们前面图中的`router`，也需要添加。但条件Node的添加方法与普通Node有个细微的差异。

由于routing_function与普通的节点函数不同，它不会对Graph的state数据进行操作，它最多利用state的数据进行一些条件满足方面的判断。因此，事实上输入routing_function的state，与其输出的state，是没有任何差异的。

因此，条件Node的添加方法如下：

```python
graph.add_node("条件节点名",lambda state：state)
```

其中节点函数的位置被一个`lambda`函数所替代，你仍然不需要知道这个的原理是什么，你只需要知道这代表着输入输出条件Node的state是相同的即可。

### 3.添加条件边

最后，我们终于来到添加条件边的环节。先来看基本代码：

```python
graph.add_conditional_edges(
	"router",
	routing_function,
	{
		"edge_1":"node_name1",
		"edge_2":"node_name2"
	}
)
```

依然是非常简单，利用了一个`add_conditional_edges`方法。这个方法需要三个参数，分别是：

- **上游节点的名字**：即分支是从哪个节点出发的，如我们前面的图中，就是从router分出了闪避和防御这两个分支。
- **routing_function**：这个不用多说了
- **path_map**：这是一个字典结构的数据，它所包含的键值对就代表着分支的走向，其中每个键值对的结构为：**边的名字：对应的下游节点的名字**

完成以上三步，我们的条件边就添加完成了。

下面我带大家来实现一下前面图中的Graph。

## 实战演练

### 1.定义State

```python
from typing import TypedDict

class AppState(TypedDict):
	char_class: str
	char_level:int
	char_HP:int
    char_action: str
	
```

在上期的state 的基础上，我增加了一个`char_action`键，并将它的值的类型设定为str。这里的想法是我们直接传入动作，是闪避还是防御，然后再由router根据我们传入的动作来确定数据走向。

我这里主要因为是演示，所以搞这么简单，大家可以尽情开脑洞去设计自己的条件判断。

### 2.定义各节点函数

为了避免冗余，这里我只分析与上期相比增加的代码。没有看过上期的朋友一定要去看啊！

首先，写出router所需的routing_function:

```python
def decide_next_node(state:AppState) -> AppState:
	"""This node will select the next node of the graph"""
	if state["char_action"] == "闪避":
		return "dodge"
	elif state["char_action"] == "防御":
		return "defend"
```

非常简单，不再赘述。

然后，增加一个“防御”节点的函数

```python

def defend_check_node(state: AppState) -> AppState:
    """检查防御结果并处理HP变化"""
    dice_roll = random.randint(1, 3)
    
    state['char_HP'] = state['char_HP'] - dice_roll
    print(f"你受到了{dice_roll}点伤害")
        
    return state
```

这里同样是搞得很简单，你完全可以在state中增加一个“防御值”的key，然后在角色受到攻击时，用敌方伤害减去防御值的净额，作为角色最终受到的伤害。总之非常灵活。

### 3.搭建Graph

这里也非常简单，因为要点前面都讲了，直接给出代码：

```python
graph = StateGraph(AppState)

graph.add_node("dodge_check",dodge_check_node)
graph.add_node("defend_check",defend_check_node)
graph.add_node("character_status",display_character_status)
graph.add_node("router",lambda state:state)


graph.add_edge(START,"router")

graph.add_conditional_edges(
	"router",
	decide_next_node,
	{
		"dodge":"dodge_check",
		"defend":"defend_check"
	}
)

graph.add_edge("dodge_check","character_status")
graph.add_edge("defend_check","character_status")
graph.add_edge("character_status",END)


app = graph.compile()
```

写这一部分代码的关键是你自己要对你的Graph的数据流向心里有数，所以一般都是先画好结构草图，真正写代码的过程其实是很简单的。

### 4.使用Graph

最后就是实际使用了。我们依然是向`invoke`方法传入一个与state定义一致的字典，我们先把action设定为闪避：

```python
result = app.invoke({"char_class":"法师","char_level":5,"char_HP":10,"char_action":"闪避"})
```

结果如下，woops，闪避失败，被砍了3滴血。

![[Pasted image 20251029225729.png]]

我们再试试让角色防御，稳如泰山：

![[Pasted image 20251029225850.png]]