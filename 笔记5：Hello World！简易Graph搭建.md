大家好，上一期我用Excel表为例大概解释了一下LangGraph的工作原理，今天我们就来着手搭建一个简单的Graph，目的是通过实践来加深对State、Node、Edge这些概念的理解。

## 关于State的一个补充

首先需要给大家解释一下，从严格意义上来讲，LangGraph中的State是由**Schema**和**Reducer**共同组成的。

在State中，Schema是必须自定义的，因为它决定了Agent所需的业务字段、各业务字段要求的数据类型，而Reducer作为一种状态（字段值）更新的控制机制，如果用户不手动设置也没事，因为会被默认为“覆盖”。

**出于循序渐进，尽快让大家上手的考虑，我所表述的“定义State”，主要指的是对Schema的定义，即对所有节点、边的输入模式的定义。后续我会在适当的位置再引入Reducer。**

## 一、定义State

### 从表格到字典

正如上期所述，我们用LangGraph创建一个AI Agent，就像是设计一个Excel表格一样，是为了通过对相关数据进行处理以获得一个结果。

为了获得这个结果，在Excel中，我们需要对表格的表头字段进行定义，而在LangGraph中，我们则是对Agent的State进行定义。

同样想想Excel表格，每个表头字段填写在第一行，而第二行则代表了各个字段对应的值。如下图所示，“”职业“”对应A2单元格内容、“等级”对应B2单元格内容、“血量”对应C2单元格内容。

![[System/Attachment/公众号文章/20251023/1.png]]

这种成对出现的数据形式，在编程中往往是用一种叫做**字典（Dict）** 的数据结构来储存，其中，表头三个字段，被称为**键**，而第二行单元格的内容，被称为键的**值**。换言之，Dict是一种使用**键值对**来储存数据的结构。

而在LangGraph中，State其实也是一种字典，只不过为了保证相关运行数据类型的准确性以提升Agent性能，LangGraph中一般是用**TypedDict**来定义State。（也有其他方式，比如pydantic的数据模型）

考虑到本系列教程的定位，我们不会太纠结一些编程理论方面的东西，比如你不需要完全搞明白什么是Dict、TypedDict，你只需要知道TypedDict允许你对State所涉及的数据类型（是字符串还是整数、布尔值等）进行限定就行了。如果你真的很有兴趣，可以直接找个AI去问。

下面先给出定义State的代码如下，大家可以拿它跟前面Excel表格图片的内容比较一下。

```python
from typing import TypedDict

class AppState(TypedDict):
	char_class: str
	char_level:int
	char_HP:int
```
### 代码解读

```python
from typing import TypedDict
```
第一行很好理解，从typing中引入TypedDict。

```python
class AppState(TypedDict):
```
接着，我们定义一个名叫AppState的**类（class）**，这个类就是我们所需要的State了。它的名字是你自己任意取得，想叫啥都可以，只要保证可读性即可。

同时，括号中填入TypedDict，意味着我们将这个类的类型定义为Type Dictionary，这样我们就可以对构成State的键值对的数据类型进行限制。


```python
	char_class: str
	char_level:int
	char_HP:int
```
接下来的三行缩进代码，分别为三个键值对，这就是State的主体了。这三个键值对的键（即冒号左边部分），与前面图中Excel表格内列示的三个表头字段，即职业、等级和血量，一一对应。

而冒号右边的内容，代表着我们对能填入这个键的数据的类型的限制，即：

- 职业：必须是字符串
- 等级：必须是整数
- 血量：必须是整数

至此，我们就创建了一个简单的State，是不是很简单。

## 二、创建Node函数

定义好了State，接着就需要创建Node函数。在LangGraph中，Node和一些特殊的Edge，本质上都是函数，需要先定义出来，然后再加入到Graph中。我们先来学习如何创建Node函数。

而在此之前，我建议大家回顾一下我前面对**Tools**的介绍，你就会发现很多内容都是相通的。换言之，我们仍然可以通过三个步骤来创建Node函数。

### 1.定义函数

由于我们现在做的是一个Hello World Graph，所以不想搞得太复杂。从前面的State不难看出，我要做的Graph是一个与游戏角色相关的东西，这里我打算搞两个Node，一个用来处理角色HP的变动（比如是否成功闪避攻击），一个用来展示角色当前状态。

先以第一个Node的函数为例，我们首先需要定义它：

```python
def dodge_check_node(state: AppState) -> AppState:
```

这段代码定义了一个名为`dodge_check_node`的函数，这个函数有一个参数`state`，然后通过类型提示，指定`state`期望接收的数据类型，以及该函数应输出的数据类型。这个数据类型，就是我们前面定义好的类型为Typed Dictionary的state。

由于本例中我们只定义了一个state，即AppState，所以没啥好说的，输入输出都是它。但LangGraph其实允许我们创建多个不同State，比如公有私有、输入输出，允许我们进行非常灵活的state操作，这个后面再讲。

### 2.利用函数文档字符串进行工具描述

```python
def dodge_check_node(state: AppState) -> AppState:
    """检查闪避结果并处理HP变化
    生成1-6的随机整数，如果大于3则闪避成功，否则HP减少特定数值
    """
```

接下来是一段文档字符串（docstring）。跟创建**Tools**的时候一样，我们需要它来对函数进行描述，以帮助模型或Agent理解使用这个函数的用途。当然，因为我们目前还没引入LLM，所以写文档字符串只是为了跟大家说明有这个事。

### 3.编写函数体，设定Node的业务逻辑

接下来就是完成函数体的内容，写明Node的具体业务逻辑，比如我这里写了一个非常简单的闪避判定与血量扣减逻辑。这里因为我们需要用随机数，所以前面有一个对random库的引入。

```python
import random

def dodge_check_node(state: AppState) -> AppState:
    """检查闪避结果并处理HP变化
    生成1-6的随机整数，如果大于3则闪避成功，否则HP减少特定数值
    """
    dice_roll = random.randint(1, 6)
    
    if dice_roll > 3:
        print("你成功闪避了伤害")
    else:
        state['char_HP'] = state['char_HP'] - dice_roll
        print(f"闪避失败！受到{dice_roll}点伤害")
        
    return state
	
```

关于这段代码，有两个要点需要注意，首先：

```python
state['char_HP'] = state['char_HP'] - dice_roll * 2
```

这里就展示了LangGraph使用State的基本方式，因为state本质是Dict，所以我们可以通过每个键值对的健去访问、修改相关的值。然后：

```python
return state
```

函数结尾的`return state`，确保了Node函数的处理结果（如代码中对HP的值的修改），将保存到我们的state（已被指定为AppState）中。

下面给出第二个Node函数的代码，内容不再赘述：

```python
def display_character_status(state: AppState) -> AppState:
    """展示角色当前的各状态值"""
    
    print("\n" + "="*30)
    print("     角色状态信息")
    print("="*30)
    print(f"职业: {state['char_class']}")
    print(f"等级: {state['char_level']}")
    print(f"生命值: {state['char_HP']}")
    print("="*30)
    
    return state
```

## 三、创建并编译Graph

到此为止，我们就可以创建Graph了。有的朋友可能会问，Edge怎么没讲？因为如前所述，只有一些特殊的Edge才是函数，需要专门写代码，比如Conditional Edge，这个我后面再讲，对于普通的Edge，直接add就行，大家一看就懂。

创建Graph的代码非常简单，主要包括4个步骤，我先贴出整体的代码，然后逐项给大家解释：

```python
from langgraph.graph import StateGraph,START,EDND


graph = StateGraph(AppState)

graph.add_node("dodge_check",dodge_check_node)
graph.add_node("character_status",display_character_status)

graph.add_edge(START, "dodge_check")
graph.add_edge( "dodge_check", "character_status")
graph.add_edge("character_status",END)

app = graph.compile()

```

### 1.激活StateGraph

```python
from langgraph.graph import StateGraph
graph = StateGraph(AppState)
```

首先，从LangGraph官方库中导入`StateGraph`，我们需要向`StateGraph`传入我们前面定义好的state，从而实例化一个Graph。

### 2.增加节点

```python
graph.add_node("dodge_check",dodge_check_node)
graph.add_node("character_status",display_character_status)
```

然后，一个一个的把前面写好的Node函数，以Node的形式加入到Graph里。方法就是使用`add_node`函数，传入两个参数，首先是字符串形式的节点名称，这个也是任意取的，然后就是前面已经定义好的节点对应的函数，代表了节点的动作。

### 3.连接节点

```python
from langgraph.graph import START,EDND

graph.add_edge(START, "dodge_check")
graph.add_edge( "dodge_check", "character_status")
graph.add_edge("character_status",END)
```

然后，就是把这些节点，根据刚才取好的名字都连接起来。

这里可以引入现成`START`和`END`节点，也可以使用`set_entry_point()`、`set_fiish_point()`这种函数来指定起始点，两种方式都可以

```python
graph.set_entry_point("dodge_check")
graph.add_edge( "dodge_check", "character_status")
graph.set_finish_point("character_status")
```

### 4.编译Graph

```python
app = graph.compile()
```

这是简单但最重要的一步，我们需要使用`cpmpile`方法来编译我们的Graph，这样我们才能使用它。

### Graph的可视化

在使用Graph之前，我们可以将它可视化一下，来看看我们的成果。我直接提供一段代码供大家使用：

```python
from IPython.display import Image,display
display(Image(app.get_graph().draw_mermaid_png()))
```

![[2.png]]

可以看到，从形式上来看，我们的Graph已经成功地构建起来了。当然，这个Graph非常简陋，除了首尾之外，只有两个直接连接的Node。但只要掌握了方法原理，复杂的Graph我们也能轻松拿捏。

下面我们来跑一下这个Graph。

## 使用Graph

使用Graph的方法也非常简单，就是调用`invoke`，并按state的定义向其传入参数字典即可。

先回顾下我们定义的state：

```python
class AppState(TypedDict):
	char_class: str
	char_level:int
	char_HP:int
```

因此，我们需要向`invoke`传入一个字典，它的信息代表着一个等级为5，血量为10的法师。

```python
result = app.invoke({"char_class":"法师","char_level":5,"char_HP":10})
```

可以看到，我们的Graph已经成功运行了~~ 

![[System/Attachment/公众号文章/20251023/3.png]]

其中，第一部分关于受到伤害的信息，来自于第一个Node，第二部分的角色状态信息，则来自第二个Node。可以看到，我们一开始设定的state中的10点HP，已经因为在第一个Node中受到了2点伤害，被更新为了8点。