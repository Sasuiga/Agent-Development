-大家好，作为学习使用LangGraph搭建AI Agent系统的基础准备，前两期我们初步了解了如何利用LangChain来调用大语言模型，如何标准化地向模型传递信息，以及如何构造Prompt模板。今天，我将继续带领大家学习一个与Agents相关的重要前置概念，**Tools（工具）**。

## 什么是工具（Tools）

正如我在以前文章中多次提到过的，AI Agents与传统AI系统的一个重要区别，就在于AI Agents能够根据任务需要，灵活地连接并使用外部系统的功能，从而极大地拓展了大语言模型的能力边界。而Tools的诞生，正是源于这种让AI Agents与API、数据库、文件系统等外部系统直接进行交互的需求。

为了实现前述目的，LangChain允许我们把可调用的函数（callable function）以及对应的输入模式（input schema）封装成为“Tools”。而通过使用“Tools”，大模型能够按照已经事先定义好的输入或输出格式，去整理自己从用户处接收，或待向外部系统输出的信息，从而更好地满足外部系统的接入要求。

某种意义上，Tools跟Message、Template一样，本质上都是根据不同的目的，对向模型输入的信息进行标准化、规范化的方式。

## 如何创建Tools

在Langchain中，创建一个工具的最简单方法包括3个主要步骤：
### 1.使用`@tool`装饰器，并定义函数

```python
from langchain_core.tools import tool 

@tool
def search_database(query: str, limit: int = 10) -> str:
```

- langchain_core 库中的` @tool` 装饰器可以将普通 Python 函数转换为 LangChain 工具（tool），使其可以被 Model或者Agent使用。

- 在定义函数时，**必须要写明类型提示（type hints）**。

类型提示是 Python 中的一种语法特性，用于在代码中声明变量、函数参数和返回值的预期类型，但它不会影响程序的实际运行，其主要目的是提升代码可读性和可维护性。

而在AI应用开发中，类型提示实际定义了工具的输入模式（input schema），使模型能够更好地知道如何使用你定义的工具。

以上述代码中的类型提示为例：

- `query: str`：告诉调用者，函数`search_database`的`query` 参数应是一个字符串类型，例如 "apple"、"John Doe"。
- `limit: int = 10`：意味着`limit`参数是一个整数类型，且其默认值为 10。
- `-> str`：表示该函数返回的是字符串，例如 "Found 5 results for 'apple'"。

### 2.利用函数的文档字符串进行工具描述

```python
@tool
def search_database(query: str, limit: int = 10) -> str:
	"""Search the customer database for records matching the query. 
	Args: query: Search terms to look for. 
	limit: Maximum number of results to return. """
```

我们需要使用文档字符串（即上述代码中被“”“包裹起来的部分），对工具的用途，参数的含义进行清晰、简明的描述，以帮助模型或Agent理解使用这个工具的目的。

### 3.编写函数体，设定工具的业务逻辑

接下来就是完成函数体的内容，写明工具的具体业务逻辑。在LangGraph官方文档中，只简单模拟了一个查询的输出，这里我们贴出一个比较简单但完整的数据库搜索逻辑供大家参考：

```python
from langchain_core.tools import tool
import json

@tool
def search_database(query: str, limit: int = 10) -> str:
	"""Search the customer database for records matching the query. 
	Args: query: Search terms to look for. 
	limit: Maximum number of results to return. """
    # 模拟数据库
    mock_db = [
        {"name": "John Doe", "email": "john@example.com", "country": "USA"},
        {"name": "Johanna Smith", "email": "johanna@example.co.uk", "country": "UK"},
        {"name": "George Johnson", "email": "george.j@mockmail.org", "country": "Canada"},
        # 可扩展客户需求数据
    ]
    
    # 简单匹配逻辑（全字段模糊匹配）
    results = []
    for record in mock_db:
        if query and query.lower() in str(record.values()).lower():
            results.append(record)
            if len(results) >= limit: break

    return json.dumps(results, indent=2) or "No matching records found"
```

上述代码定义了一个简单的数据库搜索工具，这个工具允许用户通过`query`，对代码内部定义好的数据库记录`mock_db`的所有字段（`name`、`email`、`country`)进行搜索，并支持限制返回结果数量（默认10条）。同时，返回内容为格式化后的JSON字符串或"No matching records found"。

在更实际的应用中，`mock_db`将被真是数据库连接所替代，以实现为模型提供查询特定数据库的能力的目的。


## Tools的自定义设定

###  自定义参数模式(args_schema)

正如文章开头所述，Tools的诞生是为了规范、标准化大模型与外部系统的交互，以使大模型能够更精准地访问、使用外部系统的功能。因此，为了约束外部输入数据的结构，让Tools封装的参数输入模式更为精准，LangChain允许我们在`@tool`后面的*括号中设定自定义的参数模式。*

```python
@tool(args_schema=CustomInput)
```

而关于`CustomInput`的具体设定方式，又有两种方案。
#### 1.Pydantic model

第一种方案是使用`Pydantic`先来创建数据模型，对Tools所要求的输入输出参数进行详细定义，以便对大语言模型在调用Tools时提供的数据的类型、格式进行检验。

官方示例代码解析如下：

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )
```

上述代码首先利用Pydantic的`BaseModel`定义了一个名为`WeatherInput`的数据模型。这个数据模型中有`location`、`units`、`include_forecast`三个字段。主要字段的类型及元数据定义如下：
- `location: str = Field(description="City name or coordinates")`
	- 设定字段location的值的类型为字符串（str），同时在字段元数据（Field）中提供了一个自然语言的描述（description）。由于元数据中没有指定default项，所以location字段是一个必填项。
- `units: Literal["celsius", "fahrenheit"] = Field(default="celsius",description="Temperature unit preference")`
	- 通过`Literal`将units字段的值的字面量严格设定为celsius或fahrenheit，如果传入其他值，则程序在运行时将抛出错误。同时，在字段元数据中设定默认值为celsius
- `include_forecast`字段设定方式同理，不再赘述

以下是向Tool导入自定义参数模式`WeatherInput`的代码，其主要内容与前文创建Tools相近，此处不再赘述。

```python
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

#### 2.JSON Schema

第二种方案是使用`JSON Schema`来描述数据结构必备的字段和类型约束。

```python
weather_schema = { 
	"type": "object", 
	"properties": { 
		"location": {"type": "string"}, 
		"units": {"type": "string"}, 
		"include_forecast": {"type": "boolean"} 
	}, 
	"required": ["location", "units", "include_forecast"] 
}
```

其中：
- `"type": "object"`表示拟输入的数据结构必须是一个JSON对象。所谓的JSON对象，其实就是形如`{key:value}` 的键值对结构的集合，
- `"properties"`层，描述了这个JSON对象中允许包含的字段及其规则
- `"required"`层，描述了这个JSON对象中必须包含的字段。

### 自定义Tool名称

默认情况下，Tool的名称就是函数的名称，你可以通过在`@tool`后面*加括号，并在其中使用字符串*来自定义工具的名称。

```python
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"
```

### 自定义Tool简介

同样的，你也可以通过在`@tool`后面*加括号，并定义`description`的值*来自定义工具的介绍，以便更清晰地告诉大模型在什么情况下使用这个工具。

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

