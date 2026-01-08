"""
测试 DeepSeek 是否支持使用 PydanticOutputParser 进行结构化输出
使用 PydanticOutputParser 方法（不依赖 with_structured_output）
"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Optional

# Load environment variables
load_dotenv()

# 1. 定义目标结构
class RewriteOutput(BaseModel):
    """查询重写输出模型"""
    rewritten_query: str = Field(description="重写后的查询")
    needs_search: bool = Field(description="是否需要搜索")


class MultiQueryOutput(BaseModel):
    """多查询输出模型"""
    queries: List[str] = Field(description="查询列表")
    count: int = Field(description="查询数量", ge=1, le=5)


class ComplexOutput(BaseModel):
    """复杂输出模型"""
    original_query: str = Field(description="原始查询")
    rewritten_query: str = Field(description="重写后的查询")
    keywords: List[str] = Field(description="关键词列表")
    query_type: Optional[str] = Field(description="查询类型：factual, analytical, or conversational", default="factual")


def test_basic_rewrite_output():
    """测试基本的查询重写输出"""
    print("=" * 60)
    print("测试 1: 基本查询重写输出 (RewriteOutput)")
    print("=" * 60)
    
    try:
        # 初始化 DeepSeek LLM
        llm = ChatOpenAI(
            temperature=0,
            model_name="deepseek-chat",
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com",
            max_tokens=2000
        )
        
        # 2. 初始化解析器
        parser = PydanticOutputParser(pydantic_object=RewriteOutput)
        
        # 3. 获取格式指令 (关键步骤)
        # 这会自动生成一段很长的 String，教模型怎么写 JSON
        format_instructions = parser.get_format_instructions()
        
        # 4. 构建 Prompt，必须包含 {format_instructions}
        prompt = PromptTemplate(
            template="""你是一个查询重写助手。将用户的查询改写为更具体、详细的问题。
            
{format_instructions}

用户输入: {question}

请按照上述 JSON 格式输出结果。""",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )
        
        # 5. 组合 Chain
        # 注意：这里不用 with_structured_output，而是用普通的管道
        chain = prompt | llm | parser
        
        # 6. 调用
        test_question = "它多少钱？"
        print(f"\n输入查询: {test_question}")
        print("正在调用 API...")
        
        result = chain.invoke({"question": test_question})
        
        print(f"\n✅ 成功！结构化输出结果:")
        print(f"  - 类型: {type(result)}")
        print(f"  - Rewritten Query: {result.rewritten_query}")
        print(f"  - Needs Search: {result.needs_search}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 失败！错误信息:")
        print(f"  - 错误类型: {type(e).__name__}")
        print(f"  - 错误详情: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_query_output():
    """测试多查询输出"""
    print("\n" + "=" * 60)
    print("测试 2: 多查询输出 (MultiQueryOutput)")
    print("=" * 60)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model_name="deepseek-chat",
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com",
            max_tokens=2000
        )
        
        parser = PydanticOutputParser(pydantic_object=MultiQueryOutput)
        format_instructions = parser.get_format_instructions()
        
        prompt = PromptTemplate(
            template="""为以下查询生成3个不同的变体查询，用于提高检索召回率。
            
{format_instructions}

原始查询: {question}

请按照上述 JSON 格式输出结果。""",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )
        
        chain = prompt | llm | parser
        
        test_question = "如何学习Python？"
        print(f"\n输入查询: {test_question}")
        print("正在调用 API...")
        
        result = chain.invoke({"question": test_question})
        
        print(f"\n✅ 成功！结构化输出结果:")
        print(f"  - 类型: {type(result)}")
        print(f"  - 查询数量: {result.count}")
        print(f"  - 查询列表:")
        for i, q in enumerate(result.queries, 1):
            print(f"    {i}. {q}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 失败！错误信息:")
        print(f"  - 错误类型: {type(e).__name__}")
        print(f"  - 错误详情: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_complex_output():
    """测试复杂输出"""
    print("\n" + "=" * 60)
    print("测试 3: 复杂输出 (ComplexOutput)")
    print("=" * 60)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model_name="deepseek-chat",
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com",
            max_tokens=2000
        )
        
        parser = PydanticOutputParser(pydantic_object=ComplexOutput)
        format_instructions = parser.get_format_instructions()
        
        prompt = PromptTemplate(
            template="""分析以下查询，提取信息并分类。
            
{format_instructions}

原始查询: {question}

请按照上述 JSON 格式输出结果，包括：
1. 改写后的查询
2. 关键词列表（至少3个）
3. 查询类型（factual/analytical/conversational）""",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )
        
        chain = prompt | llm | parser
        
        test_question = "Python中装饰器的原理是什么？"
        print(f"\n输入查询: {test_question}")
        print("正在调用 API...")
        
        result = chain.invoke({"question": test_question})
        
        print(f"\n✅ 成功！结构化输出结果:")
        print(f"  - 类型: {type(result)}")
        print(f"  - 原始查询: {result.original_query}")
        print(f"  - 改写查询: {result.rewritten_query}")
        print(f"  - 关键词: {', '.join(result.keywords)}")
        print(f"  - 查询类型: {result.query_type}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 失败！错误信息:")
        print(f"  - 错误类型: {type(e).__name__}")
        print(f"  - 错误详情: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_api_configuration():
    """检查 API 配置"""
    print("=" * 60)
    print("检查 DeepSeek API 配置")
    print("=" * 60)
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ 错误: 未找到 DEEPSEEK_API_KEY 环境变量")
        print("   请在 .env 文件中设置: DEEPSEEK_API_KEY=your_api_key")
        return False
    
    print(f"✅ API Key 已配置 (长度: {len(api_key)} 字符)")
    print(f"   Base URL: https://api.deepseek.com")
    print(f"   Model: deepseek-chat")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DeepSeek PydanticOutputParser 结构化输出测试")
    print("=" * 60)
    print()
    
    # 检查配置
    if not check_api_configuration():
        return
    
    print("\n开始测试...\n")
    
    # 运行测试
    results = {
        "基本查询重写": test_basic_rewrite_output(),
        "多查询输出": test_multi_query_output(),
        "复杂输出": test_complex_output()
    }
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    
    if all_passed:
        print("✅ DeepSeek 支持使用 PydanticOutputParser 进行结构化输出")
        print("   该方法可行，可以:")
        print("   1. 定义 Pydantic 模型来指定输出格式")
        print("   2. 使用 PydanticOutputParser 解析 LLM 输出")
        print("   3. 通过 format_instructions 引导模型输出 JSON 格式")
    else:
        print("⚠️  部分测试失败")
        print("   可能的原因:")
        print("   1. DeepSeek 模型输出的 JSON 格式不符合预期")
        print("   2. 需要调整 prompt 来更明确地要求 JSON 输出")
        print("   3. 可能需要添加错误重试机制")
    
    print()


if __name__ == "__main__":
    main()