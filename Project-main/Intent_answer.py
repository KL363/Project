#=====================这是接受用户信息，获取回答的主函数===================
import os
from dotenv import load_dotenv
from Intent_Recognition.code.intent_classifier import IntentClassifier
from RAGlibrary import RAG_psychology, RAG_fitness, RAG_compus, RAG_paper
# 加载 .env 文件
load_dotenv("Agent.env")

# 验证环境变量是否设置
required_env_vars = [
    "BAILIAN_API_KEY",
    "APP_ID_PSYCHOLOGY",
    "APP_ID_CAMPUS",
    "APP_ID_FITNESS",
    "APP_ID_PAPER"
]

missing_vars = []
for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"❌ 请在.env文件中设置以下环境变量: {', '.join(missing_vars)}")
    exit(1)

print("✅ 所有环境变量配置验证成功")
print(f"📝 使用的智能体应用:")
print(f"   - 心理助手: {os.getenv('APP_ID_PSYCHOLOGY')}")
print(f"   - 健身助手: {os.getenv('APP_ID_FITNESS')}")
print(f"   - 校园助手: {os.getenv('APP_ID_CAMPUS')}")
print(f"   - 论文助手: {os.getenv('APP_ID_PAPER')}")
print()




class InteractiveAgent:
    def __init__(self):
        try:
            # 初始化意图分类器
            self.classifier = IntentClassifier()
            print("✅ 意图分类器初始化成功")

            # 初始化 RAG 智能体（延迟初始化以提高启动速度）
            self.rag_agents = {}
            self.agent_classes = {
                "心理助手": RAG_psychology,
                "健身饮食助手": RAG_fitness,
                "校园知识问答": RAG_compus,
                "论文助手": RAG_paper
            }
            print("✅ RAG 智能体类加载成功")

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def get_rag_agent(self, intent):
        """延迟初始化RAG智能体"""
        if intent not in self.rag_agents:
            if intent in self.agent_classes:
                print(f"🔧 正在初始化 {intent} RAG智能体...")
                try:
                    self.rag_agents[intent] = self.agent_classes[intent]()
                    print(f"✅ {intent} RAG智能体初始化成功")
                except Exception as e:
                    print(f"❌ {intent} RAG智能体初始化失败: {e}")
                    return None
            else:
                return None

        return self.rag_agents.get(intent)

    def check_rag_status(self, intent, rag_agent):
        """检查RAG知识库状态"""
        try:
            doc_count = rag_agent.vector_store.count()
            if doc_count == 0:
                print(f"⚠️ {intent} 知识库中暂无文档")
                return False
            else:
                print(f"📚 {intent} 知识库包含 {doc_count} 个文档片段")
                return True
        except Exception as e:
            print(f"⚠️ 检查 {intent} 知识库状态失败: {e}")
            return False

    def chat(self):
        print("=== 欢迎使用智能助手系统 ===")
        print("💡 本系统使用本地RAG检索增强 + 远程智能体架构")
        print("🔍 支持交叉编码器精确检索和流式回答")
        print("输入你的问题（输入 'exit' 退出，'stream' 切换流式模式）：\n")

        stream_mode = False

        while True:
            user_input = input("🧑 你：")

            if user_input.lower() in ["exit", "quit"]:
                print("👋 再见！")
                break

            if user_input.lower() == "stream":
                stream_mode = not stream_mode
                print(f"💫 流式模式: {'开启' if stream_mode else '关闭'}")
                continue

            # 1. 预测意图
            try:
                result = self.classifier.predict_intent(user_input)
                intent = result["best_intent"]
                confidence = result["confidence"]
                print(f"🤖 识别意图：{intent} (置信度 {confidence:.2f})")
            except Exception as e:
                print(f"❌ 意图识别失败: {e}\n")
                continue

            # 2. 获取对应的RAG智能体
            rag_agent = self.get_rag_agent(intent)
            if rag_agent is None:
                print(f"❌ 暂不支持该意图类型: {intent}\n")
                continue

            # 显示使用的APP ID
            print(f"🔧 使用智能体APP ID: {rag_agent.llm.app_id}")

            # 3. 检查知识库状态
            has_docs = self.check_rag_status(intent, rag_agent)
            if not has_docs:
                print("⚠️ 知识库为空，智能体将基于通用知识回答")

            # 4. 调用 RAG 智能体
            try:
                if stream_mode:
                    print(f"🤖 {intent} 正在思考")
                    print("💬 流式回答：", end="", flush=True)

                    full_response = ""
                    for delta in rag_agent.call_RAG_stream(user_input):
                        print(delta, end="", flush=True)
                        full_response += delta
                    print("\n")  # 换行

                else:
                    print(f"🔍 {intent} 正在检索相关信息...")
                    answer = rag_agent.call_RAG(user_input)
                    print(f"🤖 {intent} 回答：{answer}\n")

            except Exception as e:
                print(f"❌ 调用 {intent} RAG智能体失败：{e}\n")


if __name__ == "__main__":
    try:
        agent = InteractiveAgent()
        agent.chat()  # ✅ 添加这一行！

    except KeyboardInterrupt:
        print("\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
