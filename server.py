import json
import uuid
from flask import Flask, request, jsonify, send_file
from openai import OpenAI
import numpy as np
import os

# ===================================================
# 初始化
# ===================================================
DEBUG = True
VIOLATION_MESSAGE = "输入内容有违规行为，请输入对话内容，不能使用动作描述。"#input裁判检测到后，给玩家提示内容

with open("config.json") as f:
    api_key = json.load(f)["api_key"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

config = {
    "client": client,
    "model_id": "anthropic/claude-haiku-4.5",#anthropic/claude-sonnet-4.6 anthropic/claude-haiku-4.5 openai/gpt-4o-mini
    "max_tokens": 500,
    "temperature": 0.9,
    "top_p": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
}

judge_config = {
    "client": client,
    "model_id": "anthropic/claude-haiku-4.5",
    "max_tokens": 200,
    "temperature": 0.3,
    "top_p": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
}

app = Flask(__name__)
sessions = {}


# ===================================================
# RAG 模块
# ===================================================
def get_embedding(text, client):
    """把一段文字转成向量"""#函数内第一个"""字符串用help()函数调用会显示
    response = client.embeddings.create(
        model="perplexity/pplx-embed-v1-4b",
        input=text
    )
    return response.data[0].embedding

def load_knowledge(file_path, client):
    """读取知识库，把每段都向量化，返回(段落列表, 向量列表)"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    chunks = []
    for c in content.split("\n\n"):#按\n\n分割
        if c.strip():
            chunks.append(c.strip())#删除首尾空格换行
    
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):#enumerate是给chunk加上序号
        embedding = get_embedding(chunk, client)
        chunk_embeddings.append(embedding)
    
    return chunks, chunk_embeddings

def search_chunks(query, chunks, chunk_embeddings, client, top_k=3):
    """把问题向量化，找出最相似的段落"""
    query_embedding = get_embedding(query, client)
    
    # 计算余弦相似度
    scores = []
    for i, chunk_vec in enumerate(chunk_embeddings):
        a = np.array(query_embedding)
        b = np.array(chunk_vec)
        score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        scores.append((score, chunks[i]))
    
    # 按分数排序，返回前top_k段
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scores[:top_k]]


# ===================================================
# 启动时读取所有关卡文件
# ===================================================

levels = {}
levels_folder = "levels"
rag_folder = "rag"

for filename in os.listdir(levels_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(levels_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            level_data = json.load(f)
        level_id = level_data["level_id"]
        levels[level_id] = level_data
        print(f"已加载关卡：{level_id}")

rag_cache = {}

for level_id, level_data in levels.items():#.items()是把字典拆成一对对 (key, value) 遍历用
    kf = level_data.get("knowledge_file")
    if kf and kf not in rag_cache:
        path = os.path.join(rag_folder, kf)
        if os.path.exists(path):
            chunks, embeddings = load_knowledge(path, client)
            rag_cache[kf] = (chunks, embeddings)
            print(f"已加载知识库：{kf}")

print(f"共加载 {len(levels)} 个关卡")#len()字典里key的数量


# ===================================================
# api_chat
# ===================================================

def api_chat(config, my_messages, json_schema=None):
    params = {
        "model": config["model_id"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "presence_penalty": config["presence_penalty"],
        "frequency_penalty": config["frequency_penalty"],
        "messages": my_messages,
        "stream": False
    }

    if json_schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema
        }

    # jsonschema报错debug
    try:
        response = config["client"].chat.completions.create(**params)
        reply = response.choices[0].message.content
        # 模型为什么停止生成的原因
        finish_reason = response.choices[0].finish_reason
        if DEBUG:
            print(f"[{config['model_id']}] finish_reason:", finish_reason)
            print(f"[{config['model_id']}] reply:", reply)

        # 记录单次chat的token量
        usage = response.usage
        if usage:
            config["last_usage"] = usage
        return reply
    except Exception as e:
        print(f"[{config['model_id']}] 调用异常:", e)
        return None


# ===================================================
# 接口定义
# ===================================================

@app.route("/start", methods=["POST"])
def start():
    """
    接收：level_id
    返回：session_id、opening_line、max_turns、user_name、assistant_name
    """
    data = request.json
    level_id = data["level_id"]
    # 传统关卡数据+ai关卡总结数据
    player_state = data.get("player_state", "")

    if level_id not in levels:
        return jsonify({"error": f"关卡 {level_id} 不存在"}), 400

    level = levels[level_id]
    assistant_name = level["assistant_name"]
    user_name = level["user_name"]

    # 替换占位符，存进session 运行时处理替换字符串replace()
    system_prompt = level["system_prompt"].replace("{assistant_name}", assistant_name).replace("{user_name}", user_name)
    judge_system  = level["judge_system"].replace("{assistant_name}", assistant_name)

    # 拼接
    if player_state:
        system_prompt += f"\n\n【你们之前发生的事】\n{player_state}"

    # 建立初始messages
    my_messages = []
    my_messages.append({"role": "system", "content": system_prompt})
    my_messages.append({"role": "assistant", "content": level["opening_line"]})

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": my_messages,
        "turn": 0,
        "judge_system": judge_system,
        "npc_schema": level["npc_schema"],
        "judge_schema": level["judge_schema"],
        "max_turns": level["max_turns"],
        "user_name": user_name,
        "assistant_name": assistant_name,
        "knowledge_file": level.get("knowledge_file"),
        "system_prompt_base": system_prompt,
        "input_judge_system": level.get("input_judge_system"),
        "input_judge_schema": level.get("input_judge_schema")
    }

    return jsonify({
        "session_id": session_id,
        "opening": level["opening_line"],
        "opening_emotion": level["opening_emotion"],
        "goal_description": level["goal_description"],
        "max_turns": level["max_turns"],
        "user_name": user_name,
        "assistant_name": assistant_name
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    接收：session_id、user_input
    返回：npc_reply、convinced、reason、turn、max_turns、game_over、usage
    """
    data = request.json
    session_id = data["session_id"]
    user_input = data["user_input"]

    if session_id not in sessions:
        return jsonify({"error": "session不存在，请重新开始游戏"}), 400
    
    session = sessions[session_id]
    my_messages = session["messages"]
    session["turn"] += 1
    current_turn = session["turn"]

    # input裁判
    if session.get("input_judge_system") and session.get("input_judge_schema"):
        input_judge_messages = [
            {"role": "system", "content": session["input_judge_system"]},
            {"role": "user", "content": f"玩家输入：{user_input}"}
        ]
        if DEBUG:
            print("[阶段] input_judge")
        input_judge_raw = api_chat(judge_config, input_judge_messages, json_schema=session["input_judge_schema"])
        #检测json_schema
        if input_judge_raw is None:
            return jsonify({"error": "裁判调用失败"}), 500
        
        input_judge_result = json.loads(input_judge_raw)
        if input_judge_result["violation"]:
            return jsonify({
                "input_violation": True,
                "violation_message": VIOLATION_MESSAGE,
                "turn": current_turn,
                "max_turns": session["max_turns"],
                "game_over": current_turn >= session["max_turns"],
            })

    kf = session.get("knowledge_file")
    if kf and kf in rag_cache:
        relevant = search_chunks(user_input, rag_cache[kf][0], rag_cache[kf][1], client)
        rag_context = "\n---\n".join(relevant)
        my_messages[0]["content"] = session["system_prompt_base"] + f"\n\n【相关背景】\n{rag_context}"

    # 玩家输入加入messages，调用NPC
    my_messages.append({"role": "user", "content": user_input})
    if DEBUG:
        print("[阶段] NPC")
        print(f"user_input:{user_input}")
    npc_reply_raw = api_chat(config, my_messages, json_schema=session["npc_schema"])
    
    #检测json_schema
    if npc_reply_raw is None:
        return jsonify({"error": "裁判调用失败"}), 500
    
    npc_result = json.loads(npc_reply_raw)
    npc_reply = npc_result["dialogue"]
    npc_emotion = npc_result["emotion"] 

    # 把NPC的对话文字存入messages
    my_messages.append({"role": "assistant", "content": npc_reply})

    # 记录NPC的token，在裁判调用前抢救出来
    npc_usage = config.get("last_usage")

    # 裁判判定 只取role是user和assistant的部分
    dialogue_only = [m for m in my_messages if m["role"] != "system"]
    judge_messages = [
        {"role": "system", "content": session["judge_system"]},
        {"role": "user", "content": f"以下是完整对话记录：{dialogue_only}，请判断"}
    ]
    
    if DEBUG:
        print("[阶段] judge")
    
    judge_reply_raw = api_chat(judge_config, judge_messages, json_schema=session["judge_schema"])
    
    #检测json_schema
    if judge_reply_raw is None:
        return jsonify({"error": "裁判调用失败"}), 500
    
    judge_result = json.loads(judge_reply_raw)

    # 记录裁判的token
    judge_usage = judge_config.get("last_usage")

    # 游戏结束判定
    game_over = judge_result["convinced"] or current_turn >= session["max_turns"]
    summary = ""
    if game_over:
        # 聊天记录存档
        summary_messages = [
            {"role": "system", "content": "请用2-3句话概括以下对话中的关键信息，作为存档摘要。只输出纯文本，不要使用JSON、代码块或任何Markdown格式。"},
            {"role": "user", "content": str(my_messages)}
        ]
        if DEBUG:
            print("[阶段] summary")
        summary = api_chat(config, summary_messages)
        del sessions[session_id]

    # 合计两次token
    usage_data = {}
    if npc_usage and judge_usage:
        usage_data = {
            "prompt_tokens": npc_usage.prompt_tokens + judge_usage.prompt_tokens,
            "completion_tokens": npc_usage.completion_tokens + judge_usage.completion_tokens,
        }

    return jsonify({
        "npc_reply": npc_reply,
        "npc_emotion": npc_emotion,
        "turn": current_turn,
        "max_turns": session["max_turns"],
        "convinced": judge_result["convinced"],
        "reason": judge_result.get("reason", ""),#从字典里reason这个key，如果不存在返回默认值""
        "game_over": game_over,
        "usage": usage_data,
        "user_name": session["user_name"],
        "assistant_name": session["assistant_name"],
        "summary": summary
    })

# web测试用
@app.route("/")
def index():
    return send_file("web+.html")

# ===================================================
# 第五块：启动服务器
# ===================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8765, use_reloader=False)
