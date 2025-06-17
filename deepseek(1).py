import json
import random
from transformers import pipeline

# 1. 环境模块初始化 - 解析城市数据
def load_city_data():
    """加载城市布局和统计数据"""
    # 解析建筑统计数据
    building_stats = {}
    with open('output.txt') as f:
        for line in f.read().splitlines():
            key, value = line.split('=')
            building_stats[key.strip()] = int(value.strip())
    
    # 解析城市布局
    with open('layout.json') as f:
        city_layout = json.load(f)
    
    return building_stats, city_layout

# 2. 个性化模块 - 生成Agent
def generate_agents():
    """创建10个不同属性的Agent"""
    agents = []
    profiles = [
        # 格式: [年龄, 职业, 性格, 年收入(万), 家庭状况]
        [35, "建筑师", "理想主义", 50, "单身"],
        [62, "退休教师", "保守", 15, "与配偶同住"],
        [28, "程序员", "理性", 45, "合租"],
        [45, "企业家", "激进", 200, "四口之家"],
        [22, "大学生", "环保主义", 5, "学生宿舍"],
        [50, "医生", "务实", 80, "三口之家"],
        [31, "艺术家", "浪漫主义", 30, "独居"],
        [40, "政府职员", "官僚主义", 25, "三口之家"],
        [55, "店主", "传统", 20, "与老人同住"],
        [27, "社工", "人道主义", 12, "合租"]
    ]

    for i, profile in enumerate(profiles):
        agents.append({
            "id": f"agent_{i+1}",
            "age": profile[0],
            "occupation": profile[1],
            "personality": profile[2],
            "income": profile[3],
            "family": profile[4]
        })
    return agents

# 3. 本地DeepSeek模型初始化
def init_deepseek_model():
    """加载本地DeepSeek-R1模型"""
    # 注意：实际路径需替换为你的模型保存位置
    model_path = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
    
    return pipeline(
        'text-generation',
        model=model_path,
        device=0,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 4. 建议提案模块
def generate_proposal(agent, stats, layout, model):
    """生成城市规划建议"""
    prompt = f"""
    你是一位{agent['age']}岁的{agent['occupation']}，性格{agent['personality']}。
    当前城市建筑分布：
    - 摩天大楼：{stats['skyscraper_num']}座
    - 地铁站：{stats['metro_num']}个
    - 商业中心：{stats['mall_num']}个
    - 普通建筑：{stats['building_num']}栋
    
    建筑位置示例：
    {random.sample(layout['buildings'], 3)}
    
    请基于你的身份和需求，提出一个具体的城市规划改进建议（只能调整现有建筑参数），建议内容需包含：
    1. 需要修改的建筑名称（必须从layout.json中存在的建筑中选择）
    2. 新的位置坐标(x, z)（必须在合理范围内）
    3. 修改理由
    
    输出必须是严格的JSON格式：
    {{
      "building": "建筑全名",
      "new_x": 新x坐标(浮点数),
      "new_z": 新z坐标(浮点数),
      "reason": "详细理由"
    }}
    不要包含任何额外文本！
    """
    
    # 调用本地模型生成建议
    response = model(
        prompt,
        return_full_text=False
    )[0]['generated_text'].strip()
    
    # 提取JSON部分
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        json_str = response[start_idx:end_idx]
        return json.loads(json_str)
    except:
        # 如果解析失败，使用默认提案
        return {
            "building": "combined_metro_1",
            "new_x": 45.8,
            "new_z": 75.2,
            "reason": "优化交通网络，连接主要商业区"
        }

# 5. 投票决策模块
def vote_on_proposal(proposal, agents, layout, model):
    """Agent投票表决提案"""
    votes = {"赞成": 0, "反对": 0}
    reasons = []
    
    for agent in agents:
        prompt = f"""
        {agent['occupation']}先生/女士({agent['age']}岁)，
        有人建议将建筑【{proposal['building']}】移动到新位置({proposal['new_x']}, {proposal['new_z']})。
        理由：{proposal['reason']}
        
        请基于您的身份({agent['personality']}性格, {agent['family']}家庭, 年收入{agent['income']}万)
        投票并说明原因：
        1. 这个改动对您的生活有何影响？
        2. 是否符合您的价值观？
        
        输出必须是严格的JSON格式：
        {{
          "vote": "赞成/反对",
          "reason": "投票理由"
        }}
        不要包含任何额外文本！
        """
        
        # 调用本地模型
        response = model(
            prompt,
            return_full_text=False
        )[0]['generated_text'].strip()
        
        # 提取JSON部分
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            vote_data = json.loads(response[start_idx:end_idx])
            
            votes[vote_data["vote"]] += 1
            reasons.append({
                "voter": f"{agent['occupation']}({agent['personality']})",
                "vote": vote_data["vote"],
                "reason": vote_data["reason"]
            })
        except:
            # 如果解析失败，随机投票
            vote = random.choice(["赞成", "反对"])
            votes[vote] += 1
            reasons.append({
                "voter": f"{agent['occupation']}({agent['personality']})",
                "vote": vote,
                "reason": "系统错误，随机投票"
            })
    
    return votes, reasons

# 6. 参数修改引擎
def update_city_layout(proposal, layout):
    """更新城市布局文件"""
    # 查找目标建筑
    target_building = None
    for building in layout["buildings"]:
        if building["name"] == proposal["building"]:
            target_building = building
            break
    
    if not target_building:
        raise ValueError(f"建筑 {proposal['building']} 不存在")
    
    # 计算偏移量
    dx = proposal["new_x"] - target_building["center"]["x"]
    dz = proposal["new_z"] - target_building["center"]["z"]
    
    # 更新所有坐标
    target_building["center"]["x"] = proposal["new_x"]
    target_building["center"]["z"] = proposal["new_z"]
    target_building["x_min"] += dx
    target_building["x_max"] += dx
    target_building["z_min"] += dz
    target_building["z_max"] += dz
    
    # 保存更新
    with open('layout_updated.json', 'w') as f:
        json.dump(layout, f, indent=2)
    
    return layout

# 7. 主程序
def main():
    print("="*50)
    print("城市规划Agent系统启动")
    print("="*50)
    
    # 初始化系统
    print("\n[阶段1] 加载城市数据...")
    building_stats, city_layout = load_city_data()
    print(f"  建筑统计: {building_stats}")
    print(f"  建筑数量: {len(city_layout['buildings'])}")
    
    print("\n[阶段2] 创建居民Agent...")
    agents = generate_agents()
    print(f"  已创建 {len(agents)} 位不同背景的居民Agent")
    for i, agent in enumerate(agents):
        print(f"  Agent{i+1}: {agent['occupation']}({agent['age']}岁, {agent['personality']})")
    
    print("\n[阶段3] 加载DeepSeek-R1模型...")
    deepseek_model = init_deepseek_model()
    print("  模型加载完成!")
    
    # 随机选择提案Agent
    proposer = random.choice(agents)
    print("\n[阶段4] 生成城市规划提案...")
    print(f"  提案者: {proposer['occupation']}({proposer['age']}岁, {proposer['personality']})")
    
    # 生成提案
    proposal = generate_proposal(proposer, building_stats, city_layout, deepseek_model)
    print(f"  提案内容: 移动建筑 {proposal['building']} 到 ({proposal['new_x']}, {proposal['new_z']})")
    print(f"  理由: {proposal['reason']}")
    
    # 投票表决
    print("\n[阶段5] 居民投票表决...")
    votes, reasons = vote_on_proposal(proposal, agents, city_layout, deepseek_model)
    print(f"  投票结果: 赞成 {votes['赞成']}票, 反对 {votes['反对']}票")
    
    # 打印投票详情
    print("\n投票详情:")
    for i, reason in enumerate(reasons):
        print(f"  {i+1}. {reason['voter']}: [{reason['vote']}] {reason['reason']}")
    
    # 执行决策
    if votes["赞成"] > votes["反对"]:
        print("\n[阶段6] 提案通过! 更新城市布局...")
        updated_layout = update_city_layout(proposal, city_layout)
        print("  城市布局已保存至: layout_updated.json")
        
        # 显示变更
        for b in updated_layout["buildings"]:
            if b["name"] == proposal["building"]:
                print(f"  建筑 {b['name']} 新位置: x={b['center']['x']}, z={b['center']['z']}")
                break
    else:
        print("\n[阶段6] 提案被否决，城市布局保持不变")

    print("\n城市规划流程完成!")

if __name__ == "__main__":
    main()