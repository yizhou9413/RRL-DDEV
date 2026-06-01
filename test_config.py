"""测试配置文件是否正确加载"""
import yaml
import os

file_path = os.path.dirname(__file__)

# 读取配置
with open(os.path.join(file_path, "rarl_config.yaml")) as f:
    runner_config = yaml.safe_load(f)

# 打印配置
print("=" * 60)
print("读取到的 rarl_config.yaml 配置:")
print("=" * 60)

# 检查 rarl 部分
if "rarl" in runner_config["params"]:
    rarl_config = runner_config["params"]["rarl"]
    print("\nRARL 配置部分:")
    print(f"  n_pro_itr: {rarl_config.get('n_pro_itr')}")
    print(f"  n_adv_itr: {rarl_config.get('n_adv_itr')}")
    print(f"  adv_reward_scale: {rarl_config.get('adv_reward_scale')}")
    
    if "adv_action_space" in rarl_config:
        adv_space = rarl_config["adv_action_space"]
        print(f"\n  adv_action_space:")
        print(f"    low:  {adv_space['low']}")
        print(f"    high: {adv_space['high']}")
    else:
        print("\n  ⚠️ 警告: 未找到 adv_action_space 配置!")
else:
    print("\n⚠️ 警告: 未找到 rarl 配置部分!")

# 模拟 run_rarl.py 的配置传递
print("\n" + "=" * 60)
print("模拟 run_rarl.py 传递配置到 config:")
print("=" * 60)

# 检查是否会被传递
if "rarl" in runner_config["params"] and "adv_action_space" in runner_config["params"]["rarl"]:
    runner_config["params"]["config"]["adv_action_space"] = runner_config["params"]["rarl"]["adv_action_space"]
    print("\n✓ adv_action_space 已传递到 config:")
    print(f"  low:  {runner_config['params']['config']['adv_action_space']['low']}")
    print(f"  high: {runner_config['params']['config']['adv_action_space']['high']}")
else:
    print("\n✗ adv_action_space 未传递!")

print("\n" + "=" * 60)
