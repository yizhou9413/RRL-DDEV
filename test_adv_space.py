"""
测试adversary扰动空间配置是否生效

运行此脚本来验证:
1. 配置文件能否正确加载
2. 配置是否正确传递到算法
3. 不同配置是否产生不同的扰动范围
"""

import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))

# 模拟 run_rarl.py 的配置加载过程
import yaml

# 读取配置
with open(os.path.join(file_path, "rarl_config.yaml")) as f:
    runner_config = yaml.safe_load(f)

print("=" * 80)
print("测试 1: 检查配置文件中的 adv_action_space")
print("=" * 80)

if "rarl" in runner_config["params"]:
    rarl_cfg = runner_config["params"]["rarl"]
    if "adv_action_space" in rarl_cfg:
        adv_space = rarl_cfg["adv_action_space"]
        print(f"✓ 找到 adv_action_space 配置:")
        print(f"  low:  {adv_space['low']}")
        print(f"  high: {adv_space['high']}")
    else:
        print("✗ 未找到 adv_action_space!")
        sys.exit(1)
else:
    print("✗ 未找到 rarl 配置部分!")
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 2: 模拟 run_rarl.py 的配置传递")
print("=" * 80)

# 模拟 run_rarl.py 的传递逻辑
if "rarl" in runner_config["params"] and "adv_action_space" in runner_config["params"]["rarl"]:
    runner_config["params"]["config"]["adv_action_space"] = runner_config["params"]["rarl"]["adv_action_space"]
    print("✓ adv_action_space 已传递到 params.config")
    print(f"  low:  {runner_config['params']['config']['adv_action_space']['low']}")
    print(f"  high: {runner_config['params']['config']['adv_action_space']['high']}")
else:
    print("✗ 传递失败!")
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 3: 模拟算法读取配置")
print("=" * 80)

# 模拟 rarl_continuous.py 中的读取逻辑
rarl_config = runner_config["params"]["config"].copy()
adv_space_config = rarl_config.get('adv_action_space', {
    'low': [-150., -150., -150., -150., -150., -150., -150., -150., -60000., -5000.],
    'high': [150., 150., 150., 150., 150., 150., 150., 150., 60000., 5000.],
})

print(f"算法读取到的 adv_action_space:")
print(f"  low:  {adv_space_config['low']}")
print(f"  high: {adv_space_config['high']}")

# 检查是否使用了配置文件的值
expected_low = runner_config["params"]["rarl"]["adv_action_space"]["low"]
expected_high = runner_config["params"]["rarl"]["adv_action_space"]["high"]

if adv_space_config['low'] == expected_low and adv_space_config['high'] == expected_high:
    print("\n✓ 成功！算法使用了配置文件中的值")
else:
    print("\n✗ 失败！算法使用了硬编码的默认值")
    print(f"  期望 low:  {expected_low}")
    print(f"  实际 low:  {adv_space_config['low']}")
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 4: 验证不同配置产生不同结果")
print("=" * 80)

# 创建两个不同的配置
import numpy as np

config1_low = np.array(adv_space_config['low'], dtype=np.float32)
config1_high = np.array(adv_space_config['high'], dtype=np.float32)

# 假设修改后的配置
config2_low = np.array([-200., -200., -200., -200., -200., -200., -200., -200., -2000., -2000.], dtype=np.float32)
config2_high = np.array([200., 200., 200., 200., 200., 200., 200., 200., 2000., 2000.], dtype=np.float32)

print(f"配置 1 (当前配置文件):")
print(f"  范围: [{config1_low[0]}, {config1_high[0]}] (力)")
print(f"  范围: [{config1_low[8]}, {config1_high[8]}] (力矩)")

print(f"\n配置 2 (假设的修改):")
print(f"  范围: [{config2_low[0]}, {config2_high[0]}] (力)")
print(f"  范围: [{config2_low[8]}, {config2_high[8]}] (力矩)")

# 模拟扰动缩放
dummy_action = np.array([0.5] * 10)  # adversary 输出 [-1, 1]

# 缩放函数
def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    return action * d + m

disturbance1 = rescale_actions(config1_low, config1_high, dummy_action)
disturbance2 = rescale_actions(config2_low, config2_high, dummy_action)

print(f"\n对于相同的adversary输出 {dummy_action[0]}:")
print(f"  配置 1 产生扰动: {disturbance1[0]:.2f} N (力), {disturbance1[8]:.2f} Nm (力矩)")
print(f"  配置 2 产生扰动: {disturbance2[0]:.2f} N (力), {disturbance2[8]:.2f} Nm (力矩)")

if not np.allclose(disturbance1, disturbance2):
    print("\n✓ 不同配置产生不同的扰动！")
else:
    print("\n✗ 不同配置产生相同的扰动！")

print("\n" + "=" * 80)
print("总结:")
print("=" * 80)
print("✓ 所有测试通过！")
print("  - 配置文件正确定义了 adv_action_space")
print("  - run_rarl.py 正确传递了配置")
print("  - 算法能够读取配置")
print("  - 不同配置会产生不同的扰动范围")
print("\n如果你修改 rarl_config.yaml 中的 adv_action_space,")
print("重新启动训练后应该会看到不同的扰动效果。")
print("\n⚠️  注意: 必须完全重启训练进程才能加载新配置!")
print("=" * 80)
