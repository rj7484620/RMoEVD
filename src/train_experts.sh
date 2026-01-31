#!/usr/bin/env bash
# 文件名：train_experts.sh

# 定义要训练的 CWE 列表
CWES=(
    "CWE-664"
    "CWE-unknown"
    "CWE-707"
    "CWE-399"
    "CWE-264"
    "CWE-691"
    "CWE-284"
    "CWE-682"
    "CWE-189"
    "CWE-other"
    "CWE-703"
    "CWE-254"
)

# 遍历列表，依次运行命令
for cwe in "${CWES[@]}"; do
  echo "开始训练 ${cwe} ..."
  python t2-expert_train.py --cwe "${cwe}" --force_cwe > train.log 2>&1
  echo "完成训练 ${cwe}"
done

echo "所有训练任务已完成。"
