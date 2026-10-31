## 本机直接运行
从项目根目录打开PowerShell：

```powershell
./.python/python.exe FINAL_SUBMISSION/run.py
```

也可以运行FINAL_SUBMISSION/run.cmd。该入口验证文件哈希、设置四线程，启动官方预测器和评分器两个独立进程，读取原始code/data，输出到FINAL_SUBMISSION/results：
- test-predictions.json：逐题选择结果。
- test-score.json：官方成绩。
- test-measurement.json：官方成绩加完整进程外部计时。

本次最终目录实测约1.645秒；CPU Intel i7-13700KF，Python3.12.10。模型资产4,970,167字节，约4.74MiB。满足本地30秒、16MiB限制，参考机器仍需复测。

## 在另一台机器复现
安装Python3.12，准备未修改的官方数据目录。命令：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r FINAL_SUBMISSION/requirements.txt
.venv/Scripts/python.exe FINAL_SUBMISSION/run.py --data-dir C:/absolute/path/to/code/data
```

无需重训。压缩包不含官方数据及便携Python，可在当前项目目录旁解压后直接复用它们。要求固定解析器和数据哈希，未采用网络服务或答案查询表。

## 最终模型的运行过程
1. 官方加载器校验输入并读取四条候选及1024维嵌入。
2. 提取35维文本/数值/答案一致性特征及8维保守算式核验特征；拼接原始嵌入，共1067维。官方parse_answer决定候选是否可评分。
3. 三个本地训练的自注意力网络分别标准化输入、裁剪至[-8,8]，输出四候选概率并平均。种子42/123/2026，结构相同，是三个随机种子模型而不是三种架构。
4. 成对逻辑回归排序模型独立计算分数，softmax转换为候选概率。
5. 75%注意力集成与25%排序概率融合，同一解析答案候选取平均，无效候选屏蔽，返回一个候选索引。
6. 独立官方评分器此时才读取测试参考答案，计算389/467。

## 固定配方从头训练
重训不会覆盖checkpoint，而是写入retrained_checkpoint。先安装requirements-training-lock.txt（需要其中CPU版PyTorch来源）：

```powershell
./.python/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r FINAL_SUBMISSION/requirements-training-lock.txt
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
./.python/python.exe FINAL_SUBMISSION/train_fixed.py
```

其他机器指定数据：设置环境变量MP2_DATA_DIR为官方data绝对路径，再执行train_fixed.py。

三个网络各5轮、宽128、2层、4头、GELU、dropout0.3、无位置编码。AdamW学习率0.0001、weight_decay0.08、batch64、梯度裁剪3。损失为0.5候选成对排序+0.5正确候选总概率负对数+0.15候选二分类。标准化只拟合训练输入。排序模型采用正负候选差值、L2逻辑回归，C=0.001、无截距、max_iter=2000。训练仅使用官方训练标签；训练脚本输出验证准确率用于复核，但不会用它改变固定配置。

重训输出可通过官方命令指定checkpoint评估，跨平台浮点差异可能造成重训差异；提交成绩以本目录冻结checkpoint的复现为准。