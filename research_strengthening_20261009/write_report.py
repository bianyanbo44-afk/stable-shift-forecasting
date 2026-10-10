"""Chinese completion/readiness notes grounded in the actual audited results."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=ROOT/'submission_eswa_strengthened_20261009';OUT.mkdir(exist_ok=True)
A=HERE/'analysis';audit=json.loads((A/'evidence_audit.json').read_text())
frames={phase:pd.read_csv(A/f'{phase}_macro_comparisons.csv') for phase in ['test','independent','withheld']}
def result(phase,ref):
    r=frames[phase].query('metric=="mae" and block_days==7 and baseline==@ref').iloc[0]
    return f'{100*r.relative_difference:+.2f}%（95% 区间 {100*r.ci_low:+.2f}% 至 {100*r.ci_high:+.2f}%）'
rows=[]
for phase,label in [('test','原基准重新评估'),('independent','EIA 原扩展重新评估'),('withheld','固定新实体迁移')]:
    rows.append('| '+label+' | '+' | '.join(result(phase,ref) for ref in ['frozen','cosa_expanded','nlms_expanded','history_clip'])+' |')
text='''# ESWA 补强结果与投稿判断

已经执行扩大验证搜索、历史反馈裁剪、五档预算、固定新实体迁移及单独计时，并据完整结果修订稿件。原检查点、原数值记录、原配置锁和语言修订版保留。

我的判断是：这版证据更完整，但不宜继续按“跨数据普遍提高预测精度”定位投稿。适合保留的贡献是一个有闭式解、可控制当前预测位移的在线校准接口；精度优势必须限于实际比较和设置。新实体上相对独立调参 NLMS 的平均优势发生反转，这是影响定位的主要结果，不能被整体相对冻结模型的改善掩盖。ESWA 的创新性与应用价值判断仍有实质风险，完整附件并不代表已经获得审稿通过。

## 完整平均 MAE 结果

以下为等权设置平均相对差异，负值表示 QNC 误差更低。区间来自预先安排的 5,000 次种子/日历块配对重采样，主块长 7 日；区间为逐项区间。全部 14/28 日敏感性与设置级 Holm 结果保留在 Source Data。

| 评估 | 相对冻结预测 | 相对扩大搜索 COSA | 相对独立调参 NLMS | 相对独立调参历史裁剪 |
|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

新实体 ECL 24 h StableShift 的 QNC/NLMS MAE 差异为 +21.62%，配对区间 +13.15% 至 +32.88%，48 项主检验族调整后 p=0.0096。ECL 168 h StableShift 的差异为 +6.39%，区间跨过零。两个 ECL DLinear 设置结果相同。四个 Ausgrid 设置均有 QNC/NLMS 改善并通过该检验族阈值。主文明确呈现这一分化。

## 补强内容与仍保留的边界

- QNC/NLMS/历史裁剪的步长扩为 0.005、0.02、0.05、0.2、0.4、0.8、1，均含不更新选项。COSA 初始网格为 4 个学习率乘 4 个更新预算；验证结果触边时按预先留档的修正加测一次，实际每个配置为 16–17 个活跃候选。12 个 QNC 配置仍选到 1，5 个 COSA 配置仍选到 50 次预算，不能称为穷尽优化。
- 历史裁剪使用同一归一化方向，只约束历史反馈预测上的位移。匹配步长时，原基准 QNC/历史裁剪平均差异约 -0.11%，新实体约 -0.35%。这使“查询依赖带来大幅额外精度收益”的表述不再合适。
- 五个预算 0.01、0.025、0.05、0.1、0.2 全量报告，分别固定主步长和按验证重选步长。主预算仍是 0.05，没有按测试结果改成最好预算。
- 新实体固定为 ECL/Ausgrid 101–116，60 份配置先锁定，再提取和评分。复用原源模型、升序位置映射和新实体训练期标准化；没有重新训练，也没有用新实体重选校准参数。它测试同一历史日历上的实体迁移，不是另一个新时间段。
- 10 个 ECL 实体在训练期全零，原数据接口的单位标准差回退保留。全部 16 个实体仍参与预测、更新和 MAE/RMSE；次要 MASE 只使用训练季节尺度有定义的 6 个实体，单独记录分母。这项计算修正在首次新实体评分前留档。各实体测试期均有非零变化，未按未来表现替换或删除实体。
- EIA July–August 原扩展已经被旧研究查看。新配置的评估标为探索性重新评估；旧配置在下载前锁定的原实验另保留，其证据身份不变。
- 对照仍是共同因果时序下的公开 COSA 输出模块，未复现其完整原生调度。未找到可以据实接入的 FAC 官方实现，未用临时自写方法冒充官方对照。
- 位移保证是一次更新在训练标准化、联合通道 RMS 坐标中的上限，不等于每通道物理单位限制，也不直接限制不同发布时间的预测变化。

## 核查与交付

60 个原基准、20 个 EIA、40 个新实体配置均完成。指标从逐预测误差和观测计数重算；所有方法评分掩码相同，更新标签先于预测起点。三轮共 '''+f"{sum(v['active_qnc_updates'] for v in audit['phases'].values()):,}"+''' 次活跃 QNC 更新没有超出记录容差的位移或反馈损失检查失败。验证时的新批量实现通过全部既有 QNC/NLMS 候选指标一致性检查。源模型和原记录经哈希保护；锁定时的实现副本与配置锁中的代码摘要一致。

单独计时采用缓存源预测、非批量适配器、预热及三次同步重复，排除共同源模型推理。复现包提供实际数据、检查点、所有选择、逐预测记录、计算修正、统计/图表代码与回放入口。正文、补充材料及各投稿字段使用同一新结果，原证据以 legacy 前缀区分。

AI 声明保持作者主导：作者承担研究方向、方法决策、核查、解释、修改与最终批准责任，并如实保留 Codex 在代码、检索、统计、图表和写作中的辅助用途。没有把实际研究辅助写成“只改语法”，也没有替作者声称已完成全部人工审阅。

本次只准备本地文件，没有提交期刊、更新公开仓库或发布新结果。作者提交前仍需核查结果和声明，并在 Elsevier declarations 工具完成正式利益声明；附件里的利益声明文件仍明确为草稿。

实验规划使用了 experimental-design 技能；其程序性软件来源及经核实的 Scientific Agent Skills 引用见复现包的 SOFTWARE_ATTRIBUTION.md，不作为预测方法的证据引用。
'''
(OUT/'STRENGTHENING_REPORT_zh.md').write_text(text,encoding='utf-8')
(OUT/'START_HERE_投稿说明.md').write_text('''# ESWA 补强版使用说明

先阅读 STRENGTHENING_REPORT_zh.md 的完整结果和定位判断。新实体上 QNC 平均误差高于独立调参 NLMS，正文已明确披露。推荐以“当前预测位移控制及其精度权衡”呈现，不能写成各场景普遍优于强基线。

upload_ready 包含新的匿名正文、补充材料、摘要、Highlights、作者标题页、投稿信、Source Data、可编辑 LaTeX 和匿名复现包。摘要可从 submission_fields.txt 或 00_Abstract.txt 复制。主文新结果对应 phase=test/independent/withheld 文件；legacy 文件保留原配置实验。AI_DECLARATION.txt 与正文一致。

旧投稿包及语言修订版未覆盖。新实验没有公开发布；标题页的公开链接明确指向较早版本。当前新证据由本次附件提供。作者在提交前完成实质核查、最终稿批准和 Elsevier 正式声明；07_Competing_interests_draft.docx 仍是草稿。

本地交付不包含期刊提交或公开发布行为。自动数值、源文件、匿名性和排版检查记录位于 quality；作者终审不可由自动检查替代。
''',encoding='utf-8')
print('READINESS AND COMPLETION REPORT WRITTEN FROM AUDITED OUTCOMES')
