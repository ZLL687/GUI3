# CFST Load–Strain Prediction · V2.2

## 上传与部署

和 V2.1 一样，上传以下 **5 个文件**到 GitHub 同一目录：

```text
packages.txt
README.md
requirements.txt
streamlitapp.py
V2.2_AB_models.joblib
```

先解压外层部署 ZIP，上传其中的 5 个文件；模型 `.joblib` 保持原样。每个文件均小于 GitHub 网页上传的 25 MiB 限制，无需 Git LFS。

Streamlit 选择自己的仓库和分支，入口填写 `streamlitapp.py`。**新部署推荐 Python 3.12**，依赖文件会安装匹配的计算库。原 Python 3.14 页面也支持通过独立 Python 3.12 运行计算，但首次准备需要联网，可能等待几分钟。

替换 V2.1 时同步更新这 5 个文件，移除本部署目录内不再使用的旧模型及旧依赖配置，避免混用。入口不再读取 `V2.1_AB_models.joblib`。

## 使用

选择 A/B 和几何材料输入或模型特征输入，设置应变窗口，点击 **Predict curve**。页面保留曲线、容量及经验区间、表格、SVG 和两种 CSV 导出。

- 尺寸 mm，强度 MPa；特征 `fcuAc`、`fyAs` 使用 N，荷载输出 kN。
- 橡胶替代率为 0–1 比例；角度为度；应变 0.01 表示 1%。
- 预测仍使用原来的十个输入。几何模式只换算这些特征；多管或特殊截面可填写有效面积。
- 新参数采用 0–0.03 的均匀 101 点参考网格；显示窗口通过插值获得，不改变完整峰值。历史数据库复算应使用各试件的实际网格。

## 模型范围

使用原始 815 个数据库试件（420 试验记录、395 原有有限元记录）的冻结 V2.2 模型，没有后生成的合成训练试件。单个模型包包含 A/B 原生模型及本地运行定义；只压缩封装，原生模型字节不变，不重训。

曲线和容量区间直接调用 V2.2 的原生 13 来源包络。90% 为经验标记；原 C 已用于点模型拟合/选择，不声称独立 split-conformal 覆盖保证。A 容量下界可为 0。曲线含历史标准化、截断/延伸，缺少未延伸端点标志。

这是预测部署包，不含完整数据库或完整 SHAP、消融、算法比较和窗口重训工作流。没有代填 DOI、许可或已上线地址；无需 Zenodo 链接或密钥。

## 本地启动

```text
python -m pip install -r requirements.txt
python -m streamlit run streamlitapp.py
```

已有兼容 Python 3.12 环境时，程序检查版本后直接使用；高级本地运行也可用 `V22_MODEL_PYTHON` 指定经版本核验的解释器。普通云端部署无需设置。

本机已核验 Python 3.14 页面调用已有 Python 3.12.7 环境。当前机器的新 Python 下载受网络限制，未将此描述为全新联网安装或 Linux 云端实测；实际 Cloud 首次安装和上线需上传后确认。

若启动或预测失败，核对 5 个文件是否匹配，并查看应用日志中 `[V2.2]` 的错误。修改输入后需再次点击预测。

平台参考：[Streamlit 部署](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)、[依赖配置](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)、[GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
