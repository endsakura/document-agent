# Document Agent

基于 LangChain + MCP 的文档智能处理系统（OCR / YOLO / RAG / 知识图谱）。

## 快速开始

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
copy .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 3. 放置 YOLO 模型（不上传 GitHub）
mkdir models
copy best.pt models\best.pt

# 4. 启动
python main.py
# 浏览器打开 http://localhost:8000
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM API 密钥（必填） |
| `OPENAI_API_BASE` | API 地址，默认 openai-hub |
| `OPENAI_MODEL` | 模型名，默认 gpt-4o |
| `YOLO_MODEL_PATH` | YOLO 权重路径，默认 `models/best.pt` |

## YOLO 模型说明

`best.pt` **不会**包含在本仓库中。使用者需自行放置模型文件，详见 [models/README.md](models/README.md)。

## 上传到 GitHub

```powershell
cd D:\imageclassifationagent

# 初始化（只需一次）
git init
git add .
git status          # 确认没有 best.pt、.env、uploads/

git commit -m "Initial commit: Document Agent"
git branch -M main

# 在 github.com 新建空仓库后：
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

推送前务必检查 `git status`，确保 **best.pt / .env / uploads / chroma_db** 不在列表中。

## 项目结构

```
agent/          LangChain Agent 编排
tools/          OCR、YOLO、路由
knowledge/      25 类知识图谱与校验
memory/         ChromaDB RAG
client/         Web 对话前端
mcp_server.py   MCP 工具服务
main.py         FastAPI 入口
```
