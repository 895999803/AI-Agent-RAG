# DocQA RAG Project

这是一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的文档问答系统，使用OpenAI嵌入、pgvector向量存储和Streamlit Web界面。

## 🚀 项目概述

DocQA RAG Project 是一个功能完整的文档问答系统，能够：

- **文档处理**：自动解析PDF文档并提取文本内容
- **向量化存储**：使用sentence-transformers将文本转换为向量并存储在PostgreSQL数据库中
- **智能检索**：基于语义相似度检索相关文档片段
- **问答生成**：利用大语言模型生成准确、相关的回答
- **质量评估**：内置DeepEval框架进行回答质量评估
- **查询扩展**：支持多种查询扩展策略提升检索效果

## 📁 项目结构

```
rag_project/
├── src/                          # 核心源代码
│   ├── config.py                # 配置文件
│   ├── database.py              # 数据库操作
│   ├── embedding.py             # 嵌入向量生成
│   ├── document_processor.py    # 文档处理
│   ├── retriever.py             # 文档检索
│   ├── generator.py             # 答案生成
│   ├── evaluator.py             # 质量评估
│   ├── query_expansion.py       # 查询扩展
│   ├── main.py                  # 命令行入口
│   └── regression_test.py       # 回归测试
├── documents/                   # PDF文档存储
│   ├── d_1.pdf                  # 机器学习在官方统计中的应用
│   ├── d_2.pdf                  # 物理启发的机器学习模型可解释性
│   └── d_3.pdf                  # 数据流主动学习综述
├── app.py                       # Streamlit Web应用
├── test_cases.json              # 测试用例
├── test_runner.py               # 测试运行器
├── requirements.txt             # 依赖包
├── .env                         # 环境配置
├── README.md                    # 项目说明
└── logs/                        # 日志文件
    └── rag_app.log
```

## 🛠️ 技术栈

### 核心技术
- **Python 3.8+** - 主要编程语言
- **Streamlit** - Web界面框架
- **PostgreSQL + pgvector** - 向量数据库
- **sentence-transformers** - 嵌入模型
- **OpenAI API** - 大语言模型

### 关键依赖
- `pypdf` - PDF文档解析
- `psycopg2-binary` - PostgreSQL连接
- `python-dotenv` - 环境变量管理
- `deepeval` - 质量评估框架
- `pytest` - 测试框架

## 📦 安装与配置

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd rag_project

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 数据库设置

#### PostgreSQL安装
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Windows
# 从官网下载安装包: https://www.postgresql.org/download/windows/
```

#### 创建数据库和扩展
```sql
-- 连接到PostgreSQL
sudo -u postgres psql

-- 创建数据库
CREATE DATABASE rag_db;

-- 创建用户（可选）
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- 启用pgvector扩展
\c rag_db
CREATE EXTENSION vector;
```

### 4. 配置环境变量

复制并编辑环境配置文件：

```bash
cp .env.example .env  # 如果有example文件
# 或直接编辑 .env
```

配置内容：
```env
OPENAI_API_KEY="your_openai_api_key_here"
DB_USER="postgres"
DB_PASSWORD="your_password"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="rag_db"
COLLECTION_NAME="document_embeddings"
```

## 🚀 使用指南

### 1. 文档注入

#### 通过Web界面
1. 启动应用：`streamlit run app.py`
2. 在侧边栏选择要注入的PDF文件
3. 点击"注入选中文档"按钮

#### 通过命令行
```bash
# 注入所有文档
python src/main.py --ingest

# 指定文档目录
python src/main.py --ingest --documents-dir my_documents
```

### 2. 问答功能

#### Web界面使用
1. 启动应用：`streamlit run app.py`
2. 在主界面输入问题
3. 可选择特定文件进行检索
4. 启用查询扩展功能（可选）
5. 点击"获取答案"按钮

#### 命令行使用
```bash
# 提问
python src/main.py --query "What is the main topic of the document?"
```

### 3. 查询扩展功能

系统支持三种查询扩展策略：

#### Basic Expansion（基础扩展）
- 使用LLM生成多个相关的查询变体
- 从不同角度探索原始查询
- 适用于一般场景

#### Synonyms Expansion（同义词扩展）
- 使用同义词和相关术语扩展查询
- 保持原始意图的同时使用不同词汇
- 适用于术语多样的领域

#### Context Expansion（上下文扩展）
- 基于检索到的上下文生成相关查询
- 探索上下文中的相关概念和细节
- 适用于深度探索场景

## 🧪 测试与评估

### 1. 运行测试套件

```bash
# 运行所有测试
python test_runner.py

# 生成测试报告
# 测试完成后会自动生成 test_report.html 和 test_report.json
```

### 2. 回归测试

```bash
# 运行回归测试
python src/regression_test.py
```

### 3. 质量评估指标

系统使用DeepEval框架评估回答质量，包括：

- **相关性（Relevance）**：回答与问题的相关程度
- **正确性（Correctness）**：回答的事实准确性
- **完整性（Completeness）**：回答的全面性
- **连贯性（Coherence）**：回答的结构和逻辑性

## 📊 性能优化

### 1. 数据库优化
- 定期重建索引：`REINDEX INDEX index_name;`
- 调整PostgreSQL配置以支持大量连接
- 考虑使用连接池

### 2. 内存管理
- 监控嵌入模型的内存使用
- 对大型文档使用分块处理
- 考虑使用流式处理

### 3. 缓存策略
- 缓存频繁查询的嵌入向量
- 实现结果缓存以减少重复计算
- 使用Redis等外部缓存系统

## 🔧 配置选项

### 嵌入模型配置
```python
# src/config.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 可更换为其他sentence-transformers模型
EMBEDDING_DIMENSION = 384
```

### 文档处理配置
```python
# src/config.py
CHUNK_SIZE = 1000        # 文本块大小
CHUNK_OVERLAP = 200      # 块重叠大小
```

### LLM配置
```python
# src/config.py
LLM_MODEL = "allenai/molmo-2-8b:free"  # 可更换为其他模型
TEMPERATURE = 0.7
```

## 🐛 故障排除

### 常见问题

#### 1. 数据库连接失败
```bash
# 检查PostgreSQL服务状态
sudo systemctl status postgresql

# 检查端口是否被占用
netstat -tlnp | grep 5432
```

#### 2. 嵌入模型加载失败
```bash
# 清除缓存重新下载
rm -rf ~/.cache/torch/sentence_transformers/
```

#### 3. 内存不足
```python
# 减少批量处理大小
# 在document_processor.py中调整chunk_size
```

### 日志查看
```bash
# 查看应用日志
tail -f logs/rag_app.log

# 查看测试日志
tail -f logs/test_app.log
```

## 📈 扩展功能

### 1. 支持更多文档格式
- 添加Word文档支持（.docx）
- 支持Markdown文件（.md）
- 支持纯文本文件（.txt）

### 2. 高级检索功能
- 实现多模态检索（文本+图像）
- 添加时间戳过滤
- 支持元数据搜索

### 3. 用户界面增强
- 添加用户认证
- 实现问答历史记录
- 支持文档上传和管理

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支：`git checkout -b feature-name`
3. 提交更改：`git commit -m 'Add feature'`
4. 推送到分支：`git push origin feature-name`
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [sentence-transformers](https://www.sbert.net/) - 优秀的嵌入模型库
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL向量扩展
- [DeepEval](https://github.com/confident-ai/deepeval) - 质量评估框架
- [Streamlit](https://streamlit.io/) - 简单易用的Web框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至项目维护者
- 参与讨论区

---

**注意**：使用本项目需要有效的OpenAI API密钥，相关费用由用户自行承担。