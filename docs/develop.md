# 发布

```bash
# 发布pip库
poetry build -f sdist
poetry publish
```

# 测试

```shell
# 新建python环境
python -m venv gp
source gp/bin/activate

# 临时取消python别名 (如果有)
unalias python

# 安装依赖
pip install .

# 测试
cd test
# 导出环境变量
export $(grep -v '^#' .env | sed 's/^export //g' | xargs)
python test.py
```