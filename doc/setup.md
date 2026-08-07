# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 1. 安装 uv

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虚拟环境和依赖。安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装 Python 3.11：

```bash
uv python install 3.11
```

项目根目录中的 `.python-version` 和 `pyproject.toml` 会将 Python 版本限制为 3.11。

---

## 2. 安装

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 安装依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

其余 Python 依赖声明在 `pyproject.toml` 中。进入项目根目录并同步环境：

```bash
cd unitree_rl_mjlab
uv sync
```

`uv sync` 会根据 `uv.lock` 创建 `.venv` 并安装项目。运行脚本时使用 `uv run`，例如：

```bash
uv run python scripts/list_envs.py
```

## 总结

按照上述步骤完成后，您已经准备好在 uv 管理的 Python 3.11 环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

