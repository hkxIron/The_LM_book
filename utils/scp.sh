echo "HOME:$HOME"
# 同步数据
remote_project_path="/home/hkx/data/work/open/The_LM_book/"
local_project_path="$HOME/work/open/project/"
user_ip="hkx@10.224.104.101"
rsync -av -e ssh --exclude='*.git'  \
--exclude='.*' \
--exclude='*checkpoints*' \
--exclude='__pycache__/' \
--exclude='*.pdf' \
--exclude='wandb/' ${user_ip}:${remote_project_path} $local_project_path
echo "=======================复制数据结束======================"
