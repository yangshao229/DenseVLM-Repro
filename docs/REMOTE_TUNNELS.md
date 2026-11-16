# 从 Mac 连接 DenseVLM

Windows 主工程位于 `D:\DenseVLM-Repro`。请在 Windows 和 Mac 上使用同一个 GitHub 或 Microsoft 账号登录 VS Code Remote Tunnels。

1. 在 Windows 上打开 `D:\DenseVLM-Repro`。启动隧道可在 VS Code 命令面板运行 `Remote Tunnels: Turn on Remote Tunnel Access...`；也可以在打开的 Tunnel 终端完成初次许可确认和账号授权。看到主机名后保持对应窗口运行。
2. Mac 的 VS Code 安装微软的 [Remote - Tunnels](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-server) 扩展。按 `⇧⌘P` 运行 `Remote Tunnels: Connect to Tunnel...`，用相同账号登录，选择 Windows 主机（命令行启动时请求的主机名为 `densevlm-windows`，以实际显示为准）。
3. 在远程窗口选择「打开文件夹」，输入 `D:\DenseVLM-Repro`。打开或保存 `docs/` 下的文件，实际操作的是 Windows D 盘。同一窗口的终端和 GPU 任务在 Windows 上运行，Mac 不需要安装 CUDA。

连接期间 Windows 电脑须开机、联网并保持 Tunnel 运行。遇到 Windows 睡眠或 Tunnel 窗口关闭，先在 Windows 上恢复隧道。Tunnel 登录会授予该账号在 Windows 用户权限内开发的能力；打开工程文件夹只决定默认显示的项目。

如果 Tunnel 窗口已经关闭，可以从 Windows 终端执行：

```powershell
Set-Location 'D:\DenseVLM-Repro'
code tunnel --name densevlm-windows
```

[微软 VS Code Remote Tunnels 文档](https://code.visualstudio.com/docs/remote/tunnels)
