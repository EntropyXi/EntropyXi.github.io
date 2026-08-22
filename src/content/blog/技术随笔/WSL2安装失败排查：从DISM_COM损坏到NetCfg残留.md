---
title: "WSL2 安装失败排查：从 DISM COM 损坏到 NetCfg 残留"
description: "WSL2 安装失败排查实录。BIOS 虚拟化已开启、HypervisorPresent=True，wsl --install 却静默失败，VirtualMachinePlatform 一重启就回滚。最终定位为三个问题叠加：DISM COM 注册损坏、MuMu 模拟器的 VirtualBox 驱动残留、VMware 与 LDPlayer 遗留的 NetCfg 通知对象失效。记录了从修复 DISM COM、清理残留驱动到重建注册的完整排查过程。"
date: "2026-08-05T10:00:00+08:00"
updated: "2026-08-05T10:00:00+08:00"
tags:
  - "WSL"
  - "Windows"
  - "虚拟化"
  - "故障排查"
categories:
  - "技术随笔"
permalink: "2026/08/05/技术随笔/WSL2安装失败排查：从DISM_COM损坏到NetCfg残留"
math: false
draft: false
---

这次遇到的问题是，BIOS 里已经开了虚拟化，但是执行

```powershell
wsl --install
```

没有正常安装，`wsl --status` 还一直提示没有启用虚拟化。后面虽然成功启用了 `VirtualMachinePlatform`，但是一重启就回滚，界面显示“无法更新这些内容”。

最后查下来，不是单独一个问题，而是三个问题叠在一起：

1. DISM 的 COM 注册损坏，导致 `wsl --install` 内部调用 DISM 时直接失败；
2. MuMu 模拟器的 VirtualBox 内核驱动还在运行；
3. VMware 和 LDPlayer 留下了失效的 NetCfg 通知对象，导致 Hyper-V 网络驱动在重启阶段安装失败。

最后修复结果如下：

```text
WSL runtime: 2.7.11
Ubuntu: 26.04 LTS
WSL version: 2
Ubuntu VHDX: D:\WSL\Ubuntu\ext4.vhdx

vmcompute: RUNNING
hns: RUNNING
vfpext: RUNNING
WSLService: RUNNING
HvHost: RUNNING
```

下面按照实际排查顺序记录。

---

## 1. 一开始的问题

最初的现象是：

```powershell
wsl --install
```

执行以后没有安装过程，也没有明确错误。

但是 BIOS 里的虚拟化已经确认开启，而且系统中还能看到：

```text
HypervisorPresent=True
```

与此同时，`wsl --status` 输出：

```text
WSL2 无法启动，因为此计算机上未启用虚拟化
```

这里一开始看起来很矛盾。既然 `HypervisorPresent=True`，为什么 WSL 还说没有启用虚拟化？

我们先把这个问题拆开。WSL2 能不能运行，不只取决于 BIOS，还取决于下面这些组件：

```text
VirtualMachinePlatform
vmcompute
hns
vfpext
Hyper-V 网络驱动
WSLService
```

所以 `wsl --status` 这里说的“未启用虚拟化”，不一定真的是 BIOS 没开，也可能是底层组件没有部署完整。

---

## 2. 先检查 WSL 到底装到了什么程度

检查以后发现，Store 版 WSL 运行时已经存在：

```text
C:\Program Files\WSL\wsl.exe
C:\Program Files\WSL\wslservice.exe
C:\Program Files\WSL\wslg.exe
C:\Program Files\WSL\tools\kernel
```

System32 中也有：

```text
C:\Windows\System32\wsl.exe
C:\Windows\System32\computecore.dll
C:\Windows\System32\computenetwork.dll
C:\Windows\System32\computestorage.dll
```

但是下面两个关键文件不存在：

```text
C:\Windows\System32\vmcompute.exe
C:\Windows\System32\vmms.exe
```

`vmcompute` 服务也不存在。

这说明 WSL 不是完全没装，而是只装了上层运行时，底层的 `VirtualMachinePlatform` 没有完整部署。

同时还发现了几套第三方虚拟化环境：

```text
MuMu 模拟器
VMware Workstation
LDPlayer 卸载残留
```

其中 MuMu 的驱动：

```text
MuMuVMMDrv
```

当时仍然是 `RUNNING` 状态。

所以最开始的判断是：

```text
VirtualMachinePlatform 没有启用
+
MuMu 的 VirtualBox 驱动可能冲突
```

接下来先尝试启用 `VirtualMachinePlatform`。

---

## 3. 第一个真正的问题：DISM 自己坏了

执行：

```powershell
dism /online /enable-feature `
  /featurename:VirtualMachinePlatform `
  /all /norestart
```

结果报错：

```text
Failed to create DismHostManager remote object
hr: 0x80040154
REGDB_E_CLASSNOTREG
```

其中：

```text
0x80040154 = COM 类未注册
```

也就是说，这时候不是 `VirtualMachinePlatform` 启用失败，而是 DISM 连工作会话都没有创建成功。

继续检查发现：

```text
C:\Windows\System32\Dism\DismHost.exe
C:\Windows\System32\DismApi.dll
```

文件都存在，但是 DismHost 对应的 CLSID 不存在：

```text
HKLM\SOFTWARE\Classes\CLSID\
{D5EC7BD0-C0E0-4B4A-8D2A-6B8A9B2C1E44}
```

这就解释了前面的静默失败。

`wsl --install` 内部要调用 DISM，DISM 又因为 COM 注册损坏而启动失败，所以外面看起来就是命令没有反应。

---

## 4. 修复 DISM COM 注册

修复时先重新注册 DISM 的核心 DLL 和 Provider，包括：

```text
DismApi.dll
CbsProvider.dll
AppxProvider.dll
GenericProvider.dll
ImagingProvider.dll
MsiProvider.dll
MsuProvider.dll
ProvProvider.dll
VhdProvider.dll
WimProvider.dll
FfuProvider.dll
SmiProvider.dll
OfflineSetupProvider.dll
```

然后重新建立 DismHost 的 CLSID：

```text
HKLM\SOFTWARE\Classes\CLSID\
{D5EC7BD0-C0E0-4B4A-8D2A-6B8A9B2C1E44}
```

对应路径为：

```text
LocalServer32
C:\Windows\System32\Dism\DismHost.exe

InprocServer32
C:\Windows\System32\DismApi.dll
```

修复以后先执行：

```powershell
dism /online /get-features /format:table
```

这次能够正常输出功能列表，退出码是 `0`。

说明 DISM 已经恢复。

再次启用：

```powershell
dism /online /enable-feature `
  /featurename:VirtualMachinePlatform `
  /all /norestart
```

结果正常完成，返回：

```text
3010
```

`3010` 表示操作成功，但是需要重启。

到这里，第一个问题解决了。

---

## 5. 卸载 MuMu 以后还要检查驱动

MuMu 当时使用的 VirtualBox 内核驱动仍然在运行：

```text
MuMuVMMDrv
SYSTEM_START
RUNNING
```

所以不能只看程序有没有卸载，还需要检查驱动和服务。

先停止相关进程，再卸载 MuMu。卸载后检查：

```text
sc query MuMuVMMDrv
```

返回：

```text
错误 1060，指定的服务未安装
```

再执行：

```powershell
driverquery | findstr MuMu
```

没有输出。

同时确认：

```text
C:\Program Files\MuMuVMMVbox
```

已经没有实际文件，卸载注册表项也消失了。

这样才能确认 MuMu 的虚拟化驱动已经不再加载。

---

## 6. 第一次重启以后，vmcompute 还是没有出现

DISM 修复了，MuMu 也卸载了，而且事件日志显示 Hypervisor 已经正常启动。

但是重启以后：

```text
vmcompute.exe 仍然不存在
vmcompute 服务仍然不存在
```

`wsl --status` 还是报虚拟化不可用。

继续执行：

```powershell
wsl --install --no-distribution
```

命令成功结束，并生成：

```text
RebootPending=True
C:\Windows\WinSxS\pending.xml
```

这里说明 `VirtualMachinePlatform` 只是进入了待重启状态，还没有真正部署完成。

它的过程大致是：

```text
Disabled
    ↓
EnablePending
    ↓ 重启
Enabled
```

`pending.xml` 中保存的就是重启阶段需要执行的组件操作。

于是再次重启。

结果系统显示：

```text
某些修改无法完成
正在撤销更改
```

说明这次组件事务发生了回滚。

---

## 7. 中间尝试过手动部署，但是这条路不对

因为 `vmcompute` 一直没有部署出来，中间尝试过从 WinSxS 手动复制 HCS 相关文件，并注册：

```text
vfpext
vmcompute
hns
```

一开始尝试建立硬链接，但是全部返回：

```text
错误 5：拒绝访问
```

后面改成 `Copy-Item`，虽然能够复制文件，也能创建服务，但是服务启动依赖不完整。

更关键的是，系统重启以后，这些手动复制的文件和服务又被 CBS 回滚了。

这说明 Windows 组件部署不只是文件复制，它还涉及：

```text
WinSxS 清单
CBS 组件状态
驱动暂存
服务依赖
NetCfg 注册
重启事务
```

所以后面停止手动复制，重新走官方流程：

```text
disable /remove
enable /all
重启
```

但是第二次重启以后仍然回滚。

这说明手动部署不是唯一问题，系统内部还有别的故障。

---

## 8. 从 CBS 日志继续往下查

第一次回滚时，CBS 日志只看到：

```text
last forward execute state:
CbsExecuteStateStageDrivers

Startup:
CbsExecuteStateUnstageDrivers
```

说明事务是在驱动暂存阶段失败的。

但是这个信息还不够具体。

第二次走纯官方流程后仍然失败，这次日志里出现了更准确的内容：

```text
Installing network driver ms_l1vhlwf

Invoke NetCfg Notify Objects for install

ErrorCode: 8007007e
ERROR_MOD_NOT_FOUND

A rollback will be initiated
```

`0x8007007E` 表示：

```text
找不到指定的模块
```

一开始还需要确认，到底是 Hyper-V 自己的驱动文件缺失，还是 NetCfg 加载了某个第三方模块以后失败。

这里的：

```text
ms_l1vhlwf
```

是 Hyper-V 网络组件的 NetCfg 组件 ID，并不是一定存在一个叫 `ms_l1vhlwf.sys` 的文件。

而日志中真正重要的一行是：

```text
Invoke NetCfg Notify Objects for install
```

NetCfg 安装网络组件时，会枚举系统中已经注册的网络通知对象，并调用对应的 COM 模块。

所以只要某个通知对象的 CLSID 还在，但是对应 DLL 已经没了，就可能导致整个安装失败。

---

## 9. 中间还误判过一次 WFP 文件缺失

当时发现下面这些文件在 System32 和 WinSxS 中都不存在：

```text
wfp.dll
wfpcal.sys
basefilterengineapi.dll
```

看名字很像 Windows Filtering Platform 的组件，所以一度怀疑是这些文件损坏了。

但是继续用 Winbindex 检查以后发现：

```text
wfp.dll                  → 没有记录
wfpcal.sys               → 没有记录
basefilterengineapi.dll  → 没有记录

bfe.dll                  → 有正常 Windows 版本记录
wfplwfs.sys              → 有正常 Windows 版本记录
```

所以前三个文件本来就不是当前系统中的标准文件。

这里的问题不是文件丢失，而是我们找错了对象。

这一步也说明，看到一个文件不存在以后，不能马上判断系统损坏，还需要先确认这个文件在当前 Windows 版本中本来是否存在。

---

## 10. 扫描 NetCfg 通知对象以后找到真正原因

接下来扫描：

```text
HKLM\SOFTWARE\Classes\CLSID
```

主要查两类内容：

1. 名称包含 `network`、`notify`、`bridge`、`filter`、`adapter` 等关键词；
2. `InprocServer32` 或 `LocalServer32` 指向的文件不存在。

最后找到了两个主要问题。

### 10.1 VMware Bridge notifier 路径已经过期

注册表中存在：

```text
CLSID:
{3d09c1ca-2bcc-40b7-b9bb-3f3ec143a87b}

Name:
VMware Bridge notifier object

InprocServer32:
D:\VMware\vmnetbridge.dll
```

但是：

```text
D:\VMware\vmnetbridge.dll
```

已经不存在。

继续查以后发现真正的 DLL 位于：

```text
C:\Windows\System32\vmnetbridge.dll
```

也就是说，VMware 之前可能装在 D 盘，后面发生过迁移或重装，但是 CLSID 里的路径没有更新。

因此 NetCfg 安装 Hyper-V 网络组件时会出现：

```text
读取 VMware Bridge notifier
    ↓
加载 D:\VMware\vmnetbridge.dll
    ↓
文件不存在
    ↓
ERROR_MOD_NOT_FOUND
    ↓
CBS 回滚
```

### 10.2 LDPlayer 卸载后还留下了 VirtualBox CLSID

系统中还存在几个 LDPlayer / VirtualBox 相关 CLSID，指向：

```text
C:\Program Files\ldplayer9box\VBoxProxyStub.dll
C:\Program Files\ldplayer9box\VBoxC.dll
C:\Program Files\ldplayer9box\Ld9BoxSVC.exe
```

这些文件都已经不存在。

所以 LDPlayer 虽然卸载了，但是部分 COM 注册还留着。平时可能没有问题，一旦 NetCfg 安装新的网络组件，就会枚举到这些残留对象。

---

## 11. 修复 VMware 路径，删除 LDPlayer 残留

VMware 的情况是 DLL 仍然存在，只是路径错了。

所以把：

```text
D:\VMware\vmnetbridge.dll
```

修改为：

```text
C:\Windows\System32\vmnetbridge.dll
```

LDPlayer 的几个 VirtualBox CLSID 对应文件已经完全不存在，因此直接删除，同时处理 WOW6432Node 中对应的项。

另外还清理了几个指向：

```text
D:\VMware\elevated.dll
```

但是文件已经不存在的 VMware 工具类 CLSID。

修复以后再次扫描：

```text
网络相关 + DLL 不存在的 CLSID = 0
```

到这里，第三个问题解决。

---

## 12. 再次部署 VirtualMachinePlatform

清理完成以后，重新执行：

```text
disable /remove
enable /all
```

系统重新进入：

```text
EnablePending
```

并重新生成：

```text
C:\Windows\WinSxS\pending.xml
```

再次重启以后，下面这些文件都已经正常部署：

```text
vmcompute.exe
HostNetSvc.dll
vfpext.sys
vmswitch.sys
vmwp.exe
```

服务状态为：

```text
vmcompute: RUNNING
hns: RUNNING
```

`pending.xml` 已经处理，`RebootPending=False`。

随后启动 `vfpext`，再执行：

```powershell
wsl --status
```

输出已经变成：

```text
默认版本: 2
```

之前的“未启用虚拟化”提示消失。

这里需要注意，`pending.xml` 消失本身不能单独证明成功，因为回滚以后它也可能消失。

还需要一起确认：

```text
组件文件存在
服务已经注册
服务能够启动
wsl --status 正常
```

---

## 13. 把 Ubuntu 安装到 D 盘

底层问题解决以后，先更新 WSL：

```powershell
wsl --update
```

版本从：

```text
2.7.10
```

更新到：

```text
2.7.11
```

然后安装 Ubuntu：

```powershell
wsl --install Ubuntu `
  --location D:\WSL\Ubuntu `
  --no-launch `
  --web-download
```

安装完成以后检查：

```text
D:\WSL\Ubuntu\ext4.vhdx
```

文件存在。

同时递归检查 C 盘用户目录，没有发现 Ubuntu 的 `ext4.vhdx`。

所以 Ubuntu 的实际数据位置就是：

```text
D:\WSL\Ubuntu\ext4.vhdx
```

---

## 14. 最终验证

执行：

```powershell
wsl -l -v
```

结果：

```text
NAME       STATE      VERSION
Ubuntu     Running    2
```

Ubuntu 内部：

```text
Ubuntu 26.04 LTS
Linux 6.18.33.2-microsoft-standard-WSL2
x86_64
```

`apt` 可以正常运行，`/tmp` 文件读写正常。

Windows 侧服务：

```text
vfpext        RUNNING
vmcompute     RUNNING
hns           RUNNING
WSLService    RUNNING
HvHost        RUNNING
```

另外也确认了：

```text
BIOS 没有改动
没有额外安装其他 Hypervisor
只安装了 Ubuntu
没有创建 .wslconfig
没有启用 Hyper-V Manager
没有修改原来的开发环境
```

---

## 15. 总结

这次问题一开始只是：

```text
wsl --install 没反应
```

但是实际的故障顺序是：

```text
DISM COM 注册损坏
    ↓
修复后可以启用 VirtualMachinePlatform
    ↓
重启阶段安装 Hyper-V 网络组件
    ↓
NetCfg 加载 VMware / LDPlayer 残留对象失败
    ↓
CBS 整体回滚
```

MuMu 的 VirtualBox 驱动也是需要先排除的冲突项。

整个过程里比较关键的是三次转向：

1. 从“BIOS 没开虚拟化”转到检查 `vmcompute` 和 `VirtualMachinePlatform`；
2. 从“VMP 启用失败”转到发现 DISM COM 注册损坏；
3. 从“Hyper-V 文件缺失”转到检查 NetCfg 通知对象和第三方 CLSID。

最后的处理方式也不是手动拼接系统文件，而是先把阻塞官方部署流程的问题修掉，再重新执行：

```text
disable /remove
enable /all
重启
```

这次排查说明一个问题：复杂的系统故障可能不只有一个根因。前一个问题修复以后，又出现新的错误，不一定是修错了，也可能只是后面的故障终于暴露出来。
