# -

自己写着玩的，封号概不负责！

```
# 安装系统组件
winget install Microsoft.DirectX
pip install dxcam numpy opencv-python pyautogui pywin32

# 启用硬件加速（必须）
reg add HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock /v AllowDevelopmentWithoutDevLicense /t REG_DWORD /d 1
```

```
pip install dxcam --pre  # 安装开发版驱动
winget install Microsoft.DirectX  # 更新运行时库

# 启用Stealth模式（需要管理员权限）
dxcam.enable_stealth()
```

### 血蓝检测

血条：颜色#FF0000

长度总计：32px宽

中间前排：399.341~430.341
中间后排：461.388~492,388

蓝条：颜色#0072FF

长度总计：32px宽

中间前排：399.344~430,344
中间后排：461.391~430.391

### 敌人坐标

指名字的中心点的位置

后排，从左到右

```
[40,258],
[95,223],
[161,186],
[225,151],
[287,113]
```

前排，从左到右

```
[91,309],
[156,273],
[221,239],
[284,202],
[348,166]
```