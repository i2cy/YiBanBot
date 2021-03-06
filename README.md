## × 免责声明 ×
_本项目仅限用于对机器学习技术以及爬虫技术的综合实践学习用途，请勿用其制作参与任何违法活动、侵犯他人权益的活动，由此产生的任何后果，皆由使用者承担。_

# 项目简介
本项目可对网页爬虫中出现的验证码事件进行自动化处理，达到智能爬虫的目的。
以易班为例，在用户登录时产生的验证码由项目中的神经网络反向识别得出验证码内容，由爬虫API整合进数据包后发送给服务器，达到自动化的目的。

# 使用方法
 1. 使用`cnn/dataGnerator.py`生成汉字验证码数据集样本
 2. 修改`cnn/captchaRecongnise_CNN.py`头部全局参数后运行按照指示训练神经网络模型
 3. 修改`app_egpaBot.py`头部全局参数后运行即可启动机器人

 - 注意本项目为对`爬虫API`以及`神经网络接口`两子项目的综合性应用实践，所示例的应用层代码未做前端设计

# 环境需求
`Python 3.7+`
`pycryptodomex`
`requests`
`tensorflow 2.3+`
`matplotlib`
`psutil`
