#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: yiban_API
# Created on: 2020/11/4

import requests
import re
from base64 import b64encode
from Cryptodome.Cipher import PKCS1_v1_5
from Cryptodome.PublicKey import RSA
from cnn.captchaRecongnise_CNN import *

BASEURL = 'https://www.yiban.cn/'
HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3938.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
TIMEOUT = 10

PROXIES = None
#PROXIES = {"http":"http://127.0.0.1:8080", "https":"http://127.0.0.1:8080"}



class yibanAPI:
    def __init__(self, configs):
        self.configs = configs  # 配置字典 {"username": str, "password": str}
        self.token = None       # 用户token
        self.groups = []        # 群组信息列表
        self.actor_id = None    # 执行ID
        self.nick = None        # 昵称

        self.vote = None        # 群组投票类

        self.session = None     # 总会话(包含cookies)

        self.vote = self._vote(self)

        self.topic = self._topic(self)

    # 检查是否已登录
    def _login_check(self):
        if self.session is None:
            raise Exception("please login first")

    # 模拟 JSEncrypt 加密
    def _rsa_encrypt(self, pwd, key):
        cipher = PKCS1_v1_5.new(RSA.importKey(key))
        ret = b64encode(cipher.encrypt(pwd.encode()))
        return ret

    # 登录获取用户 Token
    def login(self, set_session=None):
        login_page = BASEURL + "login"
        login_url = BASEURL + 'login/doLoginAjax'
        username = self.configs["username"]
        pwd = self.configs["password"]

        if set_session is None:
            session = requests.Session()
        else:
            session = set_session

        page_session = session.get(login_page,
                                   headers=HEADER,
                                   timeout=TIMEOUT)
        try:
            RSA_key = re.search(r'data-keys=\'([\s\S]*?)\'', page_session.text).group(1)
            keys_time = re.search(r'data-keys-time=\'(.*?)\'', page_session.text).group(1)
        except:
            raise Exception("failed to connect to server: {}".format(page_session.text))

        password = self._rsa_encrypt(pwd, RSA_key)

        data = {
            'account': username,
            'password': password,
            'captcha': None,
            'keysTime': keys_time,
            'is_rember': 1
        }

        atmp = 0
        token = None
        login_session = None

        while atmp < 40:
            login_session = session.post(login_url, headers=HEADER,
                                         data=data, timeout=TIMEOUT)
            try:
                token = login_session.cookies['yiban_user_token']
                break
            except:
                code = login_session.json()['code']
                if code in ("711", "201"):  # 需要填写验证码
                    captcha = captcha_handle(session)
                    data["captcha"] = captcha
                else:
                    break
            atmp += 1

        if token is None:
            raise Exception("failed to get user token: {}".format(
                            login_session.json()["message"]))
        else:
            user_session = session.post(BASEURL + "ajax/my/getLogin",
                                        headers=HEADER,
                                        timeout=TIMEOUT)
            actor_id = user_session.json()['data']['user']['id']
            nick = user_session.json()['data']['user']['nick']

            self.nick = nick
            self.token = token
            self.actor_id = actor_id

        self.token = token
        self.session = session

        if set_session is None:
            return token
        else:
            return session

    # 获取群组信息
    def scan_groups(self):
        self._login_check()

        session = self.session

        if PROXIES is None:
            try:
                get_session = session.get(BASEURL + "my/group/type/public",
                                          headers=HEADER,
                                          timeout=TIMEOUT)
            except AttributeError:
                get_session = session.get(BASEURL + "my/group/type/create",
                                           headers=HEADER,
                                           timeout=TIMEOUT)
        else:
            try:
                get_session = session.get(BASEURL + "my/group/type/public",
                                           headers=HEADER,
                                           timeout=TIMEOUT,
                                           proxies=PROXIES,
                                           verify=False)
            except AttributeError:
                get_session = session.get(BASEURL + "my/group/type/create",
                                           headers=HEADER,
                                           timeout=TIMEOUT,
                                           proxies=PROXIES,
                                           verify=False
                                           )
        group_info_raw = re.findall(r'href="/newgroup/indexPub/group_id/(\d+)/puid/(\d+)"',
                                get_session.text)

        group_info = []
        for i in group_info_raw:
            if not i in group_info:
                group_info.append(i)

        groups = []

        for i in group_info:
            group_id = i[0]
            puid = i[1]

            payload = {
                'puid': puid,
                'group_id': group_id
            }

            channel_session = session.post(BASEURL + "forum/api/getListAjax",
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)
            channel_id = channel_session.json()['data']['channel_id']

            groups.append({
                'group_id': group_id,
                'puid': puid,
                'channel_id': channel_id,
            })

        info = groups

        self.groups = groups

        return info

    def get_egpa(self, group_info):
        self._login_check()

        session = self.session

        if isinstance(group_info, int):
            group_info = self.groups[group_info]

        group_id = group_info["group_id"]
        puid = group_info["puid"]

        egpa_session = session.get(BASEURL+"newgroup/indexPub/group_id/"+
                                   "{}/puid/{}".format(group_id,
                                                       puid),
                                   headers=HEADER,
                                   timeout=TIMEOUT)

        egpa = re.search(r"EGPA：[0-9\.]*", egpa_session.text).group()
        egpa = str(egpa).split("：")[-1]

        return egpa


    def close(self):
        self.session.close()
        self.session = None

    # 投票接口
    class _vote:
        def __init__(self, upper_class):
            self.group_id = None                    # 群组ID
            self.puid = None                        # 群组PUID
            self.channel_id = None                  # 频道ID

            self.upper = upper_class                # 前置类

            self.votes = []                         # 投票信息

        def set_group(self, group_info):
            if isinstance(group_info, int):
                group_info = self.upper.groups[group_info]

            self.group_id = group_info["group_id"]
            self.puid = group_info["puid"]
            self.channel_id = group_info["channel_id"]

        # 易班发起单选双项投票
        # 参数: 标题, 正文, 选项1, 选项2
        def add(self, title, subjectTxt, subjectTxt_1, subjectTxt_2,
                subjectPic="", voteValue=1893427200,
                public_type=0, isAnonymous=0, istop=1,
                sysnotice=2, isshare=1):
            self.upper._login_check()

            session = self.upper.session

            payload = {
                'puid': self.puid,
                'scope_ids': self.group_id,
                'title': title,
                'subjectTxt': subjectTxt,
                'subjectPic': subjectPic,
                'options_num': 2,
                'scopeMin': 1,
                'scopeMax': 1,
                'minimum': 1,
                'voteValue': time.strftime("%Y-%m-%d %H:%M", time.localtime(voteValue)),
                'voteKey': 2,
                'public_type': public_type,
                'isAnonymous': isAnonymous,
                "voteIsCaptcha": 0,
                'istop': istop,
                'sysnotice': sysnotice,
                'isshare': isshare,
                'rsa': 1,
                'dom': '.js-submit',
                'group_id': self.group_id,
                'subjectTxt_1': subjectTxt_1,
                'subjectTxt_2': subjectTxt_2
            }

            add_vote_session = None

            for i in range(40):
                if PROXIES is None:
                    add_vote_session = session.post(BASEURL + 'vote/vote/add',
                                                     headers=HEADER,
                                                     data=payload,
                                                     timeout=TIMEOUT)
                else:
                    add_vote_session = session.post(BASEURL + 'vote/vote/add',
                                                     headers=HEADER,
                                                     data=payload,
                                                     timeout=TIMEOUT,
                                                     proxies=PROXIES,
                                                     verify=False)

                code = add_vote_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return add_vote_session.json()['message']

        # 获取投票
        def scan_votes(self, size=40, page=0, status=1, sort=1, time=0):
            self.upper._login_check()

            session = self.upper.session

            payload = {
                'puid': self.puid,
                'group_id': self.group_id,
                'page': page,
                'size': size,
                'status': status,
                'sort': sort,
                'time': time
            }

            get_vote_session = session.post(BASEURL + 'vote/index/getVoteList',
                                             headers=HEADER,
                                             data=payload,
                                             timeout=10)

            self.votes = get_vote_session.json()["data"]["list"]

            return self.votes

        # 参与投票
        def vote(self, vote_info, choice=0):
            self.upper._login_check()

            if isinstance(vote_info, int):
                vote_info = self.votes[vote_info]

            session = self.upper.session

            vote_id = vote_info["id"]
            voptions_id = vote_info["option_list"][choice]["id"]

            payload = {
                'puid': self.puid,
                'group_id': self.group_id,
                'vote_id': vote_id,
                'voptions_id': voptions_id,
                'minimum': 1,
                'scopeMax': 1
            }

            vote_session = None

            for i in range(40):
                vote_session = session.post(BASEURL + 'vote/vote/act',
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)

                code = vote_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return vote_session.json()["message"]

        # 评论投票
        def reply(self, vote_info, msg, comment_id=0, user_id=0):
            self.upper._login_check()

            if isinstance(vote_info, int):
                vote_info = self.votes[vote_info]

            session = self.upper.session

            mount_id = vote_info["Mount_id"]
            vote_id = vote_info["id"]
            author_id = vote_info["User_id"]

            payload = {
                'mountid': mount_id,
                'msg': msg,
                'group_id': self.group_id,
                'actor_id': self.upper.actor_id,
                'vote_id': vote_id,
                'author_id': author_id,
                'puid': self.puid,
                'reply_comment_id': comment_id,
                'reply_user_id': user_id
            }

            reply_session = None

            for i in range(40):
                reply_session = session.post(BASEURL + 'vote/vote/addComment',
                                             headers=HEADER,
                                             data=payload,
                                             timeout=TIMEOUT)

                code = reply_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return reply_session.json()["message"]

        # 删除评论
        def del_reply(self, vote_info, comment_id):
            self.upper._login_check()

            if isinstance(vote_info, int):
                vote_info = self.votes[vote_info]

            session = self.upper.session

            mount_id = vote_info["Mount_id"]
            vote_id = vote_info["id"]
            author_id = vote_info["User_id"]

            payload = {
                'mountid': mount_id,
                'commentid': comment_id,
                'puid': self.puid,
                'group_id': self.group_id,
                'author_id': self.upper.actor_id,
                'comment_author_id': self.upper.actor_id,
                'reply_name': 'noname',
                'vote_id': vote_id
            }

            reply_session = None

            for i in range(40):
                reply_session = session.post(BASEURL + 'vote/vote/addComment',
                                             headers=HEADER,
                                             data=payload,
                                             timeout=TIMEOUT)

                code = reply_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return reply_session.json()["message"]

        # 点赞投票
        def like(self, vote_info):
            self.upper._login_check()

            if isinstance(vote_info, int):
                vote_info = self.votes[vote_info]

            session = self.upper.session

            vote_id = vote_info["id"]

            payload = {
                'group_id': self.group_id,
                'puid': self.puid,
                'vote_id': vote_id,
                'actor_id': self.upper.actor_id,
                'flag': 1
            }

            like_session = None

            for i in range(40):
                like_session = session.post(BASEURL + 'vote/vote/editLove',
                                             headers=HEADER,
                                             data=payload,
                                             timeout=TIMEOUT)

                code = like_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return like_session.json()["message"]

        def dislike(self, vote_info):
            self.upper._login_check()

            if isinstance(vote_info, int):
                vote_info = self.votes[vote_info]

            session = self.upper.session

            vote_id = vote_info["id"]

            payload = {
                'group_id': self.group_id,
                'puid': self.puid,
                'vote_id': vote_id,
                'actor_id': self.upper.actor_id,
                'flag': 0
            }

            like_session = None

            for i in range(40):
                like_session = session.post(BASEURL + 'vote/vote/editLove',
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)

                code = like_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return like_session.json()["message"]

        # 删除投票
        def delete(self, vote_info):
            self.upper._login_check()

            if isinstance(vote_info, int):
                vote_info = self.votes[vote_info]

            session = self.upper.session

            vote_id = vote_info["id"]

            payload = {
                'group_id': self.group_id,
                'puid': self.puid,
                'vote_id': vote_id
            }

            delete_session = None

            for i in range(40):
                delete_session = session.post(BASEURL + 'vote/Expand/delVote',
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)

                code = delete_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return delete_session.json()["message"]

    # 动态接口（暂未开放）
    class _tweet:
        def __init__(self, upper_class):
            self.upper = upper_class

        def add(self, msg, privacy_level):
            self.upper._login_check()

            session = self.upper.session

            payload = {
                'content': msg,
                'privacy': privacy_level,
                'dom': '.js-submit'
            }

            add_session = None

            for i in range(40):
                add_session = session.post(BASEURL + 'feed/add',
                                              headers=HEADER,
                                              data=payload,
                                              timeout=TIMEOUT)

                code = add_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return add_session.json()["message"]

    # 话题接口
    class _topic:
        def __init__(self, upper_class):
            self.upper = upper_class  # 前置类
            self.group_id = None        # 群组ID
            self.puid = None            # 群组PUID
            self.channel_id = None      # 频道ID

            self.topics = []            # 话题信息

        def set_group(self, group_info):
            if isinstance(group_info, int):
                group_info = self.upper.groups[group_info]

            self.group_id = group_info["group_id"]
            self.puid = group_info["puid"]
            self.channel_id = group_info["channel_id"]

        # 获取话题
        def scan_topics(self, size=40, Sections_id=-1,
                        need_notice=0, my=0):
            self.upper._login_check()

            session = self.upper.session

            topics = []
            for i in range(1, (size//10) + 2):
                payload = {
                    'channel_id': self.channel_id,
                    'puid': self.puid,
                    'group_id': self.group_id,
                    'page': i,
                    'size': 10,
                    'orderby': 'updateTime',
                    'Sections_id': Sections_id,
                    'need_notice': need_notice,
                    'my': my
                }

                get_session = None

                for t in range(40):
                    get_session = session.post(BASEURL + 'forum/article/listAjax',
                                               data=payload,
                                               headers=HEADER,
                                               timeout=TIMEOUT)

                    code = get_session.json()['code']
                    if code in ("911", "941"):
                        captcha = captcha_handle(session)
                        payload.update({"captcha": captcha})
                    else:
                        break

                topics.extend(get_session.json()["data"]["list"])

            self.topics = topics

            return topics

        # 添加话题
        def add(self, title, msg):
            self.upper._login_check()

            session = self.upper.session

            payload = {
                'puid': self.puid,
                'pubArea': self.group_id,
                'title': title,
                'content': msg,
                'isNotice': 'false',
                'dom': '.js-submit'
            }

            add_session = None

            for i in range(40):
                add_session = session.post(BASEURL + 'forum/article/addAjax',
                                           headers=HEADER,
                                           data=payload,
                                           timeout=TIMEOUT)

                code = add_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return add_session.json()["message"]

        # 评论话题
        def reply(self, topic_info, msg,
                  reply_id=0, syncFeed=0, isAnonymous=0):
            self.upper._login_check()

            if isinstance(topic_info, int):
                topic_info = self.topics[topic_info]

            session = self.upper.session

            article_id = topic_info["id"]

            payload = {
                'channel_id': self.channel_id,
                'puid': self.puid,
                'article_id': article_id,
                'content': msg,
                'reply_id': reply_id,
                'syncFeed': syncFeed,
                'isAnonymous': isAnonymous
            }

            reply_session = None

            for i in range(40):
                reply_session = session.post(BASEURL + 'forum/reply/addAjax',
                                           headers=HEADER,
                                           data=payload,
                                           timeout=TIMEOUT)

                code = reply_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return reply_session.json()["message"]

        # 点赞话题
        def like(self, topic_info):
            self.upper._login_check()

            if isinstance(topic_info, int):
                topic_info = self.topics[topic_info]

            session = self.upper.session

            article_id = topic_info["id"]

            payload = {
                'channel_id': self.channel_id,
                'puid': self.puid,
                'article_id': article_id
            }

            like_session = None

            for i in range(40):
                like_session = session.post(BASEURL + 'forum/article/upArticleAjax',
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)

                code = like_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return like_session.json()["message"]

        # 取消点赞
        def dislike(self, topic_info):
            self.upper._login_check()

            if isinstance(topic_info, int):
                topic_info = self.topics[topic_info]

            session = self.upper.session

            article_id = topic_info["id"]

            payload = {
                'channel_id': self.channel_id,
                'puid': self.puid,
                'article_id': article_id
            }

            like_session = None

            for i in range(40):
                like_session = session.post(BASEURL + 'forum/article/upDelArticleAjax',
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)

                code = like_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return like_session.json()["message"]

        # 删除话题
        def delete(self, topic_info):
            self.upper._login_check()

            if isinstance(topic_info, int):
                topic_info = self.topics[topic_info]

            session = self.upper.session

            article_id = topic_info["id"]

            payload = {
                'channel_id': self.channel_id,
                'puid': self.puid,
                'article_id_list': article_id,
            }

            delete_session = None

            for i in range(40):
                delete_session = session.post(BASEURL + 'forum/article/setDelAjax',
                                            headers=HEADER,
                                            data=payload,
                                            timeout=TIMEOUT)

                code = delete_session.json()['code']
                if code in ("911", "941"):
                    captcha = captcha_handle(session)
                    payload.update({"captcha": captcha})
                else:
                    break

            return delete_session.json()["message"]


# 验证码处理接口
def captcha_handle(session):
    global TIMEOUT
    captcha = session.get("https://www.yiban.cn/captcha/index?{}".format(
                          str(int(time.time()))),
                          timeout=TIMEOUT)
    captcha_bytes = captcha.content
    captcha_word = captcha_recongnise(captcha_bytes)
    return captcha_word


# 测试序列
def test():
    user = "17323113220"
    pwd = "_C4o0d9y6#"
    config = {"username":user,
              "password":pwd}
    api = yibanAPI(config)
    print("tring to login user: {}".format(user))
    token = api.login()
    print("user token received: {}".format(token))
    print("group info:\n  {}".format(api.scan_groups()))
    print("targeting group 1")
    api.vote.set_group(api.groups[0])
    print("adding vote test 1:\n{}".format(
        api.vote.add("最近蚊子多么", "最近蚊子多么", "多", "不多")
    ))
    print("adding vote test 2:\n{}".format(
        api.vote.add("阿巴阿巴", "阿巴阿巴", "阿巴阿巴", "阿巴阿巴")
    ))
    print("get votes: {}".format(
        api.vote.scan_votes()
    ))
    print("voting test:\n{}".format(
        api.vote.vote(0)
    ))
    print("vote replying test 1:\n{}".format(
        api.vote.reply(0,"reply test 1")
    ))
    print("vote replying test 2:\n{}".format(
        api.vote.reply(0, "reply test 2")
    ))
    print("vote like test:\n{}".format(
        api.vote.like(0)
    ))
    print("vote dislike test:\n{}".format(
        api.vote.dislike(0)
    ))
    print("vote deleting test:\n{}".format(
        api.vote.delete(0)
    ))
    api.topic.set_group(0)
    print("topic adding test:\n{}".format(
        api.topic.add("今天天气不错","是吧")
    ))
    api.topic.scan_topics()
    print("topic like test:\n{}".format(
        api.topic.like(0)
    ))
    print("topic dislike test:\n{}".format(
        api.topic.dislike(0)
    ))
    print("topic deleting test:\n{}".format(
        api.topic.delete(0)
    ))


def api_init():
    print("yiban API initialized")


if __name__ == "__main__":
    api_init()
    test()
else:
    api_init()