#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: yiban_specialVote
# Created on: 2020/11/10

from api.yiban_API import *
from cnn.captchaRecongnise_CNN import *
from base64 import b64decode
import numpy as np

SPECIAL_URL = "https://q.yiban.cn/"

#PROXYS = {"http":"http://127.0.0.1:8080", "https":"http://127.0.0.1:8080"}
PROXYS = None

ACCOUNTS = ["accounts/DQ2020_02.json",
            "accounts/DQ2020_03.json",
            "accounts/DQ2020_04.json",
            "accounts/JING2019.json",
            "accounts/HEZI_001.json",
            "accounts/CHENWEI2019.json",
            "accounts/KC2019_03.json",
            "accounts/ZIHAN2020.json",
            "accounts/ZHANGZIHAN2020.json",
            "accounts/TIANXIU2019.json"]

APP_ID = 908892
VOTE_ID = 146580
OPTIONS = [1978546]

BATCH_VOTE = True
BATCH_SIZE = 3
BATCH_RAND_OPTIONS = [1978550,
                      1978536]


DELAYER = None


class specialVote:
    def __init__(self, token,
                 app_id, vote_id, debug=False):
        self.token = dict(yiban_user_token=token)
        self.app_id = app_id
        self.vote_id = vote_id
        self.debug = debug
        self.session = requests.Session()

    def vote(self, options, payload=None):
        if payload is None:
            payload = {
                "App_id":           str(self.app_id),
                "Vote_id":          str(self.vote_id),
                "VoteOption_id[]":  options
            }


        while True:
            if PROXYS is None:
                vote_session = self.session.post(SPECIAL_URL + "vote/insertBoxAjax",
                                                 headers=HEADER,
                                                 cookies=self.token,
                                                 data=payload,
                                                 timeout=TIMEOUT
                                                 )
            else:
                vote_session = self.session.post(SPECIAL_URL + "vote/insertBoxAjax",
                                                 headers=HEADER,
                                                 cookies=self.token,
                                                 data=payload,
                                                 timeout=TIMEOUT,
                                                 proxies=PROXYS,
                                                 verify=False
                                                 )

            if vote_session.json()["code"] == 230:
                captcha_raw = vote_session.json()["img"]["img"].split(",")[-1]
                captcha_raw = b64decode(captcha_raw)
                captcha = captcha_recongnise(captcha_raw)
                payload.update({"code": captcha})
            elif vote_session.json()["code"] == 225:
                payload.pop("code")
            else:
                break

        if self.debug:
            return vote_session


class NormalIncreasingDelay:

    def __init__(self, delay_counts, total_delay_s, warning=True, auto_fix=True):
        self.delay_list = []
        self.num = delay_counts
        self.total_delay = total_delay_s
        self.offset = 0

        self.fix = 0

        self.warning = warning
        self.warning_level = 0
        self.auto_fix = auto_fix

        self.last_delayed = -1

        self.generate()

    def seek(self, offset=0):
        self.last_delayed = -1
        if offset < 0:
            offset = len(self.delay_list) + offset
        self.offset = offset

    def delay(self):
        delay_time = self.delay_list[self.offset]
        if self.last_delayed > 0:
            need = delay_time - (time.time() - self.last_delayed) + self.fix
            if need < 0:
                if need < self.fix:
                    self.warning_level += 1
                if self.warning_level > 10 and self.warning:
                    print("warning: can not keep up, total delay time may be longer"
                          ". you should make your program run faster")
                self.fix = need
                need = 0
            else:
                if self.warning_level >= 1:
                    self.warning_level -= 1
        else:
            need = delay_time
        time.sleep(need)
        self.offset += 1
        self.last_delayed = time.time()

    def generate(self):
        dots = np.random.randn(self.num)
        for i, ele in enumerate(dots):
            if ele < 0:
                dots[i] = -ele
        dots *= (self.total_delay/dots.max())
        dots.sort()
        last = 0
        res = []
        for ele in dots:
            res.append(ele-last)
            last = ele
        self.delay_list = res

        return res


def vote_beta(accounts, opt):
    users = list(accounts.keys())
    invalids = []
    for index,i in enumerate(users):
        try:
            print("Account: {}".format(i))
            token = yibanAPI({"username":i,"password":accounts[i]}).login()
            if token is None:
                invalids.append(i)
                continue
            sv = specialVote(token, APP_ID, VOTE_ID, True)
            if isinstance(opt, list):
                if BATCH_VOTE:
                    if BATCH_SIZE > len(opt):
                        for a in range(BATCH_SIZE - len(opt)):
                            offset = index%len(BATCH_RAND_OPTIONS) + a
                            if offset >= len(BATCH_RAND_OPTIONS):
                                offset = len(BATCH_RAND_OPTIONS) - offset
                            opt.append(BATCH_RAND_OPTIONS[offset])
                    print("voting result: {}".format(sv.vote(opt).json()["message"]))
                else:
                    for i2 in opt:
                        print("voting result: {}".format(sv.vote(i2).json()["message"]))
                #opt = opt[0]
            else:
                print("voting result: {}".format(sv.vote(opt).json()["message"]))
        except Exception as err:
            print("exception:", err)
        DELAYER.delay()

    return invalids


def main():
    global DELAYER
    last_day = "0"
    accounts = {}
    for i in ACCOUNTS:
        accounts.update(json.load(open(i,"r")))
    votes = input("how many would you vote per day ({} in total): ".format(
        len(accounts.keys())))
    time_for_vote = input("when would you like to vote everyday(type 8 for 8 o'clock): ")
    time_last_for_vote = input("how long do you think the voting should use(type 30 for 30 sec): ")
    DELAYER = NormalIncreasingDelay(int(votes), int(time_last_for_vote))
    if len(time_for_vote) == 1:
        time_for_vote = "0" + time_for_vote
    while True:
        time.sleep(2)
        day, hour = time.strftime("%d,%H").split(",")
        if (hour == time_for_vote and last_day != day) or last_day == "0":
            users = list(accounts.keys())
            random.shuffle(users)
            user_chose = users[:int(votes)]
            account_chose = {}
            for i in user_chose:
                account_chose.update({i:accounts[i]})
            vote_beta(account_chose, OPTIONS)
            last_day = day


if __name__ == "__main__":
    main()