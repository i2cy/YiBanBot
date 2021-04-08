#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: app_egpaBot
# Created on: 2020/11/13

from api.yiban_API import *
from cnn.captchaRecongnise_CNN import *
import threading
import sys

ACCOUNT_FILE = "accounts/example.json"  # 帐号密码信息

VOTE_COUNT = 30  # 每帐号每天发起的投票数
TOPIC_COUNT = 30  # 每帐号每天发起的话题数
REPLY_COUNT = 8  # 每帐号每天对每个项目参与的回复数

DELAY = 75  # 每帐号的操作最大延迟

TEXT_URL = "https://v1.hitokoto.cn/?encode=text"

THREADS = 0
EGPA = 0
LAST_TOP = ""
LIVE = True


def rand_delay():
    dt = random.randint(int(DELAY * 0.8), DELAY)
    time.sleep(dt)


def echo(msg, top=False):
    global LAST_TOP
    msg = str(msg)
    if top:
        sys.stdout.write("\r" + len(LAST_TOP) * " " + "\r")
        sys.stdout.flush()
        sys.stdout.write(msg)
        LAST_TOP = msg
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + len(LAST_TOP) * " " + "\r")
        sys.stdout.flush()
        sys.stdout.write(msg + "\n")
        sys.stdout.write(LAST_TOP)
        sys.stdout.flush()


def get_rand_text():
    try:
        session = requests.get(TEXT_URL,
                               timeout=TIMEOUT)
        ret = session.text
    except:
        ret = "Opps"
    return ret


def load_accounts(filename):
    accounts = []
    accounts_raw = json.load(open(filename, "r"))
    keys = list(accounts_raw.keys())
    for i in keys:
        accounts.append({
            "username": i,
            "password": accounts_raw[i]
        })
    return accounts


def bot_thread(account, get_egpa=False):
    global THREADS, EGPA
    THREADS += 1

    bot_head = "[{}]".format(account["username"])

    echo("{} info: bot started".format(bot_head))

    try:
        api = yibanAPI(account)
        ret = None
        for i in range(10):
            try:
                ret = api.login()
            except Exception as err:
                echo("{} warning: {}".format(bot_head, err))
            if ret is None:
                time.sleep(3)

        if ret is None:
            raise Exception("account \"{}\" failed to login after 10 tries".format(
                account["username"]))

        ret = api.scan_groups()
        if ret in (None, []):
            raise Exception("no group found in user")

        api.vote.set_group(0)

        if get_egpa:
            try:
                EGPA = api.get_egpa(0)
            except:
                pass

        if not LIVE:
            raise Exception("exiting")

        for i in range(VOTE_COUNT):
            try:
                ret = api.vote.add(get_rand_text(),
                                   get_rand_text(),
                                   get_rand_text(),
                                   get_rand_text())
                echo("{} info: adding vote, {}, count {}".format(bot_head,
                                                                 ret, i))
            except Exception as err:
                echo("{} warning: failed to add vote, {}, count {}".format(bot_head,
                                                                           err, i))

            rand_delay()
            if not LIVE:
                raise Exception("exiting")
            if get_egpa:
                try:
                    EGPA = api.get_egpa(0)
                except:
                    pass

        api.topic.set_group(0)

        for i in range(TOPIC_COUNT):
            try:
                ret = api.topic.add(get_rand_text(),
                                    get_rand_text())
                echo("{} info: adding topic, {}, count {}".format(bot_head,
                                                                  ret, i))
            except Exception as err:
                echo("{} warning: failed to add topic, {}, count {}".format(bot_head,
                                                                            err, i))
            rand_delay()
            if not LIVE:
                raise Exception("exiting")
            if get_egpa:
                try:
                    EGPA = api.get_egpa(0)
                except:
                    pass

        votes = api.vote.scan_votes()

        if len(votes) < VOTE_COUNT:
            count = votes
        else:
            count = VOTE_COUNT

        for i in range(count):
            try:
                ret = api.vote.vote(i)
                echo("{} info: voting votes, {}, count {}".format(bot_head,
                                                                  ret, i))
            except Exception as err:
                echo("{} warning: failed to vote, {}, count {}".format(bot_head,
                                                                       err, i))

            try:
                ret = api.vote.like(i)
                echo("{} info: liking votes, {}, count {}".format(bot_head,
                                                                  ret, i))
            except Exception as err:
                echo("{} warning: failed to like vote, {}, count {}".format(bot_head,
                                                                            err, i))

            for t in range(REPLY_COUNT):
                try:
                    ret = api.vote.reply(i, get_rand_text())
                    echo("{} info: replying votes, {}, count {}".format(bot_head,
                                                                        ret, i))
                except Exception as err:
                    echo("{} warning: failed to like vote, {}, count {}".format(bot_head,
                                                                                err, i))
                rand_delay()
                if not LIVE:
                    raise Exception("exiting")
                if get_egpa:
                    try:
                        EGPA = api.get_egpa(0)
                    except:
                        pass

            rand_delay()
            if not LIVE:
                raise Exception("exiting")
            if get_egpa:
                try:
                    EGPA = api.get_egpa(0)
                except:
                    pass

        topics = api.topic.scan_topics()

        if len(topics) < TOPIC_COUNT:
            count = topics
        else:
            count = TOPIC_COUNT

        for i in range(count):
            try:
                ret = api.topic.like(i)
                echo("{} info: liking topic, {}, count {}".format(bot_head,
                                                                  ret, i))
            except Exception as err:
                echo("{} warning: failed to like topic, {}, count {}".format(bot_head,
                                                                             err, i))

            for t in range(REPLY_COUNT):
                try:
                    ret = api.topic.reply(i, get_rand_text())
                    echo("{} info: replying topics, {}, count {}".format(bot_head,
                                                                         ret, i))
                except Exception as err:
                    echo("{} warning: failed to like topic, {}, count {}".format(bot_head,
                                                                                 err, i))
                rand_delay()
                if not LIVE:
                    raise Exception("exiting")
                if get_egpa:
                    try:
                        EGPA = api.get_egpa(0)
                    except:
                        pass

            rand_delay()
            if not LIVE:
                raise Exception("exiting")
            if get_egpa:
                try:
                    EGPA = api.get_egpa(0)
                except:
                    pass


    except Exception as err:
        echo("{} error: {}".format(bot_head,
                                   err))

    echo("{} info: bot exited".format(bot_head))

    THREADS -= 1


def main():
    global LIVE

    lst_day = 0
    tick = 5
    tip = False

    while True:
        if tick == 5:
            tick = 0
            t = time.time_ns()
            th = t // (1000 * 1000 * 1000 * 60 * 60)
            td = th // 24
            if (th % 24 == 0 and td != lst_day) or lst_day == 0:
                lst_day = td
                start()
                echo("all bots started")
                time.sleep(1)
                tip = True
        try:
            echo("EGPA: {}  Bots: {}  CNN used counts: {}".format(EGPA,
                                                                  THREADS,
                                                                  CAPTCHA_CNN.call_times),
                 True)
            time.sleep(0.5)
            if THREADS == 0:
                if LIVE is False:
                    break
                else:
                    if tip:
                        tip = False
                        echo("[main] info: quests today has been completed, rerun until next 8:00 am")
            tick += 1
        except KeyboardInterrupt:
            LIVE = False
            echo("sending stopping signal...")


def start():
    accounts = load_accounts(ACCOUNT_FILE)
    bot = threading.Thread(target=bot_thread, args=(accounts[0], True))
    bot.start()
    for i in accounts[1:]:
        bot = threading.Thread(target=bot_thread, args=(i,))
        bot.start()
        time.sleep(5)


if __name__ == '__main__':
    main()
