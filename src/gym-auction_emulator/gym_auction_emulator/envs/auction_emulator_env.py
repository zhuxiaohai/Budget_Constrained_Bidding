"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import json
import os
import pandas as pd
import numpy as np
import math

class AuctionEmulatorEnv(gym.Env):
    """
    AuctionEmulatorEnv can be used with Open AI Gym Env and is used to generate
    the bid requests reading the iPinYou dataset files.
    Toy data set with 100 lines are included in the data directory.
    """
    metadata = {'render.modes': ['human']}

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.data_src = cfg['data']['dtype']
        if self.data_src == 'ipinyou':
            self.file_in = cfg['data']['ipinyou_path'][1:-1]
        self.metric = str(cfg['data']['metric'])

    def __init__(self):
        """
        Args:
        Populates the bid requests to self.bid_requests list.
        """
        self._load_config()
        self._step = 1
        fields =    [
                    'weekday',
                    'hour',
                    'auction_type',
                    # 'bidprice',
                    'slotprice',
                    # 'payprice',
                    'click_prob'
                    ]
        self.bid_requests = self._get_data('1458')
        # self.bid_requests = pd.read_csv(self.file_in, sep="\t", usecols=fields)
        self.total_bids = len(self.bid_requests)
        self.bid_line = {}

    def _get_data(self, camp_n):
        """
        This function extracts data for certain specified campaigns
        from a folder in the current working directory.
        :param camp_n: a list of campaign names
        :return: two dictionaries, one for training and one for testing,
        with data on budget, bids, number of auctions, etc. The different
        campaigns are stored in the dictionaries with their respective names.
        """
        data_path = self.file_in
        train_data = pd.read_csv(data_path + '/' + 'train.theta_{}.txt'.format(camp_n),
                                 header=None, index_col=False, sep=' ', names=['click', 'slotprice', 'click_prob'])
        train_data['auction_type'] = 'SECOND_PRICE'
        train_data['hour'] = train_data.index.map(lambda x: math.floor(x / 500))
        train_data['weekday'] = train_data.index.map(lambda x: math.floor((x) / (500*96)) + 1)

        return train_data

    def _get_observation(self, bid_req):
        observation = {}
        if bid_req is not None:
            observation['weekday'] = bid_req['weekday']
            observation['hour'] = bid_req['hour']
            observation['auction_type'] = bid_req['auction_type']
            observation['slotprice'] = bid_req['slotprice']
            observation['click_prob'] = bid_req['click_prob']
        return observation

    def _bid_state(self, bid_req):
        self.auction_type = bid_req['auction_type']
        # self.bidprice = bid_req['bidprice']
        # self.payprice = bid_req['payprice']
        self.click_prob = bid_req['click_prob']
        self.slotprice = bid_req['slotprice']

    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 1
        bid_req = self.bid_requests.iloc[self._step]
        self._bid_state(bid_req)
        # observation, reward, cost, done
        return self._get_observation(bid_req), 0.0, 0.0, False

    def step(self, action):
        """
        Args:
            action: bid response (bid_price)
        Reward is computed using the bidprice to payprice difference.
        """
        done = False
        r = 0.0 # immediate reward
        r_p = 0.0 # temp reward
        c = 0.0 # cost for the bid impression

        if self.metric == 'clicks':
            r_p = self.click_prob
        else:
            raise ValueError(f"Invalid metric type: {self.metric}")

        # mkt_price = max(self.slotprice, self.payprice)
        mkt_price = self.slotprice
        if action > mkt_price:
            if self.auction_type == 'SECOND_PRICE':
                r = r_p
                c = mkt_price
            elif self.auction_type == 'FIRST_PRICE':
                r = r_p
                c = action
            else:
                raise ValueError(f"Invalid auction type: {self.auction_type}")

        next_bid = None
        if self._step < self.total_bids - 1:
            next_bid = self.bid_requests.iloc[self._step]
            self._bid_state(next_bid)
        else:
            done = True

        self._step += 1

        return self._get_observation(next_bid), r, c, done

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
