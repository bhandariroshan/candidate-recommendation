#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
import requests
import json
import networkx as nx
import pickle
import time
from networkx.readwrite import json_graph
import time
from networkx.algorithms.distance_measures import diameter
import csv
import operator
import argparse
import sys
import pandas as pd
import numpy as np

graph = None

companies = [
    'uber',
    'salesforce',
    'vmware',
    'adobe',
    'facebook',
    'microsoft',
    'linkedin',
    'apple',
    'google'

]

processed = []
request_count = 0

credentials = [
    ('username',     'token1'),
    ('username2',    'token2'),
    ('username3',    'token3'),
]

user_link = 'https://api.github.com/orgs/{}/public_members?page={}'
user_api_url = 'https://api.github.com/user/{}'
model = None
unique_language_list = []

def make_data():
    global unique_language_list
    language_tags = {}
    count = 0
    for i in list(graph.nodes):
        try:
            alpha = tag_count(i)
        except:
            print("Error")
        count += 1
        
        for i in alpha:
            if i in language_tags:
                language_tags[i] += alpha[i]
            else:
                language_tags[i] = alpha[i]

    unique_language_list = []
    for key in language_tags.keys():
        if key is not None:
            unique_language_list.append(key)

def prepare_user_list_from_tech_companies():
    return_user_list = []
    for company in companies:
        print(company)
        for i in range(1, 100):
            data_link = user_link.format(company, i)
            users = get_data(data_link)
            for each_user in users:
                each_user['company'] = company

            if len(users) == 0:
                break
            return_user_list += users

            return return_user_list

    return return_user_list

def get_data(url):
    global request_count, start
    request_count += 1
    if request_count % 500 == 0:
        print("Request made: ", request_count)
        timedelta = time.time() - start
        
        if timedelta < 60:
            time.sleep(60-timedelta)

        start = time.time()
        
    data = None
    for credential in credentials:
        data = requests.get(url, auth=(credential[0], credential[1]))
        try:
            data = data.json()
            if type(data) == dict and 'message' in data:
                if 'API rate limit exceeded for' in data['message']:
                    continue
        except Exception as e:
            print(str(e))
        break

    if type(data) == dict and 'message' in data:
        if 'API rate limit exceeded for' in data['message']:
            time.sleep(30)
            return get_data(url)

    return data


def write_pickle():    
    # degrees = sorted(graph.degree, key=lambda x: x[1], reverse=False)
    # print("Minimum degree of a graph is: ", degrees[0])
    pickle.dump(graph, open('github.txt', 'wb'))
    print("Completed Writing Pickle File. (Node, edges) ", len(graph.nodes()), len(graph.edges()))


def get_user_attributes(each_user, counter):
    user_id = 'U:' + str(each_user['id'])
    try:
        company = each_user['company']
    except:
        company = ''

    user_data = get_data('https://api.github.com/users/' + str(each_user['login']))

    try:
        if user_data['following'] > 5000 or             user_data['followers'] > 5000 or             not user_data or             user_data['followers'] == 0 or             user_data['following'] == 0:
            return
    except:
        return

    page = 1
    followers = []
    following = []
    starred_projects = []

    while True:
        foll = get_data(each_user['followers_url'] +'?page={}&page_count=100'.format(page))
        if not foll or len(foll) == 0:
            break

        followers += foll
        page += 1

        if page == 3:
            break

    page = 1
    while True:
        foll = get_data(each_user['following_url'].replace('{/other_user}','') + '?page={}&page_count=100'.format(page))
        if not foll or len(foll) == 0:
            break

        following += foll
        page += 1
        
        if page == 3:
            break

    page = 1
    repos = []
    while True:
        rep = get_data(each_user['repos_url']+ '?page={}&page_count=100'.format(page))
        if not rep or len(rep) == 0:
            break

        page += 1
        repos += rep
        
        if page == 3:
            break

    for each_follower in followers:
        graph.add_edges_from([('U:' + str(each_follower['id']), str(user_id))])
        
    for each_following in following:
        graph.add_edges_from([(user_id,  'U:' + str(each_following['id']))])
    

    graph.add_node(
        str(user_id),
        repos_count=user_data['public_repos'],
        company=company,
        following_count=user_data['following'],
        followers_count=user_data['followers'],
        star_count=len(starred_projects),
        public_gists_count=user_data['public_gists']
    )

    project_followers = []

    for each_repo in repos:
        has_data = True
        page = 1
        commit_count = 0
        user_commit = 0
        contributors_count = 0
        # Need improvement to add edges between users and projects based on contributions url
        while has_data:
            url = each_repo['contributors_url'].replace('', '') + '?page=' + str(page) + '&per_page=100'
            commits = get_data(url)
            if not commits or type(commits) != list or type(each_user) != dict:
                break

            if page > 3:
                break

            for count, each_user_comit in enumerate(commits):

                try:
                    graph.add_edges_from(
                        [('U:' + each_user['login'], 'R:' + str(each_repo['id']))], 
                        commit_count=each_user_comit['contributions']
                    )
                    project_followers.append(
                        {'login': each_user['login'], 'type': each_user_comit['type']}
                    )
                    contributors_count += 1
                    if each_user_comit['login'] == each_user['login']:
                        user_commit = each_user_comit['contributions']

                    commit_count += each_user_comit['contributions']
                except Exception as e:
                    print("except ", str(e))
                    print("each_user", each_user)
                    print("each_commit", each_user_comit)

            page += 1
            if len(commits) ==0:
                has_data = False

        graph.add_node(
            'R:' + str(each_repo['id']),
            watchers_count=each_repo['watchers_count'],
            created_at=each_repo['created_at'],
            contributors_count=contributors_count,
            language=each_repo['language'],
            forks_count=each_repo['forks_count'],
            open_issues_count=each_repo['open_issues_count'],
            commit_count=commit_count
        )

        try:
            graph.add_edges_from([(user_id, 'R:' + str(each_repo['id']))], commit_count=user_commit)
        except:
            print(each_repo)

    write_pickle()
    print('----------------------------------')
    return followers + following + project_followers


def program_start():
    users = prepare_user_list_from_tech_companies()
    counter = 0
    while users:
        counter += 1
        user = users.pop()
        if user['type'].lower() != 'user' or user['login'] in processed:
            continue

        new_users = get_user_attributes(user, counter)
        if new_users:
            users += new_users

        if user['login'] not in processed:
            processed.append(user['login'])



def update_organization():
    for count, each_node in enumerate(graph.nodes()):
        if 'U:' in str(each_node) and ('company' not in graph.nodes[str(each_node)] or not graph.nodes[str(each_node)]['company']):
            try:
                user_api_url = 'https://api.github.com/user/{}'.format(str(each_node).replace('U:','')) 
                data = get_data(user_api_url)
                graph.nodes[str(each_node)]['company'] = data['company']
                print(str(each_node), data['company'], count)
            except:
                user_api_url = 'https://api.github.com/users/{}'.format(str(each_node).replace('U:',''))
                data = get_data(user_api_url)
                try:
                    print(str(each_node), data['company'], count)
                    graph.nodes[str(each_node)]['company'] = data['company']
                except:
                    time.sleep(5)
            

        if count % 500 == 0:
            write_pickle()

    write_pickle()


def prune_graph():
    # Prune the graph
    for each_node in all_nodes:
        if graph.degree(each_node) < 10 and graph.degree(each_node) > 4:
            graph.remove_node(each_node)
    print(len(graph.nodes), len(graph.edges))
    

def run():
    program_start()
    write_pickle()

run()


# In[ ]:




