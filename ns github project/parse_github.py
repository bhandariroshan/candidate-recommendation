import csv
import requests
import json
import networkx as nx
import pickle
import time
from networkx.readwrite import json_graph
import time
from networkx.algorithms.distance_measures import diameter

graph = nx.DiGraph()

user_link = 'https://api.github.com/users?since='

processed = []
request_count = 0

credentials = [
]

start = time.time()


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
    print("Completed Writing Pickle File. (Node, edges, diameter) ", len(graph.nodes), len(graph.edges))


def get_user_attributes(each_user, counter):
    user_id = 'U:' + str(each_user['id'])
    user_data = get_data('https://api.github.com/users/' + str(each_user['login']))

    try:
        if user_data['following'] > 5000 or \
            user_data['followers'] > 5000 or \
            not user_data or \
            user_data['followers'] == 0 or \
            user_data['following'] == 0:
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

    page = 1
    while True:
        foll = get_data(each_user['following_url'].replace('{/other_user}','') + '?page={}&page_count=100'.format(page))
        if not foll or len(foll) == 0:
            break

        following += foll
        page += 1

    page = 1
    while True:
        star = get_data(each_user['starred_url'].replace('{/owner}{/repo}','') + '?page={}&page_count=100'.format(page))
        if not star or len(star) == 0:
            break

        starred_projects += star
        page += 1

    page = 1
    repos = []
    while True:
        rep = get_data(each_user['repos_url']+ '?page={}&page_count=100'.format(page))
        if not rep or len(rep) == 0:
            break

        page += 1
        repos += rep

    # print("Userid, followers", str(user_id), len(followers))
    # print("Userid, following", str(user_id), len(following))
    # print("Userid, Repos", str(user_id), len(repos))

    for each_follower in followers:
        graph.add_edges_from([('U:' + str(each_follower['id']), str(user_id))])
        
    for each_following in following:
        graph.add_edges_from([(user_id,  'U:' + str(each_following['id']))])
    
    graph.add_node(
        str(user_id),
        repos_count=user_data['public_repos'],
        following_count=user_data['following'],
        followers_count=user_data['followers'],
        star_count=len(starred_projects),
        public_gists_count=user_data['public_gists']
    )

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

            for each_user_comit in commits:
                try:
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
    return followers + following

def program_start():
    for i in range(0, 1):
        new_link = user_link + str(i)
        users = get_data(new_link)
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
            if len(processed) > 5:
                break

# program_start()
# # print(len(get_data('https://api.github.com/repos/bmizerany/amazon-ec2/contributors')))
# write_pickle()
