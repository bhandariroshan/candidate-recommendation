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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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


class LogisticRegression:
    
    def fit(self,X,y,alpha=0.01, epocs=2000):
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        self.params = np.zeros(len(X[0]))
        
        for i in range(epocs):
            z = np.dot(X, self.params)
            h = 1 / (1 + np.exp(-z))
            gradient = np.dot(X.T, (h - y)) / y.size
            self.params -= alpha * gradient
            
    def predict(self,X):
        ones = np.ones((len(X), 1))
        X = np.concatenate((ones, X), axis=1)
        z = np.dot(X,self.params)
        return (1 / (1 + np.exp(-z))) >= 0.5


def prepare_file_for_training():
    tags = {}
    count = 0
    for i in list(G_latest.nodes):
        try:
            alpha = tag_count(i)
        except:
            print("Error")
        count += 1
        
        for i in alpha:
            if i in tags:
                tags[i] += alpha[i]
            else:
                tags[i] = alpha[i]
                
    nameList = []
    for key in tags.keys():
        if key is not None:
            nameList.append(key)

    # write a csv file
    import csv
    with open("Desktop/ns github project/company.csv", 'w', newline='') as myfile:
        count = 0
        comp_count = 0
        wr = csv.writer(myfile)
        wr.writerow(['auth_ratio', 'cont_index' ] + nameList + ['result'])
        for i in list(G_latest.nodes()):
            k = str(i)
            if k[0] == 'U':
                if len(G_latest.nodes[i].keys())  > 2:
                    if 'company' in G_latest.nodes[i].keys() and G_latest.nodes[i]['company'] != None:
                        result = 1
                        comp_count += 1

                    elif 'company' in G_latest.nodes[i].keys() and G_latest.nodes[i]['company'] == None:
                        result = 0
                        comp_count += 1

                    else:
                        result = 0
                        comp_count += 1
                else:
                     continue
            
                try:
                    tags_count = tag_count(i)
                    if len(tags_count) > 0: 
                        count += 1
                except:
                    print("skip repocount for ",i)
                try:
                    following_count = G_latest.nodes[i]['following_count']
                    followers_count = G_latest.nodes[i]['followers_count']
                    auth_ratio = int(followers_count/following_count)    
                except:
                    auth_ratio = 0

                cont_index = calculate_contribution_index(i)
                
                
                row = [auth_ratio, cont_index]
                for each_tag in nameList:
                    if each_tag is None:
                        continue
                    if each_tag in tags_count:
                        row.append(tags_count[each_tag])
                    else:
                        row.append(0)
                row.append(result)
                wr.writerow(row)
                print(i)
        print(count)
        print(comp_count)

def tag_count(node):
    neighbour_node = list(graph.neighbors(node))
    list1 = []
    for i in neighbour_node:
        if i[0] == str("R"):
            list1.append(graph.nodes[i]['language'])
    dict1 = {}
    for i in list1:
        if i in dict1:
            dict1[i] += 1
        else:
            dict1[i] = 1
    return dict1

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

def load_graph():
    global graph
    graph=nx.read_gpickle("github2.txt")

def get_praph_properties():
    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=False)
    print("Minimum degree of a graph is: ", degrees[0][1]) 

    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    print("Maximum degree of a graph is: ", degrees[0][1])

    num_nodes = len(graph.nodes)
    print("Number of Nodes", num_nodes)

    num_edges = len(graph.edges)
    print("Number of edges", num_edges)

    print("Average Degree of the graph is: ", num_edges / num_nodes)

    strongest = max([len(c) for c in sorted(nx.strongly_connected_components(graph),key=len, reverse=True)])
    print("Nodes in strongest connected component", strongest)

    num_users = 0
    num_projects = 0

    for each_node in graph.nodes():
        if 'U:' in each_node:
            num_users += 1
        else:
            num_projects += 1

    print("Number of user nodes: ", num_users)
    print("Number of project nodes: ", num_projects)

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

def train_a_model():
    global model
    traincompany = pd.read_csv("company.csv")
    # shuffle(traincompany)

    y_traincompany = traincompany.result
    traincompany.drop(labels=["result"],axis = 1,inplace=True)
    model = LogisticRegression()
    model.fit(traincompany,y_traincompany)

def predict_using_ml(user):
    global model
    global unique_language_list
    try:
        tags_count = tag_count(user)
    except:
       tags_count = {}

    try:
        following_count = graph.nodes[user]['following_count']
        followers_count = graph.nodes[user]['followers_count']
        auth_ratio = int(followers_count/following_count)    
    except:
        auth_ratio = 0
    
    try:
        cont_index = calculate_contribution_index(graph.nodes[user])
    except:
        cont_index = 0
    
    row = [auth_ratio, cont_index]
    for each_tag in unique_language_list:
        if each_tag in tags_count:
            row.append(tags_count[each_tag])
        else:
            row.append(0)

    predict = model.predict([row])
    return predict

def get_user_attributes(each_user, counter):
    user_id = 'U:' + str(each_user['id'])
    try:
        company = each_user['company']
    except:
        company = ''

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

    write_pickle()


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


def print_degree():
    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=False)
    print("Minimum degree of a graph is: ", degrees[0])


def create_adjacency_list_file():
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Source', 'Target'])
        for each_edge in graph.edges:
            writer.writerow([each_edge[0], each_edge[1]])

def calculate_contribution_index(user):
    projects = []
    neighbors = graph.neighbors(user)
    for each_neighbor in neighbors:
        if 'R:' in each_neighbor:
            projects.append(each_neighbor)

    contribution_index = 0
    for each_project in projects:
        edge_data = graph.get_edge_data(str(user), str(each_project))
        if graph.nodes[str(each_project)]['commit_count'] != 0:
            contribution_index += edge_data['commit_count'] / graph.nodes[str(each_project)]['commit_count']
    return contribution_index

def get_popular_language_by_project_count():
    language = {}
    for each_node in graph.nodes:
        lan = graph.nodes[str(each_node)].get('language','')
        if str(lan).lower() != 'none' and str(lan).lower() != '' and str(lan).lower() not in language:
            language[str(lan).lower()] = 1
        elif str(lan).lower() != 'none' and str(lan).lower() != '':
            language[str(lan).lower()] += 1
    
    language_list = []

    for each_language in language:
        language_list.append({'language':each_language, 'count':language[each_language]})

    sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True)  [0:10]
    with open('language_project.csv', 'w', newline='') as file: 
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Language', 'Project Count'])
        for each_language in sorted_language:
            writer.writerow([each_language['language'], each_language['count']])
    return sorted_language

def get_popular_language_by_commit_count():
    language = {}
    for each_node in graph.nodes:
        lan = graph.nodes[str(each_node)].get('language','')
        if str(lan).lower() != 'none' and str(lan).lower() != '' and str(lan).lower() not in language:
            language[str(lan).lower()] = graph.nodes[str(each_node)].get('commit_count',0)
        elif str(lan).lower() != 'none' and str(lan).lower() != '':
            language[str(lan).lower()] += graph.nodes[str(each_node)].get('commit_count',0)
    
    language_list = []

    for each_language in language:
        language_list.append({'language':each_language, 'count':language[each_language]})

    sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True) [0:10]
    with open('language_commit.csv', 'w', newline='') as file: 
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Language', 'Commit Count'])
        for each_language in sorted_language:
            writer.writerow([each_language['language'], each_language['count']])

    return sorted_language

def get_popular_language_by_contributors_count():
    language = {}
    for each_node in graph.nodes:
        lan = graph.nodes[str(each_node)].get('language','')
        if str(lan).lower() != 'none' and str(lan).lower() != '' and str(lan).lower() not in language:
            language[str(lan).lower()] = graph.degree[str(each_node)]

        elif str(lan).lower() != 'none' and str(lan).lower() != '':
            language[str(lan).lower()] += graph.degree[str(each_node)]

    language_list = []

    for each_language in language:
        language_list.append({'language':each_language, 'count':language[each_language]})


    sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True)
    return sorted_language

def find_communities_by_contribution_index():
    tag_count = {}
    for i in graph.nodes():
        if i[0] == 'R':
            ci = 0
            node_data = graph.nodes[i]
            for each_edge in graph.in_edges(i):
                data = graph.get_edge_data(each_edge[0], each_edge[1])
                commit_count = data['commit_count']
                if node_data['commit_count'] > 0:
                    ci += commit_count / node_data['commit_count']
            
            if node_data['language'] not in tag_count:
                tag_count[node_data['language']] = ci
            else:
                tag_count[node_data['language']] += ci

    language_list = []
    for each_language in tag_count.keys():
        if str(each_language).lower() != 'none':
            language_list.append({'language': each_language, 'count': tag_count[each_language]})

    sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True)
    return sorted_language[0:10]

def find_tag_similarity_score(node1, node2, predict_tag=[]):
    tag_similarity = 0
    repos_1 = []
    repos_2 = []
    neighbour_nodes1 = list(graph.neighbors(node1))
    neighbour_nodes2 = list(graph.neighbors(node2))

    for each_node in neighbour_nodes1:
        if 'R' in str(each_node):
            repos_1.append(graph.nodes[each_node]['language'])
    
    for each_node in neighbour_nodes2:
        if 'R' in str(each_node):
            repos_2.append(graph.nodes[each_node]['language'])

    result_sum = 0
    inter = list(set(repos_1).intersection(set(repos_2)))
    for each_tag in repos_1 + repos_1:
        if not predict_tag and each_tag in inter:
            result_sum += 1
        elif each_tag in inter and each_tag in predict_tag:
            result_sum += 1
    try:
        tag_sim = result_sum / len(repos_1 + repos_1)
    except:
        tag_sim = 0

    return tag_sim

def find_follower_similarity(node1, node2):
    follower_similarity = 0
    list1 = []
    list2 = []
    neighbors = list(graph.in_edges(node1))
    for i in neighbors:
        one = str(i).split(',')
        one = one[0]
        list1.append(one[4:-1])
    
    neighbors = list(graph.in_edges(node2))
    for i in neighbors:
        one = str(i).split(',')
        one = one[0]
        list2.append(one[4:-1])
    
    list1_set = set(list1)
    list2_set = set(list2)
    follower_similarity += len(list1_set.intersection(list2_set)) 
    try:
        follower_similarity = follower_similarity / len(list1+list2)
    except:
        follower_similarity = 0

    return follower_similarity

def find_following_similarity(node1, node2):
    following_similarity = 0
    list1 = []
    list2 = []
    neighbors = list(graph.out_edges(node1))
    for i in neighbors:
        one = str(i).split(',')
        one = one[0]
        list1.append(one[4:-1])
    
    neighbors = list(graph.out_edges(node2))
    for i in neighbors:
        one = str(i).split(',')
        one = one[0]
        list2.append(one[4:-1])
    
    list1_set = set(list1)
    list2_set = set(list2)
    following_similarity += len(list1_set.intersection(list2_set))
    try: 
        following_similarity = follower_similarity / len(list1+list2)
    except:
        following_similarity = 0

    return following_similarity

def find_contribution_similarity(node1, node2):
    node1_projects = []
    for each_edge in graph.edges(node1):
        if 'R:' in each_edge[1] :
            node1_projects.append(each_edge[1])
    
    node2_projects = []
    for each_edge in graph.edges(node2):
        if 'R:' in each_edge[1] :
            node2_projects.append(each_edge[1])
    
    contribution_score = 0
    contribution_score_1 = 0
    contribution_score_2 = 0
    # do it for node1
    for project in node1_projects:
        try:
            commit = graph.get_edge_data(node1, project)
            if commit:
                commit_count = commit['commit_count']
                total_commits_in_project = graph.nodes[project]['commit_count']
                if total_commits_in_project > 0:
                    contribution_score_1 += commit_count / total_commits_in_project
        except:
            continue

    # do it for node2
    for project in node2_projects:
        try:
            commit = graph.get_edge_data(node2, project)
            if commit is not None:
                commit_count = commit['commit_count']
                total_commits_in_project = graph.nodes[project]['commit_count']
                if total_commits_in_project > 0:
                    contribution_score_2 += commit_count / total_commits_in_project
        except:
            continue

    try:
        contribution_score = 1 / abs(contribution_score_2-contribution_score_1)
    except:
        pass

    return contribution_score

def categorize_users(users):
    # Categorize good fit
    good_fit = []
    mean = 0
    for each_user in users:
        mean += each_user['goodness_score']

    mean = mean / len(users)
    print("Average goodness score in this pool is: ", mean)
    for each_fit in users:
        if each_fit['goodness_score'] >= mean:
            each_fit['type'] = 'Good'
        else:
            each_fit['type'] = 'Below Average'

        good_fit.append(each_fit)

    return good_fit

def find_score(user, select_neighbors=5, predict_tag=[]):
    return_value = {'user': user, 'goodness_score': 0}
    neighbors = []
    for each_neighbor in graph.neighbors(user):
        if 'U:' in each_neighbor:
            neighbors.append({'neighbor':each_neighbor, 'score': 0})
    
    similarity_score_with_neighbor = []

    for each_neighbor in neighbors:
        contribution_score = find_contribution_similarity(user, each_neighbor['neighbor'])
        tag_similarity_score = find_tag_similarity_score(user, each_neighbor['neighbor'], predict_tag)
        # following_similarity_score = find_following_similarity(user, each_neighbor['neighbor'])
        # follower_similarity_score = find_follower_similarity(user, each_neighbor['neighbor'])
        
        total_score = contribution_score + tag_similarity_score # + following_similarity_score + follower_similarity_score
        
        similarity_score_with_neighbor.append({
            'neighbor': each_neighbor, 'score': total_score, 'contribution_score': contribution_score
        })
    
    # Select to n good neighbors, add their contribution to the network to improve the user's score
    sorted_neighbors = sorted(
        similarity_score_with_neighbor, key = lambda i: i['score'], reverse=True)[0:select_neighbors]
    
    for each_neighbor in sorted_neighbors:
        return_value['goodness_score'] += each_neighbor['score']
    
    return return_value

def prep_data(candidates):
    return_list = []
    for each_candidate in candidates:
        if 'U:' not in str(each_candidate).replace("'",''):
            return_list.append('U:' + str(each_candidate).replace("'",''))
    return return_list

def predict_top_users(candidates=[], predict_size=5, predict_tag=[], select_neighbors=5):
    predict_tag = [tagraph.lower() for tag in predict_tag]
    user_with_score = []
    for each_user in candidates:
        result = find_score(each_user, predict_tag=predict_tag, select_neighbors=select_neighbors)
        user_with_score.append(result)
    
    sorted_users = sorted(user_with_score, key = lambda i: i['goodness_score'], reverse=True)
    categorized_users = categorize_users(sorted_users)
    return_list = []
    for each_data in categorized_users:
        if len(categorized_users) > 10:
            if each_data['goodness_score'] > 0:
                return_list.append(each_data)
        else:
            return_list.append(each_data)

    return return_list[0: predict_size]

def suggest_project_for_user(user):
    node = graph.nodes['U:'+str(user)]
    contribution_in_tag_count = {}
    for each_neighbor in graph.neighbors('U:' + user):
        if 'R:' in each_neighbor:
            repository = graph.nodes[each_neighbor]
            if repository['language'] not in contribution_in_tag_count:
                contribution_in_tag_count[repository['language']] = 1
            else:
                contribution_in_tag_count[repository['language']] += 1
    
    if len(list(contribution_in_tag_count.keys())) > 0:
        # find hub in neighborhood
        largest_tag = list(contribution_in_tag_count.keys())[0]
        for each_tag in contribution_in_tag_count:
            if contribution_in_tag_count[each_tag] > contribution_in_tag_count[largest_tag]:
                largest_tag = each_tag

    else:
        # suggest any hub project
        largest_tag = get_popular_language_by_project_count()[0]['language']

    count = 0
    project = []
    for each_node in graph.nodes():
        repo = graph.nodes[each_node]
        if 'R:' in each_node and largest_tag == str(repo['language']):
            count += 1
            project.append({
                'id': each_node, 
                'commit': repo['commit_count'], 
                'language': repo['language'], 
                'contributors':repo['contributors_count'] 
            })

            if count > 20:
                break

    sorted_projects = sorted(project, key = lambda i: i['commit'], reverse=True)[0:5]
    return sorted_projects

def parse_choices(choice):
    # parser = argparse.ArgumentParser(description='Github Predictor and Analysis System.')
    # parser.add_argument('--help', '-h' help='Action choices can be: 1,2 .. 10, Please refer menu for details. \n ', required=False)

    if str(choice) == '1': # done
        get_praph_properties()

#     elif str(choice) == '2': # done
#         return get_popular_language_by_project_count()[0:10]

#     elif str(choice) == '3': # done
#         return get_popular_language_by_contributors_count()[0:10]

#     elif str(choice) == '4': # done
#         return get_popular_language_by_commit_count()[0:10]

    elif str(choice) == '2': # done
        indata = str(input(" Please Enter 2 users or help:  "))
        if indata == 'help':
            print("Sample User pairs for reference:(32393462, 47487758), (32393462,47487758), (38408226,27479278), (1146706,5253113)")
            users = str(input(" Please Enter 2 users: ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')
        else:
            users = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')

        if not users:
            print("Sorry, Users not entered.")
        else:
            users = prep_data(users)
            
            return find_tag_similarity_score(users[0], users[1], [])

    elif str(choice) == '3':
        indata = str(input(" Please Enter 2 users or help:  "))
        if indata == 'help':
            print("Sample Users for reference:(44335683,28487279), (50693516 , 53458526), (2699235, 32908457), (3971576, 21125224)")
            users = str(input(" Please Enter 2 users:  ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')
        else:
            users = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')

        if not users:
            return "Sorry, Users not entered."
        else:
            users = prep_data(users)
            return find_follower_similarity(users[0], users[1])

    elif str(choice) == '======':
        indata = str(input(" Please Enter 2 users or help:  "))
        if indata == 'help':
            print("Sample User pairs for reference:43542836,43542836         45422842,45422842         5839889,5839889           1607149,1607149      15112911,15112911         30888949,30888949         cariepointer,cariepointer       31701757,31701757          44219625,44219625           9916797,9916797       53843558,53843558")
            users = str(input(" Please Enter 2 users:  ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')
        else:
            users = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')

        if not users:
            return "Sorry, Users not entered."
        else:
            users = prep_data(users)
            return find_following_similarity(users[0], users[1])

    elif str(choice) == '4':
        indata = str(input(" Please Enter 2 users or help:  "))
        if indata == 'help':
            print("Sample Users for reference:(33784424,55615887),(12587522,45255725), (26531373,18627147), (31626581,1420977)")
            users = str(input(" Please Enter 2 users:  ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')
        else:
            users = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')

        if not users:
            return "Sorry, Users not entered."
        else:
            return find_contribution_similarity('U:' + users[0], 'U:'+users[1])

    elif str(choice) == '5':
        # parser.add_argument('--tag', help='Ex: python, java, c, javascript', required=False)
        indata = str(input("Enter users or type help: "))
        if indata == 'help':
            print("Sample Users for reference:'46','22583415','3276478','1657159','2312400','7664514','13069189'")
            users = str(input("Enter users or type help: ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')
        else:
            users = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')

        if not users:
            print("Sorry, Users not entered.")
        else:
            users = prep_data(users)
            return predict_top_users(users)

    elif str(choice) == '6':
        return prepare_data_for_ml_training()

    elif str(choice) == '7':
        indata = str(input("Enter a user or type help: "))
        if indata == 'help':
            print("Sample Users for reference:'46','22583415','3276478','1657159','2312400','7664514','13069189'")
            user = str(input("Enter a user: ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'')
        else:
            user = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'')

        if not user:
            print("Sorry, User not entered.")

        return suggest_project_for_user(user)

    elif str(choice) == '8':
        raise Exception("Exit")

def prepare_data_for_ml_training():
    indata = str(input("Enter users or type help: "))
    # indata = "'46','22583415'"
    result = []
    if indata == 'help':
        print("Sample Users for reference:'46','22583415','3276478','1657159','2312400','7664514','13069189'")
        users = str(input("Enter users or type help:  ")).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')
    else:
        users = str(indata).replace("'",'').replace(" ",'').replace("(",'').replace(")",'').split(',')

    if not users:
        print("Sorry, Users not entered.")
    else:
        for user in users:
            predicted = predict_using_ml('U:'+user)
            if predicted:
                result.append({'user': user, 'type': 'Good'})
            else:
                result.append({'user': user, 'type': 'Not Good'})

    return result

def find_score_data():
    nodes = list()
    for each_node in graph.nodes():
        if 'U:' in str(each_node):
            nodes.append(each_node)

    from random import randint
    count = 0
    while True:
        n1 = nodes[randint(0, len(nodes)-1)]
        n2 = nodes[randint(1, len(nodes)-1)]
        sim_score = find_contribution_similarity(n1, n2)
        if sim_score > 0:
            print("Contibutions: ", n1.replace('U:',''),",",n2.replace('U:',''), sim_score)
            count +=1

        sim_score = find_follower_similarity(n1, n2)
        if sim_score > 0:
            print("Followers: ", n1.replace('U:',''),",",n2.replace('U:',''), sim_score)
            count +=1

        sim_score = find_following_similarity(n1, n2)
        if sim_score > 0:
            print("Following: ", n1.replace('U:',''),",",n2.replace('U:',''), sim_score)
            count +=1

        sim_score = find_tag_similarity_score(n1, n2, [])
        if sim_score > 0:
            print("Tag: ", n1.replace('U:',''),",",n2.replace('U:',''), sim_score)

        if count > 10000:
            break

if __name__ == '__main__':
    print("Loading graph....")
    load_graph()
    print("completed, training prediction model....")
    # train_a_model()
    make_data()
    print("completed...")
    print('********************************************************************\n')
    while True:
        print("Menu:\n")
        print("1:  Graph Properties")
        #  print("2:  Popular Language by Project Count")
        #  print("3:  Popular Language based on Contributor Count")
        #  print("4:  Popular Language based on Commit")
        print("2:  Tags Similarity")
        print("3:  Follower Similarity")
        print("4:  Contribution Similarity")
        print("5:  Predict Good Candidates(based on Similarity and Contribution) ")
        print("6:  Predict Good Candidates(based on Machine Learning)")
        print("7: Suggest project for user")
        print("8: Exit \n")
        print('********************************************************************\n')
        choice = input("Your choice: ")
        print('********************************************************************\n')
        print("Output:\n")
        try:
            outputs = parse_choices(choice)
            if outputs and (type(outputs) == dict or type(outputs) == list):
                for each_output in outputs:
                    print_line = ''
                    for each_key in each_output:
                        print_line += str(each_key) + ': ' + str(each_output[each_key]) + '\t'
                    print(print_line)
            else:
                print("\nScore: ", outputs)

        except Exception as e:
            if "exit" in str(e).lower():
                sys.exit()
            
            print("There is no such user. Please try another. ")

        print("")
        print('********************************************************************\n')