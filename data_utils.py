import pickle
import numpy as np
import networkx as nx
import time
import random
import gensim
import matplotlib.pyplot as plt
import os
from itertools import groupby
from collections import Counter
random.seed(0)

traj_path = './data/'
groundtruth_path = './data/manually_labels/'
slot = 24
G = nx.read_adjlist(traj_path+"chengdu.adjlist", create_using=nx.DiGraph, nodetype=int)
data_dict_chengdu = pickle.load(open(traj_path+'data_dict_chengdu.pkl', 'rb'), encoding='bytes')

# The following intermediate outputs have been dumped, you can unzip and load them here
SD_pair_data = pickle.load(open(traj_path+'SD_pair_data_'+str(slot), 'rb'), encoding='bytes')
SD_pair_time = pickle.load(open(traj_path+'SD_pair_time_'+str(slot), 'rb'), encoding='bytes')
sample_table = pickle.load(open(traj_path+'SD_pair_'+str(slot)+'_sample_table', 'rb'), encoding='bytes')
transition_proba1 = pickle.load(open(traj_path+'transition_proba_1_'+str(slot), 'rb'), encoding='bytes')
transition_proba2 = pickle.load(open(traj_path+'transition_proba_2_'+str(slot), 'rb'), encoding='bytes')
w2v_toast = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(traj_path+'embed/embedding_epoch9')
groundtruth_table = pickle.load(open(traj_path+'groundtruth_table_'+str(slot), 'rb'), encoding='bytes')
route_to_slot = pickle.load(open(traj_path+'route_to_slot_'+str(slot), 'rb'), encoding='bytes')
SD_reference = pickle.load(open(traj_path+'SD_reference_'+str(slot), 'rb'), encoding='bytes')

def get_signal(partial, references):
    for reference in references:
        for i in range(len(reference) - len(partial) + 1):
            if reference[i: i+len(partial)] == partial:
                return 0
    return 1

def get_initial_labels(route, threshold=0.3):
    label = [0]
    prior = transition_proba1[(route[0], route[-1])]
    for i in range(1, len(route)-1):
        if (route[i-1], route[i]) not in prior:
            label.append(1)
        elif prior[(route[i-1], route[i])] < threshold:
            label.append(1)
        else:
            label.append(0)
    label.append(0)
    return label

def sort_nomal_rep_SD(normal_representation):
    sorted_normal_representation = {}
    for key, value in normal_representation.items():
        sorted_normal_representation[key] = []
        for item in value:
            if len(item) == 0:
                sorted_normal_representation[key].append([])
                continue
            else:
                tmp = []
                for item_ in item:
                    tmp.append(tuple(item_))
                d2 = Counter(tmp)
                sorted_normal_representation[key].append(sorted(d2.items(), key=lambda x: x[1], reverse=True))
    return sorted_normal_representation

def my_fscore_whole_determine(Label_Test, Label_Pred, flag='soft'):
    ground_num, pred_num, correct_num = 0, 0, 0
    for label_test, label_pred in zip(Label_Test, Label_Pred):
        if sum(label_test) == 0 and sum(label_pred) == 0:
            continue
        label_test, label_pred = label_test[1:-1], label_pred[1:-1]
        fun = lambda x: x[1] - x[0]
        listA, listB = [], []
        lstA = [i for i,v in enumerate(label_test) if v==1]
        lstB = [i for i,v in enumerate(label_pred) if v==1]
        for k, g in groupby(enumerate(lstA), fun):
            listA.append([v for i, v in g])
        for k, g in groupby(enumerate(lstB), fun):
            listB.append([v for i, v in g])
        ground_num += len(listA)
        pred_num += len(listB)
        listA = sum(listA, [])
        listB = sum(listB, [])
        if flag == 'strict':
            correct_num += int(listA==listB)
        if flag == 'soft':
            if len(set(listA).union(set(listB))) == 0.0:
                correct_num += 0
            else:
                correct_num += len(set(listA).intersection(set(listB)))/len(set(listA).union(set(listB)))
    if pred_num == 0 or ground_num == 0:
        return 0.0
    else:
        precision = correct_num/pred_num
        recall = correct_num/ground_num
        if precision+recall==0:
            return 0.0
        else:
            F1 = 2*((precision*recall)/(precision+recall))
            return F1

def SD_Data(slot=24, clean=5):
    c = 0
    for key, value in raw_data.items():
        c+=1
        if c%5000 == 0:
            print('process', c, '/', len(raw_data))
        if value['match'] == False or len(value['cpath'])<2:
            continue
        pair = (value['cpath'][0], value['cpath'][-1])
        hour = time.localtime(value['tms'][0]).tm_hour
        if pair in SD_pair_data:
            endidx = value['cpath'].index(pair[1])
            staidx = len(value['cpath'][0:endidx+1]) - 1 -  value['cpath'][0:endidx+1][::-1].index(pair[0])
            route = value['cpath'][staidx:endidx+1]
            route = sorted(set(route),key=route.index)
            if has_the_route_in_G(route) and len(route)>=2:
                SD_pair_data[pair][hour%slot].append(route)
                SD_pair_time[pair][hour%slot].append(value['tms'][-1] - value['tms'][0])
        else:
            SD_pair_data[pair] = [[] for i in range(slot)]
            SD_pair_time[pair] = [[] for i in range(slot)]
            endidx = value['cpath'].index(pair[1])
            staidx = len(value['cpath'][0:endidx+1]) - 1 -  value['cpath'][0:endidx+1][::-1].index(pair[0])     
            route = value['cpath'][staidx:endidx+1]
            route = sorted(set(route),key=route.index)
            if has_the_route_in_G(route) and len(route)>=2:
                SD_pair_data[pair][hour%slot].append(route)
                SD_pair_time[pair][hour%slot].append(value['tms'][-1] - value['tms'][0])
    print('Done SD pair data and start clean')
    collect_clean = []
    for key, value in SD_pair_data.items():
        c = 0 
        for val in value:
            c+=len(val)
        if c < clean:
            collect_clean.append(key)
    for key in collect_clean:
        del SD_pair_data[key]
        del SD_pair_time[key]

def SD_distribution():
    SD_rows, SD_cols = {}, {}
    for key, value in SD_pair_data.items():
        SD_cols[key] = []
        for val in value:
            SD_cols[key].append(len(val))
        SD_rows[key] = sum(SD_cols[key])
    print('Done sample table')
    return SD_rows, SD_cols

def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def SD_sampling():
    SD_rows, SD_cols = sample_table[0], sample_table[1]
    PAIR = list(SD_rows.keys())
    index = random_index(SD_rows.values())
    sample_pair = PAIR[index]
    sample_slot = random_index(SD_cols[sample_pair])
    return sample_pair, sample_slot
    
def verify_distribution():
    A = {}
    for count in range(1000):
        sample_pair, sample_slot = SD_sampling()
        if sample_pair in A:
            A[sample_pair]+=1
        else:
            A[sample_pair]=1
    return A 

def get_average_len(detour_data_batch):
    detour_len= 0
    for c in detour_data_batch:
        detour_len += (c[4]-c[3]+1)
    print('average detour len', detour_len/len(detour_data_batch))
    return detour_len/len(detour_data_batch)

def has_the_route_in_G(route):
    for r in route:
        if r not in G:
            return False
    return True

def is_decision(pre, suf, last_label):
    #output: is_decision, the decision label
    #print('pre, suf, last_label', pre, suf, last_label)
    #one-to-more
    if G.out_degree(pre)>1 and G.in_degree(suf)==1:
        #print('case one-to-more')
        if last_label==1:
            return [False, last_label]
        else:
            return [True, None]
    #one-to-one
    if G.out_degree(pre)==1 and G.in_degree(suf)==1:
        return [False, last_label]
    #more-to-one
    if G.out_degree(pre)==1 and G.in_degree(suf)>1:
        if last_label==0:
            return [False, last_label]
        else:
            return [True, None]
    #more-to-more
    if G.out_degree(pre)>1 and G.in_degree(suf)>1:
        return [True, None]
    return [True, None]

def get_noise_data_with_label_batch(threshold=0.3, batch_size=10, clean=5): #prior+road_network
    noise_data_batch = []
    sd_slot_batch = []
    count = batch_size
    while count != 0 :
        sample_pair, sample_slot = SD_sampling()
        normal_SD_set = SD_pair_data[sample_pair][sample_slot]
        (nr, num) = SD_reference[sample_pair][sample_slot][0]
        if num < clean:
            continue
        seed = random.randint(0, len(normal_SD_set)-1)
        route = normal_SD_set[seed]
        if len(route) >= 2 and has_the_route_in_G(route):
            label = [0] #0 normal 1 abnormal
            prior = transition_proba2[sample_pair][sample_slot] 
            for i in range(1, len(route)-1):
                if (route[i-1], route[i]) not in prior:
                    label.append(1)
                elif prior[(route[i-1], route[i])] < threshold:
                    label.append(1)
                else:
                    label.append(0)
            label.append(0)
            if sum(label) == 0:
                continue
            noise_data_batch.append([route,label])
            sd_slot_batch.append([sample_pair, sample_slot])
            count -= 1
    return noise_data_batch, sd_slot_batch

def load_data(threshold, batch_size, consider=0.2):
    #get noise labelling with prior_knowledge
    label, text, sentence_len, normal_signal = [], [], [], []
    print('threshold', threshold)
    noise_data_batch, sd_slot_batch = get_noise_data_with_label_batch(threshold, batch_size)
    
    for ddb, ssb in zip(noise_data_batch, sd_slot_batch):
        label.append(ddb[1])
        text.append(ddb[0])
        sentence_len.append(len(ddb[1]))
        normal_num = len(SD_pair_data[ssb[0]][ssb[1]])
        normal_reference = SD_reference[ssb[0]][ssb[1]]
        references = []
        for idx in range(len(normal_reference)):
            nr = normal_reference[idx]
            if idx != 0 and nr[1]/normal_num < consider:
                continue
            references.append(list(nr[0]))
            
        # the sliding window with closing gaps
        tmp = [0]*len(ddb[0])
        flagidx = []
        if len(tmp) <= 2:
            normal_signal.append(tmp)
        else:
            for i in range(len(ddb[0])-1):
                sig = get_signal(list(ddb[0][i:i+2]), references)
                tmp[i+1] = sig
                if sig == 0:
                    tmp[i] = 0
                else:
                    flagidx.append(i+1)
            tmp[-1] = 0
            normal_signal.append(tmp)
        
    return label, text, sentence_len, sd_slot_batch, normal_signal

def build_embed(symbols, embed_units):
    print("Loading word vectors...")
    embed = np.zeros([symbols, embed_units], dtype=np.float32)
    for key in w2v_toast.vocab.keys():
        if key == 'PAD' or key == 'MASK':
            continue
        embed[int(key)] = w2v_toast[key]
    return embed

def load_data_groundtruth_sd(clean=2, consider=0.2):
    #get noise labelling with prior_knowledge
    label_pre, text_pre, sentence_len_pre, normal_signal_pre = [], [], [], []
    label_test, text_test, sentence_len_test, normal_signal_test = [], [], [], []
    sd1, sd2 = set(), set()
    for key in groundtruth_table:
        if not has_the_route_in_G(key[2]):
            continue
        sample_slot = route_to_slot[key[2]]
        sample_pair = (key[0], key[1])

        normal_num = len(SD_pair_data[sample_pair][sample_slot])
        normal_reference = SD_reference[sample_pair][sample_slot]
        references = []
        for idx in range(len(normal_reference)):
            nr = normal_reference[idx]
            if idx != 0 and nr[1]/normal_num < consider:
                continue
            references.append(list(nr[0]))
        
        flagidx = []
        if len(key[2]) <= 2:
            tmp = [0]*len(key[2])
        else:
            tmp = [0]*len(key[2])
            for i in range(len(key[2])-1):
                sig = get_signal(list(key[2][i:i+2]), references)
                tmp[i+1] = sig
                if sig == 0:
                    tmp[i] = 0
                else:
                    flagidx.append(i+1)
        if len(sd1) < 100:
            sd1.add(sample_pair)
            label_pre.append(groundtruth_table[key])
            text_pre.append(list(key[2]))
            sentence_len_pre.append(len(key[2]))
            tmp[-1] = 0
            normal_signal_pre.append(tmp)
        else:
            sd2.add(sample_pair)
            label_test.append(groundtruth_table[key])
            text_test.append(list(key[2]))
            sentence_len_test.append(len(key[2]))
            tmp[-1] = 0
            normal_signal_test.append(tmp)
            
    return [label_pre, text_pre, sentence_len_pre, normal_signal_pre], [label_test, text_test, sentence_len_test, normal_signal_test]

def prior_knowledge1():
    transition_proba = {}
    c = 0 
    for key, value in SD_pair_data.items():
        c+=1
        if c%200 == 0:
            print('process', c, '/', len(SD_pair_data))
        transition_proba[key] = {}
        for idx, val in enumerate(value):
            for idx_val in range(len(val)):
                traj = val[idx_val]
                for i in range(len(traj)-1):
                    if (traj[i],traj[i+1]) in transition_proba[key]:
                        transition_proba[key][(traj[i],traj[i+1])]+=1
                    else:
                        transition_proba[key][(traj[i],traj[i+1])]=1
            #count frequent
        for key_ in transition_proba[key]:
            transition_proba[key][key_] /= len(sum(value, []))
            #in case map matching issue for repeatly transiting
            transition_proba[key][key_] = min(transition_proba[key][key_] , 1.0) 
    return transition_proba

def prior_knowledge2():
    transition_proba = {}
    c = 0 
    for key, value in SD_pair_data.items():
        c+=1
        if c%200 == 0:
            print('process', c, '/', len(SD_pair_data))
        transition_proba[key] = [{} for i in range(slot)]
        for idx, val in enumerate(value):
            for idx_val in range(len(val)):
                traj = val[idx_val]
                for i in range(len(traj)-1):
                    if (traj[i],traj[i+1]) in transition_proba[key][idx]:
                        transition_proba[key][idx][(traj[i],traj[i+1])]+=1
                    else:
                        transition_proba[key][idx][(traj[i],traj[i+1])]=1
            #count frequent
            for key_ in transition_proba[key][idx]:
                transition_proba[key][idx][key_] /= len(val)
                #in case map matching issue for repeatly transiting
                transition_proba[key][idx][key_] = min(transition_proba[key][idx][key_] , 1.0) 
    return transition_proba

def vis_paths(paths):
    SD_path = {}
    for path in paths:
        if (path[0],path[-1]) in SD_path:
            SD_path[(path[0],path[-1])].append(path)
        else:
            SD_path[(path[0],path[-1])] = [path]
            
    for key in SD_path:
        G1 = nx.DiGraph()
        for path in SD_path[key]:
            nx.add_path(G1, path)
        plt.figure(figsize=(7, 7))
        plt.title('SD pair: '+str(key)+' #paths: '+str(len(SD_path[key])))
        nx.draw(G1, with_labels=True, node_color='red', node_size=300, font_size=12 ,alpha=0.5)
        plt.show()

def vis_paths_by_SD(paths, slot=None, labelling=False):
    #print('paths',paths)
    G1 = nx.DiGraph()
    for path in paths:
        nx.add_path(G1, path)
    path_weight = {}
    for path in paths:
        for i in range(len(path)-1):
            #print((path[i],path[i+1]))
            if (path[i],path[i+1]) in path_weight:
                path_weight[(path[i],path[i+1])]+=1
            else:
                path_weight[(path[i],path[i+1])]=1
    plt.figure(figsize=(9, 9))
    color_map = []
    color_edge = []
    for node in G1:
        if node == path[0] or node == path[-1]:        
            color_map.append('green')        
        elif transition_proba1[(path[0],path[-1])]:        
            color_map.append('red')
    for edge in G1.edges:
        color_edge.append(path_weight[edge]/len(paths)*5)
    
    check_abnormal = False
    if len(set(path_weight.values()))>1:
        check_abnormal=True
    pos = nx.spring_layout(G1)
    plt.title('SD pair: '+str((path[0],path[-1]))+', Time slot: '+str(slot)+', #paths: '+str(len(paths)))
    nx.draw(G1, pos, with_labels=True, node_color=color_map, width=color_edge, node_size=150, font_size=12, alpha=0.6)
    
    if labelling and check_abnormal:
        prepare_data(plt, path[0], path[-1], slot, paths)
        return -1
    return 0

def prepare_data(plt, S, D, slot, paths):
    save = traj_path+'manually_raw/'+str(S)+'_'+str(D)+'_'+str(slot)+'_'+str(len(paths))+'_'+str(time.time())
    os.makedirs(save)
    plt.savefig(save+'/vis.jpg',dpi=300)
    plt.show()
    path_dict = {}
    for path in paths:
        if tuple(path) in path_dict:
            path_dict[tuple(path)]+=1
        else:
            path_dict[tuple(path)]=1
    #paths = [list(t) for t in set(tuple(_) for _ in paths)]
    c = 0
    is_manually = False
    threshold = 0.3
    if is_manually:
        for path in path_dict:
            f = open(save+'/'+str(c)+'.txt','w')
            f.write('The labelling path occurs {} times in total {} paths \n'.format(path_dict[path],len(paths)))
            for idx in range(len(path)):
                if idx == 0 or idx == len(path)-1:
                    f.write(str(path[idx])+',0'+'\n') #0 normal and 1 abnormal
                else:
                    f.write(str(path[idx])+','+'\n')
            f.close()
            c+=1
    else:
        path_list = sorted(path_dict.items(), key=lambda x: x[1], reverse=True)
        normal = list(path_list[0][0])
        f = open(save+'/'+str(c)+'.txt','w')
        f.write('The labelling path occurs {} times in total {} paths \n'.format(path_list[0][1],len(paths)))
        for idx in range(len(normal)):
            f.write(str(normal[idx])+',0'+'\n') #0 normal and 1 abnormal
        f.close()
        c+=1
        for idx in range(1, len(path_list)):
            route, fre = path_list[idx][0], path_list[idx][1]
            f = open(save+'/'+str(c)+'.txt','w')
            f.write('The labelling path occurs {} times in total {} paths \n'.format(fre,len(paths)))
            if fre/len(paths) > threshold:
                for idx_ in range(len(route)):
                    f.write(str(route[idx_])+',0'+'\n') #0 normal and 1 abnormal
            else:
                label = [0]*len(route)
                for i in range(len(route)):
                    if normal[i] != route[i]:
                        detourS = i
                        break
                for i in range(len(route)):
                    if normal[::-1][i] != route[::-1][i]:
                        detourD = len(route)-1-i
                        break
                label[detourS:detourD+1] = [1]*(detourD-detourS+1)
                for idx_ in range(len(route)):
                    f.write(str(route[idx_])+','+str(label[idx_])+'\n') #0 normal and 1 abnormal
            f.close()
            c+=1

def get_file_path(root_path,file_list,dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)
    return file_list, dir_list

def obtain_groundtruth():
    file_list = []
    dir_list = []
    groundtruth_table = {}
    file_list, dir_list = get_file_path(groundtruth_path, file_list, dir_list)
    NUM, NUM_detour = 0, 0
    abroads, roads = 0, 0
    for fl in file_list:
        tmp = fl.split(groundtruth_path)[1]
        if '.txt' not in tmp:
            continue
        tmp_ = tmp.split('_')
        S, D, slot, num = int(tmp_[0]), int(tmp_[1]), int(tmp_[2]), int(tmp_[3])
        NUM+=num
        print('fl', fl)
        f = open(fl)
        line_count = 0
        traj, label = [],[]
        line_tmp = ''
        for line in f:
            line_count+=1
            if line_count == 1:
                line_tmp = line
                continue
            temp = line.strip().split(',')
            traj.append(int(temp[0]))
            label.append(int(temp[1]))
        if sum(label)>0:
            NUM_detour+=int(line_tmp.split(' ')[4])
            abroads += (sum(label)*int(line_tmp.split(' ')[4]))
        roads += (len(traj)*num)
        f.close()
        groundtruth_table[(S,D,tuple(traj))] = label
    print('There are {} detours (with {} abnormal raods) in total {} paths (with {} roads)'.format(NUM_detour,abroads,NUM,roads))
    return groundtruth_table

def refine_groundtruth(groundtruth_table):
    re_groundtruth_table = {}
    record = []
    for key, label in groundtruth_table.items():
        route = key[2]
        if has_the_route_in_G(tuple(route)):
            relabel = [0]
            for i in range(1, len(route)-1):
                is_dec, tag = is_decision(route[i-1], route[i], relabel[i-1])
                if is_dec:
                    relabel.append(label[i])
                else:
                    relabel.append(tag)    
            relabel.append(0)
            re_groundtruth_table[key] = relabel
            if relabel != label:
                record.append([key, label, relabel])
        else:
            print('not in G')
            re_groundtruth_table[key] = label
    return re_groundtruth_table, record

if __name__ == '__main__':
    print('Data preprocessing is here, you may release the comments below to generate the intermediate outputs step-by-step!')
    '''
    #Preprocess 1: obtain locations and durations for SD-Pairs
    SD_pair_data, SD_pair_time = {}, {}
    for name in ['20161101.pickle', '20161102.pickle', '20161103.pickle', '20161104.pickle', '20161105.pickle']:
        raw_data = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        SD_Data(slot)
        print('len(SD_pair_data), len(SD_pair_time)', len(SD_pair_data), len(SD_pair_time))
    pickle.dump(SD_pair_data, open(traj_path+'SD_pair_data_'+str(slot), 'wb'), protocol=2)
    pickle.dump(SD_pair_time, open(traj_path+'SD_pair_time_'+str(slot), 'wb'), protocol=2)
    SD_rows, SD_cols = SD_distribution()
    sample_table = (SD_rows, SD_cols)
    pickle.dump(sample_table, open(traj_path+'SD_pair_'+str(slot)+'_sample_table', 'wb'), protocol=2)    
    #verify distribution sampling
    verifier = verify_distribution()
    
    #Preprocess 2: obtain normal route references
    SD_reference = sort_nomal_rep_SD(SD_pair_data)
    pickle.dump(SD_reference, open(traj_path+'SD_reference_'+str(slot), 'wb'), protocol=2)
    
    #Preprocess 3: obtain transition probability table
    transition_proba1 =  prior_knowledge1()
    transition_proba2 =  prior_knowledge2()
    pickle.dump(transition_proba1, open(traj_path+'transition_proba_1_'+str(slot), 'wb'), protocol=2)
    pickle.dump(transition_proba2, open(traj_path+'transition_proba_2_'+str(slot), 'wb'), protocol=2)
    
    #Preprocess 4: obtain SD-Pairs for manually labeling
    batch, _ = get_noise_data_with_label_batch(0.4, 200)
    paths = []
    c = 0
    for idx, path in enumerate(batch):
        paths.append(path[0])
        if 1 in path[1]:
            print(idx, path)
            c+=1
            continue
    print(c)
    vis_paths(paths)
    manual = 20
    while manual:
        sample_pair, sample_slot = SD_sampling()
        manual += vis_paths_by_SD(SD_pair_data[sample_pair][sample_slot], sample_slot, True) #[[],[],[]]
        print('#', sample_pair, sample_slot, len(SD_pair_data[sample_pair][sample_slot]))
    
    #Preprocess 5: load manually labels after double check
    Map, clean, route_to_slot = [], 5, {}
    manual = 200
    while manual:
        sample_pair, sample_slot = SD_sampling()
        if sample_pair in Map:
            continue
        tmp = []
        for i in range(len(SD_pair_data[sample_pair])):
            if len(SD_reference[sample_pair][i]) == 0:
                continue
            (nr, num) = SD_reference[sample_pair][i][0]
            if num < clean:
                continue
            tmp += SD_pair_data[sample_pair][i]
            for route in SD_pair_data[sample_pair][i]:
                route_to_slot[tuple(route)] = i
        if len(tmp) == 0:
            continue
        print('#', len(tmp))
        flag = vis_paths_by_SD(tmp, sample_slot, labelling=True)
        manual += flag
        if flag == -1:
            Map.append(sample_pair)
    pickle.dump(Map, open(traj_path+'labelled_SD_'+str(slot), 'wb'), protocol=2)
    pickle.dump(route_to_slot, open(traj_path+'route_to_slot_'+str(slot), 'wb'), protocol=2)
    groundtruth_table = obtain_groundtruth()
    groundtruth_table, record = refine_groundtruth(groundtruth_table)
    pickle.dump(groundtruth_table, open(traj_path+'groundtruth_table_'+str(slot), 'wb'), protocol=2) 
    '''