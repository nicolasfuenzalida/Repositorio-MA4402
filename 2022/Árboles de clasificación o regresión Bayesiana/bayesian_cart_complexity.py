
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets, svm, metrics

import numpy as np
import re
import copy
import graphviz
import scipy as sp  
import matplotlib.pyplot as plt
import dot2tex
import pydotplus

from sklearn.metrics import accuracy_score

from scipy.special import gamma, factorial
from scipy.special import gammaln
from decimal import Decimal
from scipy import special
from numpy import exp


## https://github.com/jpnevrones/Decision-Tree-CART-/blob/master/DecisionTree/DecisionTree.py
# https://github.com/jpnevrones/Decision-Tree-CART-


#### Random State seed on Beta

class DecisionTree(object):
    """
    Class to create decision tree model (CART)
    """
    def __init__(self, alpha = 0.5, beta = 0.5, prob = [0.25]*4, random_seed = None):
        self.alpha = alpha
        self.beta = beta
        self.name = "10X"
        self.nodes = [0]
        self.prob = prob
        self.seed = random_seed
        #self.depth = 1

    def copy(self):
        new_tree = DecisionTree(alpha= self.alpha, beta = self.beta)
        new_tree.alpha = self.alpha
        new_tree.beta = self.beta
        new_tree.name = self.name
        new_tree.feature = self.feature
        new_tree.label = self.label 
        new_tree.train_data = self.train_data
        new_tree.categories = self.categories
        new_tree.colors_rgb = self.colors_rgb
        new_tree.root = copy.deepcopy(self.root)
        new_tree.nodes = self.nodes
        new_tree.shape = self.shape
        new_tree.feature_names = self.feature_names
        new_tree.label_names = self.label_names
        new_tree.alpha_prior = self.alpha_prior
        new_tree.seed = self.seed
        return new_tree

    def info(self, name = None):
        if name is None:
            name = self.name
        #Split
        matches0 = re.finditer("([X0])",name)
        matches0 = list(matches0)
        
        #Numero de splits posibles (Nodos hoja)
        N_split = len(matches0)

        #Numero de change posibles (Nodo interno)
        matches2 = re.finditer("([^0X])",name)
        matches2 = list(matches2)
        N_change = len(matches2)
        #Profundidad
        depth = 0
        
        rand_grow_indexes=[match0.start() for match0 in matches0]
        
        raw_paths = [name[:rand_grow_index] for rand_grow_index in rand_grow_indexes]
        for raw_path in raw_paths:
            raw_path
            path = self.get_path(raw_path)
            l = len(path)
            if l > depth:
                depth = l
                
        return [depth, N_split,N_change] 
            
        
        N_change = len(matches2)
        return np.array([N_split, N_prune, N_change, N_swap])

    def leaves_num(self, name = None):
        if name is None:
            name = self.name
        #Split
        matches0 = re.finditer("([X0])",name)
        matches0 = list(matches0)
        
        #Numero de splits posibles (Nodos hoja)
        N_split = len(matches0)

        return N_split
            
        
        N_change = len(matches2)
        return np.array([N_split, N_prune, N_change, N_swap])
    
    
    def fit(self, _feature, _label):
        """

        :param _feature:
        :param _label:
        :return:
        """
        self.feature = _feature
        self.label = _label
        
        self.train_data = np.column_stack((self.feature,self.label))
        self.build_tree()
    
    
    def fit_MC(self, _feature, _label, alpha_prior=None):
        """

        :param _feature:
        :param _label:
        :return:
        """
        #Data
        self.feature = _feature
        self.label = _label
        self.categories = np.unique(_label)
        self.train_data = np.column_stack((self.feature,self.label))
        self.shape = ( np.shape(self.feature)[-1] , len(self.categories) ) 
        if alpha_prior is None:
            self.alpha_prior = np.array([1]* (self.shape[1]))
        elif self.shape[1] == len(alpha_prior):
            self.alpha_prior = alpha_prior
        else:
            print("Mal Dimension de 'alpha_prior' ")
            self.alpha_prior = None
            
        
        #Estetica
        self.feature_names = [f"x_{x}" for x in range(self.shape[0])]
        self.label_names = [f"c_{x}" for x in range(self.shape[1])]
        self.colors_rgb = self._color_brew(self.shape[1])
        
        self.build_tree_MC()
        
    def reset_labels(self):
        self.feature_names = [f"$x_{x}$" for x in range(self.shape[0])]
        self.label_names = [f"$c_{x}$" for x in range(self.shape[1])]
        
    def set_labels(self, feature_names,label_names):
        if ( (len(feature_names), len(label_names) ) == self.shape):
            self.feature_names = feature_names
            self.label_names = label_names
        else:
            print("Dimensión incorrecta")


    def compute_gini_similarity(self, groups, class_labels):
        """
        compute the gini index for the groups and class labels

        :param groups:
        :param class_labels:
        :return:
        """
        num_sample = sum([len(group) for group in groups])
        gini_score = 0

        for group in groups:
            size = float(len(group))

            if size == 0:
                continue
            score = 0.0
            for label in class_labels:
                porportion = (group[:,-1] == label).sum() / size
                score += porportion * porportion
            gini_score += (1.0 - score) * (size/num_sample)

        return gini_score
    
    def p_split(self, alpha, altura, beta):
        return alpha*(1+altura)**(-beta)

    def terminal_node(self, _group):
        """
        Function set terminal node as the most common class in the group to make prediction later on
        is an helper function used to mark the leaf node in the tree based on the early stop condition
        or actual stop condition which ever is meet early
        :param _group:
        :return:
        """
        class_labels, count = np.unique(_group[:,-1], return_counts= True)
        return class_labels[np.argmax(count)]

    def split(self, index, val, data):
        """
        split features into two groups based on their values
        :param index:
        :param val:
        :param data:
        :return:
        """
        data_left = np.array([]).reshape(0,self.train_data.shape[1])
        data_right = np.array([]).reshape(0, self.train_data.shape[1])

        for row in data:
            if row[index] <= val :
                data_left = np.vstack((data_left,row))

            if row[index] > val:
                data_right = np.vstack((data_right, row))

        return data_left, data_right

    def best_split(self, data):
        """
        find the best split information using the gini score
        :param data:
        :return best_split result dict:
        """
        class_labels = np.unique(data[:,-1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        for idx in range(data.shape[1]-1):
            for row in data:
                groups = self.split(idx, row[idx], data)
                gini_score = self.compute_gini_similarity(groups,class_labels)

                if gini_score < best_score:
                    best_index = idx
                    best_val = row[idx]
                    best_score = gini_score
                    best_groups = groups
        result = {}
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        return result



    def insert(self, source_str, insert_str, pos):
        return source_str[:pos] + insert_str + source_str[pos:]

    def split_MC(self, data, n=0):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        if n>30:
            print("No se pudo hacer el primer split")
            return

        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
            
        class_labels = np.unique(data[:,-1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        idx = np.random.choice(range(data.shape[1]-1))
        val_aux = np.random.choice(np.unique(data[:,idx]))
        groups = self.split(idx, val_aux, data)
        gini_score = self.compute_gini_similarity(groups,class_labels)


        if True : #gini_score < best_score:
            best_index = idx
            best_val = val_aux
            best_score = gini_score
            best_groups = groups
        result = {}
        result["depth"] = 1
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        left_node, right_node = best_groups
        if len(left_node) == 0 or len(right_node)==0:
            result = self.split_MC(data,n+1)
            return result
        result["left"] = self.terminal_node(left_node)
        result["right"] = self.terminal_node(right_node)
        
        #Agregar if len(groups)==0, para que no haga vacio.
        
        n_ik = self.compute_n_ik(best_groups)
        result["n_ik"] = n_ik

        

        return result


    def compute_alpha_score(self):
        return 0.5

    def split_MC2(self, U, node, data):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        
        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
            
        class_labels = np.unique(data[:,-1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        idx = np.random.choice(range(data.shape[1]-1))
        val_aux = np.random.choice(np.unique(data[:,idx]))
        groups = self.split(idx, val_aux, data)
        gini_score = self.compute_gini_similarity(groups,class_labels)

        alpha_score = self.compute_alpha_score()

        if U < alpha_score:
            best_index = idx
            best_val = val_aux
            best_score = gini_score
            best_groups = groups
            result = {}
            result["depth"] = node["depth"]+1
            result['index'] = best_index
            result['val'] = best_val
            result['groups'] = best_groups
            left_node, right_node = best_groups
            result["left"] = self.terminal_node(left_node)
            result["right"] = self.terminal_node(right_node)
            node["left"] = result


    def light_root(self, tree = None):
        
        if tree is None:
            tree_aux = copy.deepcopy(self.root)
            del(tree_aux["groups"])
            tree_aux["left"] = self.light_root(tree_aux["left"])
            tree_aux["right"] = self.light_root(tree_aux["right"])
            return tree_aux
        else:
            tree_aux = tree
            if isinstance(tree_aux , dict):
                del(tree_aux["groups"])
                tree_aux["left"] = self.light_root(tree_aux["left"])
                tree_aux["right"] = self.light_root(tree_aux["right"])
                return tree_aux
            else:
                return tree
                

    def get_node_by_path(self, tree, string):
        if len(string)==0:
          return tree
        path = string[0]
        if path == 1:
          node = tree["left"]
        if path == 2:
          node = tree["right"]
        return self.get_node_by_path(node,string[1:])
                                     

    # def set_node_by_path(self, tree, string):
    #     if len(string)==0:
    #       return tree
    #     path = string[0]
    #     if path == 1:
    #       node = tree["left"]
    #     if path == 2:
    #       node = tree["right"]
    #     return self.get_node_by_path(node,string[1:])

    def get_path(self, string):
        string_aux=string
        while len(re.findall("12*0",string_aux))>0:
          string_aux = re.sub("12*0","2", string_aux)
        return string_aux

    def _color_brew(self, n):
        """Generate n colors with equally spaced hues.
        Parameters
        ----------
        n : int
            The number of colors required.
        Returns
        -------
        color_list : list, length n
            List of n tuples of form (R, G, B) being the components of each color.
        """
        color_list = []

        # Initialize saturation & value; calculate chroma & value shift
        s, v = 0.75, 0.9
        c = s * v
        m = v - c

        for h in np.arange(25, 385, 360.0 / n).astype(int):
            # Calculate some intermediate values
            h_bar = h / 60.0
            x = c * (1 - abs((h_bar % 2) - 1))
            # Initialize RGB with same hue & chroma as our color
            rgb = [
                (c, x, 0),
                (x, c, 0),
                (0, c, x),
                (0, x, c),
                (x, 0, c),
                (c, 0, x),
                (c, x, 0),
            ]
            r, g, b = rgb[int(h_bar)]
            # Shift the initial RGB values to match value and store
            rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
            color_list.append(rgb)

        return color_list
        
    
    def color_alpha(self, alpha, color):
        # compute the color as alpha against white
        color0 = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color0)
    
    
    def compute_n_ik(self, grupos):
        left_node, right_node = grupos
        categories = self.categories.reshape(-1,1)
        n_ik = ((left_node[:,-1] == categories).sum(axis=1), (right_node[:,-1] == categories).sum(axis=1))
        return n_ik


    def split_MC3(self):

        #Cambiar para  que no se pueda elegir split que generan conjuntos vacios( sale en el paper que no se pueden generar vacios)
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
        
        name = self.name
        #Eligimos split aleatorio
        matches = re.finditer("([X0])",name)
        matches = list(matches)
        # Cambiar con respecto a la altura
        rand_grow_match_index=np.random.choice(range(len(matches)))
        rand_grow_index=matches[rand_grow_match_index].start()
        
        raw_path = name[:rand_grow_index]
        path = self.get_path(raw_path)
        last_order = path[-1]
        path = path[:-1]
        
        tree = self.root
        while  len(path)>0:
            order = path[0]
            if order == "1":
                tree = tree["left"]
            if order == "2":
                tree = tree["right"]
            path = path[1:]


        if last_order=="1": 
            data = tree['groups'][0]
        elif last_order=="2":
            data = tree['groups'][1]

        father_depth = tree["depth"]
        class_labels = np.unique(data[:,-1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        idx = np.random.choice(range(data.shape[1]-1))
        val_aux = np.random.choice(np.unique(data[:,idx]))
        groups = self.split(idx, val_aux, data)
        gini_score = self.compute_gini_similarity(groups,class_labels)


        if True : #gini_score < best_score:
            best_index = idx
            best_val = val_aux
            best_score = gini_score
            best_groups = groups
        result = {}
        result["depth"] = father_depth+1
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        left_node, right_node = best_groups


        if len(right_node)==0 or len(left_node)==0:
            if not seed is None:
                self.seed = np.random.get_state()[1][0]
            self.split_MC3()
            return
        result["left"] = self.terminal_node(left_node)
        result["right"] = self.terminal_node(right_node)

        
        if last_order=="1":
            tree["left"] = result
        elif last_order=="2":
            tree["right"] = result

        #Cambiamos el nombre
        self.name = self.insert(name,"10",rand_grow_index)
        

    def random_choice(self, array):
        # "Given a estocastical vector choose value based on U uniform dist"
        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
        
        p_acum=0
        U = np.random.uniform(0,1)
        for i, p in enumerate(array):
            p_acum += p
            if U <= p_acum:
                return i
        return i
        
    def get_num_operation(self, name = None):
        if name is None:
            name = self.name
        #Split
        matches0 = re.finditer("([X0])",name)
        matches0 = list(matches0)
        
        #Numero de splits posibles
        N_split = len(matches0)
        
        
        matches1 = re.finditer("10([X0])",name)
        matches1 = list(matches1)
        
        #Numero de prunes posibles
        if name == "10X":
            N_prune = 0
        else:
            N_prune = len(matches1)
        
        #Elegiir no interno no pruneable (tiene como hijo al menos un nodo interno)

        matches2 = re.finditer("([^0X])",name)
        matches2 = list(matches2)
        
        matches1_index = [match.start() for match in matches1]
        matches2_index = [match.start() for match in matches2]
        matches3_index = [x for x in matches2_index if x not in matches1_index]
        
         #Numero de swaps posibles
        N_swap = len(matches3_index)
        
        N_change = len(matches2)
        return np.array([N_split, N_prune, N_change, N_swap])
        
    def split_MC4(self, n = 0):
        alpha = self.alpha
        beta = self.beta
        prob = np.array(self.prob)

        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
        
        #Cambiar para  que no se pueda elegir split que generan conjuntos vacios( sale en el paper que no se pueden generar vacios)
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        name = self.name
        #Eligimos split aleatorio
        matches = re.finditer("([X0])",name)
        matches = list(matches)
        
        #Numero de splits posibles
        N_split = len(matches)
        
        # Calculamos las probabilidades de  p_split
        rand_grow_indexes=[match.start() for match in matches]
        
        raw_paths = [name[:rand_grow_index] for rand_grow_index in rand_grow_indexes]
        paths = [self.get_path(raw_path) for raw_path in raw_paths]
        p_splits = np.array([self.p_split(alpha, len(path), beta ) for path in paths])
        sum_p_splits = sum(p_splits)
        p_split_stand = p_splits/sum_p_splits
        
        p_splits_mean = p_splits.mean()
        
        
#        print(f"vector split stand: {p_split_stand} ")
#         print(f"suma vector stand : {sum(p_split_stand)} ")
        choice_path_index = self.random_choice(p_split_stand)
        
        #Elegimos el indice con los pesos de p_split
#         print(f"eleccion : {choice_path_index} ")
        path = paths[choice_path_index]
        
        last_order = path[-1]
        path = path[:-1]
        
        tree = self.root
        while  len(path) > 0:
            order = path[0]
            if order == "1":
                tree = tree["left"]
            if order == "2":
                tree = tree["right"]
            path = path[1:]


        if last_order=="1": 
            data = tree['groups'][0]
        elif last_order=="2":
            data = tree['groups'][1]

        father_depth = tree["depth"]
        class_labels = np.unique(data[:,-1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        idx = np.random.choice(range(data.shape[1]-1))
        val_aux = np.random.choice(np.unique(data[:,idx]))
        groups = self.split(idx, val_aux, data)
        gini_score = self.compute_gini_similarity(groups,class_labels)

        if True : #gini_score < best_score:
            best_index = idx
            best_val = val_aux
            best_score = gini_score
            best_groups = groups
        result = {}
        result["depth"] = father_depth+1
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        left_node, right_node = best_groups

        #Agregue esta linea
        if len(right_node)==0 or len(left_node)==0:
            if n>300:
                return
            if not seed is None:
                self.seed = np.random.get_state()[1][0]
            self.split_MC4(n=n+1)
            
            
            return
        result["left"] = self.terminal_node(left_node)
        result["right"] = self.terminal_node(right_node)
        n_ik = self.compute_n_ik(best_groups)
        result["n_ik"] = n_ik


        
        if last_order=="1":
            tree["left"] = result
        elif last_order=="2":
            tree["right"] = result

        #Cambiamos el nombre
        
        rand_grow_index = rand_grow_indexes[choice_path_index]
        new_name = self.insert(name,"10",rand_grow_index)
        matches_prune = re.finditer("10([X0])",new_name)
        matches_prune = list(matches_prune)

        
        ####################
        
        #Numero de prunes (T*) posibles
        N_prune = len(matches_prune)

        # Calculamos las probabilidades de  prune
        rand_prune_indexes = [match_prune.start() for match_prune in matches_prune]
        raw_paths_prune = [new_name[:rand_prune_index] for rand_prune_index in rand_prune_indexes]
        paths_prune = [self.get_path(raw_path_prune) for raw_path_prune in raw_paths_prune]
        p_prunes = 1 - np.array([self.p_split(alpha, len(path), beta ) for path in paths])
        
        p_prunes_mean = p_prunes.mean()
        
        N_Ti = self.get_num_operation(name)
        N_Tf = self.get_num_operation(new_name)
        
        B_tree_split = (p_splits_mean/p_prunes_mean) * ((N_Ti*prob).sum()/((N_Tf*prob).sum()))  *  (prob[1] / prob[0]) * ( 1 - self.p_split(alpha,len(path)+1,beta))**2
        self.name = new_name
        
        return B_tree_split
    
    def P_YX(self):
        alphal = self.alpha_prior
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        name = self.name
        #Eligimos split aleatorio
        matches = re.finditer("([X0])",name)
        matches = list(matches)
        b = len(matches)
        
        # Calculamos las probabilidades de  p_split
        rand_grow_indexes=[match.start() for match in matches]
        n = np.zeros((b,len(self.categories)))
        
        raw_paths = [name[:rand_grow_index] for rand_grow_index in rand_grow_indexes]
        paths = [self.get_path(raw_path) for raw_path in raw_paths]
        for i, path in enumerate(paths):
            last_order = path[-1]
            path = path[:-1]
            
            tree = self.root
            while  len(path) > 0:
                order = path[0]
                if order == "1":
                    tree = tree["left"]
                if order == "2":
                    tree = tree["right"]
                path = path[1:]

            if last_order=="1": 
                data = tree['n_ik'][0]
            elif last_order=="2":
                data = tree['n_ik'][1]
            n[i,:] = data

        pt = b * (gammaln(np.sum(alphal)) - np.sum(gammaln(alphal)))
        alphal1 = alphal.reshape((-1, len(self.categories)))
        a = np.sum( np.sum(gammaln(n + alphal1), axis = 1)  - gammaln(np.sum(n, axis =1) + np.sum(alphal)), axis = 0 )   
        return a + pt

    def extract (self, source_str, len_extract_str, pos):
        return source_str[:pos] + source_str[pos+len_extract_str:]

    def prune_MC3(self):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        alpha = self.alpha
        beta = self.beta
        prob = np.array(self.prob)
        
        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
        
        name = self.name
        if name == "10X":
            print("Árbol minimo no se puede podar")
            N_prune = 0
            return
        #Elegimos prune aleatorio
        matches = re.finditer("10([X0])",name)
        matches = list(matches)
        
        ####################
        
        #Numero de splits posibles
        N_prune = len(matches)
        
        # Elejimos prune aleatorio
        # Calculamos las probabilidades de  prune
        rand_prune_indexes=[match.start() for match in matches]
        
        raw_paths = [name[:rand_prune_index] for rand_prune_index in rand_prune_indexes]
        paths = [self.get_path(raw_path) for raw_path in raw_paths]
        p_prunes = 1 - np.array([self.p_split(alpha, len(path), beta ) for path in paths])
        sum_p_prunes = sum(p_prunes)
        p_prune_stand = p_prunes/sum_p_prunes
        
        p_prunes_mean = p_prunes.mean()
        
        
#        print(f"vector prune stand: {p_prune_stand} ")
        choice_path_index = self.random_choice(p_prune_stand)
        path = paths[choice_path_index]

        last_order = path[-1]
        path_original = path
        path = path[:-1]
        
        
        tree = self.root
        while  len(path)>0:
            order = path[0]
            if order == "1":
                tree = tree["left"]
            if order == "2":
                tree = tree["right"]
            path = path[1:]

        if last_order=="1": 
            data = tree['groups'][0]
            tree["left"] = self.terminal_node(data)
        elif last_order=="2":
            data = tree['groups'][1]
            tree["right"] = self.terminal_node(data)
            
        # Calculamos las probabilidades de  p_split
        
        
        rand_prune_index =  rand_prune_indexes[choice_path_index]
        new_name = self.extract(name,2,rand_prune_index)
        
        #Eligimos split
        matches_split = re.finditer("([X0])",new_name)
        matches_split = list(matches_split)
        
        rand_grow_indexes=[match_split.start() for match_split in matches_split]
        
        raw_paths_split = [new_name[:rand_grow_index] for rand_grow_index in rand_grow_indexes]
        paths_split  = [self.get_path(raw_path_split ) for raw_path_split  in raw_paths_split ]

        p_splits = np.array([self.p_split(alpha , len(path_split ), beta ) for path_split  in paths_split ])
        sum_p_splits = sum(p_splits)
        p_splits_mean = p_splits.mean()
        
        N_Ti = self.get_num_operation(name)
        N_Tf = self.get_num_operation(new_name)
        
        B_tree_prune  =(p_prunes_mean/p_splits_mean) * ((N_Ti*prob).sum()/((N_Tf*prob).sum())) * (prob[0] / prob[1]) * ( 1 - self.p_split(alpha , len(path_original)+1, beta ))**(-2)
        
        #Cambiamos el Nombre
        self.name = self.extract(name,2,rand_prune_index)
        return B_tree_prune



    def change_MC2(self):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """

        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
        
        name = self.name
        #Elegimos nodo interno aleatorio para change
        matches = re.finditer("([^X0])",name)
        matches = list(matches)
        # Cambiar con respecto a la altura
        rand_change_match_index=np.random.choice(range(len(matches)))
        rand_change_index=matches[rand_change_match_index].start()
        raw_path = name[:rand_change_index]
        path = self.get_path(raw_path)

        
        tree = self.root
        while  len(path)>0:
            order = path[0]
            if order == "1":
                tree = tree["left"]
            if order == "2":
                tree = tree["right"]
            path = path[1:]

        data = np.concatenate(tree["groups"],axis=0)
        class_labels = np.unique(data[:,-1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        idx = np.random.choice(range(data.shape[1]-1))
        val_aux = np.random.choice(np.unique(data[:,idx]))
        tree['index'] = idx
        tree['val'] = val_aux


    def update_tree(self, tree, data_prev=None):
        """
        given a tree dict anotation, with new decision rules
        return (bool , tree) where bool if feasuble tree and
        tree with correct grupes.

        """
        # Check if tree is a leaf
        
        if not isinstance(tree,dict):
            if data_prev is None:
                return True, tree
            else:
                return True, self.terminal_node(data_prev)

        if data_prev is None:
            data = np.concatenate(tree["groups"],axis=0)
        else:
            data = data_prev

        depth = tree["depth"]
        idx = tree['index']
        val_aux = tree['val']
        groups = self.split(idx, val_aux, data)

        if len(groups[0])==0 or len(groups[1])==0:
            return False, None 
        left_node = tree["left"]
        right_node = tree["right"]
        
        bool_left, tree_left = self.update_tree(left_node, groups[0])  
        bool_right, tree_right = self.update_tree(right_node, groups[1])
        n_ik = self.compute_n_ik(groups)
        tree["n_ik"] = n_ik
        

        # Si alguno es falso returnar que no se pudo (False, None)
        # If any is False return (False, None) (We cant change)
        if not (bool_left and bool_right):
            return False, None
        
        tree_aux = copy.deepcopy(tree)

        tree_aux["groups"] = groups
        tree_aux["left"] = tree_left
        tree_aux["right"] = tree_right

        return True, tree_aux




    def change_MC3(self, n=0):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        self.change_MC2()
        condition , tree = self.update_tree(self.root)
        if n>10:
            #print("Se intento mucho")
            return None

        if condition :
            self.root = tree
            return 1 # Valor de B = q(Ti,Ti+1)P(Ti+1)/q(Ti+1,Ti)P(Ti) 
        else:
            seed = self.seed
            if not seed is None:
                self.seed = np.random.get_state()[1][0]
            self.change_MC3(n+1)


    def swap_MC2(self):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """

        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
            
        name = self.name
        #Elegir no interno no pruneable (tiene como hijo al menos un nodo interno)
        matches1 = re.finditer("10([0X])",name)
        matches1 = list(matches1)

        matches2 = re.finditer("([^0X])",name)
        matches2 = list(matches2)
        matches1_index = [match.start() for match in matches1]
        matches2_index = [match.start() for match in matches2]
        matches3_index = [x for x in matches2_index if x not in matches1_index]
        if len(matches3_index)==0:
            print("No swap posible")
            return None
        # Cambiar con respecto a la altura
        rand_change_index = np.random.choice(matches3_index)
        raw_path = name[:rand_change_index]
        path = self.get_path(raw_path)

        
        tree_aux = copy.deepcopy(self.root)
        tree = tree_aux
        while  len(path)>0:
            order = path[0]
            if order == "1":
                tree = tree["left"]
            if order == "2":
                tree = tree["right"]
            path = path[1:]

        idx = tree['index'] 
        val_aux = tree['val']

        tree_left = tree["left"]
        tree_right = tree["right"]
        if isinstance(tree_left,dict) and isinstance(tree_right,dict):
            #Falta agregar que si los dos hijos tienen la misma regla, cambiar ambos hijos
            if np.random.randint(2) == 0:
                tree['index'], tree_left['index'] = tree_left['index'], tree['index']
                tree['val'], tree_left['val'] = tree_left['val'], tree['val']
            else:
                tree['index'], tree_right['index'] = tree_right['index'], tree['index']
                tree['val'], tree_right['val'] = tree_right['val'], tree['val']

        elif isinstance(tree_left,dict):
            tree['index'], tree_left['index'] = tree_left['index'], tree['index']
            tree['val'], tree_left['val'] = tree_left['val'], tree['val']

        elif isinstance(tree_right,dict):
            tree['index'], tree_right['index'] = tree_right['index'], tree['index']
            tree['val'], tree_right['val'] = tree_right['val'], tree['val']

        return tree_aux
    

    def swap_MC3(self,n=0):
        """
         split information using Montecalo
        :param data:
        :return best_split result dict:
        """
        tree_aux = self.swap_MC2()
        if not tree_aux is None:
            condition , tree = self.update_tree(tree_aux)
        else: 
            condition = False
        if n>10:
#            print("Se intento mucho [Swap]")
            return None

        if condition :
            self.root = tree
#            print(f"{n+1} intentos ")
            return 1 # Valor de B = q(Ti,Ti+1)P(Ti+1)/q(Ti+1,Ti)P(Ti) 
        else:
            seed = self.seed
            if not seed is None:
                self.seed = np.random.get_state()[1][0]
            self.swap_MC3(n+1)



    def split_branch(self, node, depth):
        """
        recursively split the data and
        check for early stop argument based on self.max_depth and self.min_splits
        - check if left or right groups are empty is yess craete terminal node
        - check if we have reached max_depth early stop condition if yes create terminal node
        - Consider left node, check if the group is too small using min_split condition
            - if yes create terminal node
            - else continue to build the tree
        - same is done to the right side as well.
        else
        :param node:
        :param depth:
        :return:
        """
        left_node , right_node = node['groups']
        #del(node['groups'])

        if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
            node['left'] = self.terminal_node(left_node + right_node)
            node['right'] = self.terminal_node(left_node + right_node)
            return

        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return

        if len(left_node) <= self.min_splits:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'],depth + 1)


        if len(right_node) <= self.min_splits:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'],depth + 1)

    def split_branch_MC(self, node, depth =1):
        """
        recursively split the data and
        check for early stop argument based on self.max_depth and self.min_splits
        - check if left or right groups are empty is yess craete terminal node
        - check if we have reached max_depth early stop condition if yes create terminal node
        - Consider left node, check if the group is too small using min_split condition
            - if yes create terminal node
            - else continue to build the tree
        - same is done to the right side as well.
        else
        :param node:
        :param depth:
        :return:
        """
        left_node , right_node = node['groups']
        del(node['groups'])

        if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
            node['left'] = self.terminal_node(left_node + right_node)
            node['right'] = self.terminal_node(left_node + right_node)
            return

        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return

        if len(left_node) <= self.min_splits:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'],depth + 1)


        if len(right_node) <= self.min_splits:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'],depth + 1)

    def build_tree(self):
        """
        build tree recursively with help of split_branch function
         - Create a root node
         - call recursive split_branch to build the complete tree
        :return:
        """
        self.root = self.best_split(self.train_data)
        self.split_branch(self.root, 1)
        return self.root

    def build_tree_MC(self):
        """
        build tree recursively with help of split_branch function
         - Create a root node
         - call recursive split_branch to build the complete tree
        :return:
        """
        self.root = self.split_MC(self.train_data)
        return self.root

    

    def _predict(self, node, row):
        """
        Recursively traverse through the tress to determine the
        class of unseen sample data point during prediction
        :param node:
        :param row:
        :return:
        """
        if row[node['index']] < node['val']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']

        else:
            if isinstance(node['right'],dict):
                return self._predict(node['right'],row)
            else:
                return node['right']

    def predict(self, test_data):
        """
        predict the set of data point
        :param test_data:
        :return:
        """
        self.predicted_label = np.array([])
        for idx in test_data:
            self.predicted_label = np.append(self.predicted_label, self._predict(self.root,idx))

        return self.predicted_label
    


    def write_nodes(self, dic, node=0):
        text = ""
        if not isinstance(dic,dict):
            return text

        left_node = self.nodes[-1] + 1
        self.nodes.append(left_node)
        text +=  "{} -> {} [color=\"green\"];\n".format(node, left_node)
        text += self.write_nodes(dic["left"], left_node)

        right_node = self.nodes[-1] + 1
        self.nodes.append(right_node)
        text +=  "{} -> {} [color=\"yellow\"];\n".format(node, right_node)
        text += self.write_nodes(dic["right"], right_node)
        return text

    def write_dot(self):
        dic = self.root
        self.nodes = [0]
        text = 'digraph Tree {\nnode [shape=box, style="filled", color="skyblue"] ;\n'
        text += self.write_nodes(dic)
        text += "}"
        return text



    def write_nodes2(self, dic, feature_names, label_names, node=0):
        text = ""
        if not isinstance(dic,dict):
            return text

        pre_text = ('{}  [label="{} <= {}\\nsamples = ',
                '{}\\nvalue = {}\\nclass = {}", fillcolor={}];\n')

        pre_text2 = ('{}  [label="samples = {}',
                '\\nvalue = {}\\nclass = {}", fillcolor={}];\n')

        value = sum(dic["n_ik"])
        text +=  "".join(pre_text).format(node, feature_names[dic["index"]],
                                         dic["val"], sum(value), value, label_names[np.argmax(value)], "orange")

        #if isinstance(dic["left"], dict) and isinstance(dic["right"], dict) :
        left_node = self.nodes[-1] + 1
        self.nodes.append(left_node)
        text +=  "{} -> {} [color=\"green\"];\n".format(node, left_node)

        left_dict = dic["left"]
        if isinstance(left_dict, dict): 
            text += self.write_nodes2(left_dict, feature_names, label_names, left_node)
        else:
            value = dic["n_ik"][0]
            text +=  "".join(pre_text2).format(left_node,sum(value), value,
                                          label_names[np.argmax(value)], "orange")

        right_node = self.nodes[-1] + 1
        self.nodes.append(right_node)
        text +=  "{} -> {} [color=\"yellow\"];\n".format(node, right_node)

        right_dict = dic["right"]
        if isinstance(right_dict, dict): 
            text += self.write_nodes2(right_dict, feature_names, label_names, right_node)
        else:
            value = dic["n_ik"][1]
            text +=  "".join(pre_text2).format(right_node,sum(value), value,
                                          label_names[np.argmax(value)], "orange")
        return text

    def write_dot2(self):
        dic = self.root
        feature_names = self.feature_names
        label_names = self.label_names
        self.nodes = [0]
        text = 'digraph Tree {\nnode [shape=box, style="filled", color="skyblue"] ;\n'
        text += self.write_nodes2(dic, feature_names, label_names)
        text += "}"
        return text
    


    def write_nodes3(self, dic, feature_names, label_names, colors, node=0):
        text = ""

        if not isinstance(dic,dict):
            return text

        pre_text = ('{}  [label="{} <= {}\\nsamples = ',
                '{}\\nvalue = {}\\nclass = {}", fillcolor="{}"];\n')

        pre_text2 = ('{}  [label="samples = {}',
                '\\nvalue = {}\\nclass = {}", fillcolor="{}"];\n')


        value = sum(dic["n_ik"])
        total = sum(value)
        cat_index = np.argmax(value)
        pre_color = colors[cat_index]
        alpha = 2*value[cat_index]/total-1
        color = self.color_alpha(alpha, pre_color)
        text +=  "".join(pre_text).format(node, feature_names[dic["index"]],
                                         dic["val"], total, value, label_names[np.argmax(value)], color)

        #if isinstance(dic["left"], dict) and isinstance(dic["right"], dict) :
        left_node = self.nodes[-1] + 1
        self.nodes.append(left_node)
        text +=  "{} -> {} [color=\"green\"];\n".format(node, left_node)

        left_dict = dic["left"]
        if isinstance(left_dict, dict): 
            text += self.write_nodes3(left_dict, feature_names, label_names, colors, left_node)
        else:
            value = dic["n_ik"][0]
            total = sum(value)
            cat_index = np.argmax(value)
            pre_color = colors[cat_index]
            alpha = 2*value[cat_index]/total-1
            color = self.color_alpha(alpha, pre_color)
            text +=  "".join(pre_text2).format(left_node,total, value,
                                          label_names[np.argmax(value)], color)

        right_node = self.nodes[-1] + 1
        self.nodes.append(right_node)
        text +=  "{} -> {} [color=\"yellow\"];\n".format(node, right_node)

        right_dict = dic["right"]
        if isinstance(right_dict, dict): 
            text += self.write_nodes3(right_dict, feature_names, label_names, colors, right_node)
        else:
            value = dic["n_ik"][1]
            total = sum(value)
            cat_index = np.argmax(value)
            pre_color = colors[cat_index]
            alpha = 2*value[cat_index]/total-1
            color = self.color_alpha(alpha, pre_color)
            text +=  "".join(pre_text2).format(right_node,total, value,
                                          label_names[np.argmax(value)], color)


        return text

    def write_dot3(self):
        self.nodes = [0]
        dic = self.root
        feature_names = self.feature_names
        label_names = self.label_names
        colors = self.colors_rgb
        text = 'digraph Tree {\nnode [shape=box, style="filled", color="black"] ;\n'
        text += self.write_nodes3(dic, feature_names, label_names, colors)
        text += "}"
        return text

    
    def graph_structure(self):
        return graphviz.Source(self.write_dot(), format="png")
    
    def graph_info(self):
        return graphviz.Source(self.write_dot2(), format="png")
    
    def graph_desiciontree(self):
        return graphviz.Source(self.write_dot3(), format="png")
    
    def save_img(self, dir0):
        graph = self.graph_desiciontree()
        graph.render(dir0,cleanup=True)
        
    def save_img_hd(self, dir0):
        pydot_graph = pydotplus.graph_from_dot_data(self.write_dot3())
        pydot_graph.set_size('"20,20!"')
        pydot_graph.write_png(dir0 + ".png")



class BayesianDecisionTree(object):
    """
    Class to create decision tree model (CART)
    """
    def __init__(self, prob= [0.25,0.25,0.25,0.25], alpha = 0.5, beta = 0.5 , dynamic_prob = False, random_seed = None, gamma_t = None):
        self.alpha = alpha
        self.beta = beta
        self.prob = prob
        self.path = []
        self.iteration = 0
        self.choices = []
        self.chosen = []
        self.feature_names = None
        self.labels_names = None
        self.dynamic_prob = []
        self.dynamic_gamma = []
        self.seed = random_seed
        self.gamma_t = gamma_t
    
    def copy(self):
        new_BayesianTree = BayesianDecisionTree()
        new_BayesianTree.alpha = self.alpha
        new_BayesianTree.beta = self.beta
        new_BayesianTree.prob = self.prob
        new_BayesianTree.path = self.path
        new_BayesianTree.iteration = self.iteration
        new_BayesianTree.choices = self.choices
        new_BayesianTree.chosen = self.chosen
        new_BayesianTree.feature_names = self.feature_names
        new_BayesianTree.labels_names = self.labels_names 
        new_BayesianTree.dynamic_prob = self.dynamic_prob
        new_BayesianTree.seed = self.seed
        new_BayesianTree.gamma = self.gamma
        return new_BayesianTree

    def fit(self, n,X,Y, alpha_prior = None):
        alpha = self.alpha
        beta = self.beta
        gamma_t = self.gamma_t
        
        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
        
        if self.iteration == 0:
            tree = DecisionTree(alpha, beta, random_seed = seed)
            tree.fit_MC(X ,Y ,alpha_prior)
            self.iteration +=1
        else:
            tree = self.path[-1]
        
        
        self.alpha_prior = tree.alpha_prior
        
        
        operation = ["grow", "prune", "change", "swap"]
        
        U = np.random.uniform(size = n)
        V = np.random.uniform(size = n)
        self.path.append(tree)
        self.choices.append("start")
        self.chosen.append(True)
        
        for i in range(n):
            #choice = np.random.choice(operation)
            
            new_tree = tree.copy()
            leaves = new_tree.leaves_num()

            gamma_norm = (1-gamma_t(i+1)*(leaves))
            gamma_split = (1-gamma_t(i+1)*(leaves+1))
            gamma_prune = (1-gamma_t(i+1)*(leaves-1))
            if gamma_prune<=0 :
                dynamic_gamma = np.array([0,2,1,1])
                self.dynamic_gamma.append(-1)
            elif gamma_split <=0:
                dynamic_gamma = np.array([0,1,1,1])
                self.dynamic_gamma.append(0)
            else:
                dynamic_gamma = np.array([1,1,1,1])
                self.dynamic_gamma.append(1)

            
            zeros = np.zeros(4)
            dynamic_prob = new_tree.get_num_operation()
            self.dynamic_prob.append(list(dynamic_prob))
            prob_aux = np.array(self.prob) * dynamic_prob * dynamic_gamma
            prob = prob_aux/(prob_aux.sum())
            
            p = 0
            for j, p_aux in enumerate(prob):
                p += p_aux
                if V[i] <= p:
                    choice = operation[j]
                    break
            
            #print("eleccion:", choice)
            if choice == "grow":
                B = new_tree.split_MC4()
                C = gamma_split / gamma_norm
                    
            if choice == "prune":
                B = new_tree.prune_MC3()
                if gamma_norm<=0 :
                    C = np.inf
                else:
                    C = gamma_prune / gamma_norm
            if choice == "change":
                B = new_tree.change_MC3()
                C = 1
            if choice == "swap":
                B = new_tree.swap_MC3()
                C = 1

            P_Y_tree = tree.P_YX()
            P_Y_new_tree = new_tree.P_YX()
            

            if not (B is None):
#                print(f"prueba : {P_Y_new_tree - P_Y_tree}")
                A = np.min([1, B * C * np.exp(P_Y_new_tree - P_Y_tree)])
                if U[i] <= A:
                    tree = new_tree
                    self.chosen.append(True)
                else:
                    self.chosen.append(False)
            else:
                self.chosen.append(False)
            self.choices.append(choice)
            self.path.append(tree)
            
        self.iteration += n
        
        
    def fit_batch(self, n, batches, X, Y, alpha_prior = None):
        alpha = self.alpha
        beta = self.beta
        prob = self.prob

        seed = self.seed
        if not seed is None:
            np.random.seed(seed)
            
        if self.iteration == 0:
            tree = DecisionTree(alpha, beta, random_seed=seed)
            tree.fit_MC(X ,Y ,alpha_prior)
        else:
            tree = self.path[-1]
        
        self.alpha_prior = tree.alpha_prior
        
        operation = ["grow", "prune", "change", "swap"]
        
        U = np.random.uniform(size = n * batches)
        V = np.random.uniform(size = n * batches)
        self.path.append(tree)
        self.choices.append("start")
        self.chosen.append(True)
        
        for i in range(n):
            Beta_total =  1
            new_tree = tree.copy()
            choices = []
            for k in range(batches):
                #choice = np.random.choice(operation)
                p = 0
                for j, p_aux in enumerate(prob):
                    p += p_aux
                    if V[i*batches + k] <= p:
                        choice = operation[j]
                        break
                        

                if choice == "grow":
                    B = new_tree.split_MC4()
                    if not(B is None):
                        Beta_total *= B*prob[0]/prob[1]
                if choice == "prune":
                    B = new_tree.prune_MC3()
                    if not(B is None):
                        Beta_total *= B*prob[1]/prob[0]

                if choice == "change":
                    B = new_tree.change_MC3()
                    if not(B is None):
                        Beta_total *= B
                if choice == "swap":
                    B = new_tree.swap_MC3()
                    if not(B is None):
                        Beta_total *= B
                
                choices.append(choice)
            P_Y_tree = tree.P_YX()
            P_Y_new_tree = new_tree.P_YX()

            if not (Beta_total is None):
#                print(f"prueba : {P_Y_new_tree - P_Y_tree}")
                A = np.min([1, Beta_total* np.exp(P_Y_new_tree - P_Y_tree)])
                if U[i] <= A:
                    tree = new_tree
                    self.chosen.append(True)
                else:
                    self.chosen.append(False)
            else:
                self.chosen.append(False)
            self.choices.append(choices)
            self.path.append(tree)
            
        self.iteration += n * batches
        
        
        
        
        
    def set_labels(self, feature_names,labels_names):
        self.feature_names = feature_names
        self.labels_names = labels_names
        
    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def set_beta(self, beta):
        self.beta = beta
        
    def set_prob(self, prob):
        self.prob = prob

    def set_seed(self, seed):
        self.seed = seed
        
    def render_decisiontrees(self, dir0):
        path = self.path
        largo = len(path)
        largo_str = len(str(largo))
        for i, tree in enumerate(path):
            str_i = str(i)
            len_str_i = len(str_i)
            padding = "0"*(largo_str - len_str_i) + str_i
            if not ((self.feature_names is None) or (self.labels_names is None)):
                tree.set_labels(self.feature_names,self.labels_names)
                tree.save_img_hd(dir0+ "/" + padding)
                
    def tree_changes(self):
        path = np.array(self.path)
        largo = len(path)
        idx_chosen = np.arange(largo)[self.chosen]
        path_change = np.array(path)[idx_chosen]
        return path_change
                
    def render_changes(self, dir0):
        path_change = self.tree_changes()
        largo_2 = len(path_change)
        largo_str = len(str(largo_2))
        
        for i, tree in enumerate(path_change):
            str_i = str(i)
            len_str_i = len(str_i)
            padding = "0"*(largo_str - len_str_i) + str_i
            if not ((self.feature_names is None) or (self.labels_names is None)):
                tree.set_labels(self.feature_names,self.labels_names)
                tree.save_img_hd(dir0+ "/" + padding)
    
    def path_acurracy(self, X, y ):
        path = self.path
        accuracy = []
        for i, tree in enumerate(path):
            accuracy.append(accuracy_score(tree.predict(X), y))
        return accuracy

    def path_info(self):
        # (depth, leafs, inner_node)
        path = self.path
        info = []
        
        for i, tree in enumerate(path):
            info.append(tree.info())
        return np.array(info)
    

