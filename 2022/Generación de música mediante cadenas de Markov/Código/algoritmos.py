import numpy as np
from matplotlib import pyplot as plt

def randomSample(v):
    """ Recibe un v (numpy array unidimensional) que actúa como densidad de probabilidad.
    Retorna un índice aleatorio de v con la ley dada por sus valores"""
    assert len(v.shape) == 1, f"El vector dado no es unidimensional, tiene forma {v.size}"
    assert np.isclose(np.sum(v), 1), f"El vector dado no está normalizado, sus coordenadas suman {np.sum(v)}"
    u = np.random.random()
    j = 0
    s = 0
    s += v[0]
    while s <= u:
        j += 1
        s += v[j]
    return j


# Cadena de Markov Observable
class OMM:
    """Observable markov model: Clase que encapsula el comportamiento de una cadena de Markov
    """
    def __init__(self, N):
        """ Recibe un entero que será su dimensionalidad y crea un modelo
        de Markov observable."""
        self.N = N  # Cantidad de estados

        # Inicializamos los parámetros como uniformes
        self.lam = (1/N)*np.ones(N)
        self.A = (1/N)*np.ones((N, N))

    def simulate(self, steps):
        """ Recibe un número de pasos y simula esa cantidad de transiciones de la cadena.
        Devuelve un vector de tamaño steps + 1"""
        C = np.zeros(steps + 1)  # Inicializamos la cadena en cero
        x = randomSample(self.lam)  # Elegir un estado de la distribución inicial
        C[0] = x
        for j in range(steps):
            x = randomSample(self.A[x])  # Elegir un estado según la ley de la fila x de A
            C[j+1] = x  # Añadirlo a la cadena
        return C  # Devolver la cadena

    def seqProb(self, seq):
        """ Recibe una secuencia de estados observados de la cadena.
        Retorna la probabilidad de haber observado esa secuencia de estados"""
        assert ((0 <= seq) & (seq < self.N)).all(), "No todos los estados dados en la secuencia son estados válidos de la cadena"
        p = self.lam[seq[0]]
        for i in range(1, len(seq)):
            p *= self.A[seq[i-1], seq[i]]
        return p
    
    def train(self, seq):
        """ Recibe una secuencia de estados observados de la cadena.
        Actualiza self.A para que sea la matriz de transición que mejor explica el comportamiento observado."""
        assert ((0 <= seq) & (seq < self.N)).all(), "No todos los estados dados en la secuencia son estados válidos de la cadena"
        freq = np.zeros((self.N, self.N))
        # Recorro toda la secuencia contando las transiciones
        for i in range(len(seq)-1):
            freq[seq[i], seq[i+1]] += 1
        
        # Ahora normalizamos cada fila
        for i in range(self.N):
            S = np.sum(freq[i])
            if S != 0:
                freq[i] /= S
        self.A = freq


# Cadena de Markov Oculta
class HMM:
    """ Hidden markov model: Clase que encapsula el comportamiento de una cadena oculta de Markov"""
    def __init__(self, N, M):
        """Recibe el número de estados ocultos y de observables. Crea un Modelo de Markov Oculto
        Cuyos parámetros son inicialmente distribuciones de probabilidad uniformes"""
        self.N = N
        self.M = M

        # Inicializamos los parámetros aleatoriamente y normalizamos
        # Para obtener matrices estocásticas.
        self.lam = np.random.rand(self.N)
        self.lam /= np.sum(self.lam)
        self.A = np.random.rand(self.N, self.N)
        self.B = np.random.rand(self.N, self.M)
        for i in range(self.N):
            self.A[i] /= np.sum(self.A[i])
            self.B[i] /= np.sum(self.B[i])

    def simulate(self, steps):
        """ Recibe un número de pasos y simula esa cantidad de pasos del modelo.
        Devuelve un vector de observaciones de tamaño steps + 1"""
        H = np.zeros(steps + 1, dtype=np.int8)  # Inicializamos la cadena en cero
        x = randomSample(self.lam)  # Elegir un estado de la distribución inicial
        H[0] = x
        for j in range(steps):
            x = randomSample(self.A[x])  # Elegir un estado según la ley de la fila x de A
            H[j+1] = x  # Añadirlo a la cadena
        
        # Genial, ahora H es una cadena de estados ocultos. Para cada uno de ellos escogemos una observación
        O = np.zeros(steps + 1)
        for j in range(steps + 1):
            O[j] = randomSample(self.B[H[j]])
        return O  # Devolver la cadena de observaciones

    def forward(self, obs):  
        """ Recibe una secuencia de estados observables de la cadena.
        Retorna la verosimilitud de esa secuencia con los parámetros de la cadena
        Implementando el algoritmo Forward"""
        assert ((obs >= 0) & (obs < self.M)).all(), "Las observaciones dadas no son válidas"
        T = len(obs)
        # Creamos matriz forward de tamaño MxN 
        ForwardMatrix = np.zeros((self.N, T))
        # Inicializamos la matriz 
        ForwardMatrix[:,0] = self.lam * self.B[:,obs[0]]
        # Trabajamos recursión    
        for t in range(1,T):
            ForwardMatrix[:, t] = ((self.A * self.B[:, obs[t]]).T).dot(ForwardMatrix[:, t-1])
        # Entregamos la verosimilitud de la observación        
        return np.sum(ForwardMatrix[:,T-1]), ForwardMatrix
    
    def viterbi(self, obs, states):
        ''' Recibe observaciones y estados, y entrega la cadena óptima de estados (es decir, en el 
        sentido de ser el camino más probable)'''
        T = len(obs)
        ViterbiMatrix = np.zeros([self.N,T])
        Backpointer = np.zeros([self.N,T])
        for s in range(self.N):
            ViterbiMatrix[s,0] =  lam[s]*self.B[s,obs[0]]
            Backpointer[s,0] = -1 # Estado inicial corresponde a distribución inicial 
            # -1 dado que el 0 confunde con el primer estado posible de la cadena 
        for t in range(1,T):
            for s in range(self.N):
                maxVit = []
                for i in range(self.N):
                    maxVit.append(ViterbiMatrix[i,t-1]*self.A[i,s]*self.B[s,obs[t]])   
                ViterbiMatrix[s,t] = max(maxVit)     
                Backpointer[s,t] = np.argmax(maxVit)
        VitFinal = ViterbiMatrix[:,T-1]   
        BestPathProb = max(VitFinal)     
        BestPathPointer = np.argmax(VitFinal)
        # Falta reconstruir camino óptimo de estados
        Camino = np.append(Backpointer[BestPathPointer],BestPathPointer)
        CaminoEstados = []
        for i in Camino:
            if i==-1:
                CaminoEstados.append('Origen')
            else:
                CaminoEstados.append(states[int(i)])                        
        return BestPathProb, CaminoEstados
    
    def backward(self,obs):
        T = len(obs)
        Beta = np.zeros([self.N,T])
        for i in range(self.N):
            Beta[i,T-1] = 1  
        for t in reversed(range(T-1)):
            for i in range(self.N):
                Beta[i,t] = 0
                for j in range(self.N):
                    Beta[i,t] = Beta[i,t] + self.A[i,j]*self.B[j,obs[t+1]]*Beta[j,t+1]
        P = 0
        for j in range(self.N):
            P = P + self.lam[j]*self.B[j,obs[0]]*Beta[0,j]
        return P, Beta

    def baumwelch(self, obs):
        ''' Realiza 1 iteración del algoritmo de Baum Welch, encontrando matrices A_bar y B_bar
        para el entrenamiento del modelo'''
        T = len(obs)

        A_bar = np.zeros([self.N,self.N])
        B_bar = np.zeros([self.N,self.M])
        lam_bar = np.zeros([self.N])

        # Definimos matrices alfa y beta según lo anterior

        Alfa = self.forward(obs)[1]
        Beta = self.backward(obs)[1]
        
        # Creamos Xi_t(i,j)

        Xi = []
        for i in range(T-1):
            Xi.append(np.zeros([self.N,self.N]))
        for t in range(T-1):
            for j in range(self.N):
                for i in range(self.N):
                    P = 0
                    for s in range(self.N):
                        P = P + Alfa[s,t]*Beta[s,t]      
                    Xi[t][i,j] = (Alfa[i,t]*self.A[i,j]*self.B[j,obs[t+1]]*Beta[j,t+1])/P        

        # Cálculo de A_bar

        # Numerador
        for j in range(self.N):
            for i in range(self.N):
                Xi_sum = sum(Xi)
                A_bar[i,j] = Xi_sum[i,j]

        # Denominador
        for j in range(self.N):
            for i in range(self.N):
                Den = 0
                for t in range(T-1):
                    Den = Den + sum(Xi[t][i])
                A_bar[i,j] = A_bar[i,j]/Den    

        # Cálculo de B_bar y lam_bar

        Gamma = np.zeros([self.N,T])

        for t in range(T):
            for j in range(self.N):
                P = 0
                for s in range(self.N):
                    P = P + Alfa[s,t]*Beta[s,t] 
                Gamma[j,t] = Alfa[j,t]*Beta[j,t]/P        

        for k in range(self.M):
            for j in range(self.N):
                S1 = 0
                S2 = 0
                for t in range(T):
                    if obs[t] == k:
                        S1 = S1 + Gamma[j,t]
                    S2 = S2 + Gamma[j,t]
                B_bar[j,k] = S1/S2 

        for i in range(self.N):
            lam_bar[i] = Gamma[i,0]

        # Actualizamos los atributos del modelo con los valores nuevos
        self.A = A_bar
        self.B = B_bar
        self.lam = lam_bar

    def train(self, seq, eps, num, verbose=False):
        """ Recibe una secuencia de observaciones de la cadena, un grado de tolerancia y un número
        máximo de iteraciones. Itera el algoritmo de B-W hasta que o bien la distancia entre las tres
        matrices sucesivas es menor a {eps}, o bien se ha alcanzado el número máximo de iteraciones {num}."""
        counter = 0
        while counter < num:
            counter += 1

            lam_0 = np.copy(self.lam)
            A_0 = np.copy(self.A)
            B_0 = np.copy(self.B)

            self.baumwelch(seq)

            lam_1 = self.lam
            A_1 = self.A
            B_1 = self.B

            dA = np.linalg.norm(A_1 - A_0, ord='fro')
            dB = np.linalg.norm(B_1 - B_0, ord='fro')
            dlam = np.linalg.norm(lam_1 - lam_0, ord=2)
            if verbose:
                print(f"Iteración {counter} | dA = {dA:.2f} | dB = {dB:.2f} | dlam = {dlam:.2f}")

            if max([dA, dB, dlam]) < eps:
                if verbose:
                    print(f"Convergencia alcanzada. El proceso termina a las {counter} iteraciones")
                break
        
        print("Número máximo de iteraciones alcanzado.")
