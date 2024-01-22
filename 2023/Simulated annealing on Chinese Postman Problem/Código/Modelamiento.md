# Modelamiento

Para comenzar a modelar el problema, lo primero es definir los elementos con los cuales trabajaremos, para esto, primero describamos el CPP.


### 1. Chinese Postman Problem y el grafo de configuraciones
Dado un grafo $G = (V,E)$ simple, finito y conexo con una función de pesos $w$ en sus aristas, buscamos un camino cerrado desde un vértice fijo inicial y de peso mínimo, que pase por todas sus aristas. Para esto, notamos que definir una secuencia de aristas como soluciones factibles es algo complejo, dado que son elementos de $E^\mathbb{N}$, como consecuencia, lo que haremos será  considerar un $\textbf{orden de prioridad}$ para pasar por las aristas y así construir nuestro grafo de configuraciones. Esto es, dado $|E|=m$, definimos $s_0 = (e_1,e_2,\dots,e_m)$ como un orden de prioridad por el orden en el cual se deben recorrer todas las aristas del grafo.

- Notamos que para haber factibilidad, debe pasar por todas las aristas, o sea que $e_i\neq e_j$ para todo $i\neq j$ dentro de $s_0$, de tal manera que cada arista de $E$ aparezca de forma única en la secuencia ya que todas tienen que tener prioridad para pasar por ellas.
- Definimos de forma análoga $s_{n+1}$ como el intercambio de $e_i$ con $e_j$ dentro de $s_n$ para algún par $i,j \in [m]$, o sea, $s_{n+1}$ es una permutación entre el $i$-ésimo y el $j$-ésimo elemento de la secuencia $s_n$.
- Diremos que dos ordenes de prioridad $\sigma_1$ y $\sigma_2$ (secuencias) son $\textbf{adyacentes}$ si $\sigma_2$ se puede obtener de $\sigma_1$ permutando dos elementos, claramente con la definición anterior $s_n$ es adyacente de $s_{n+1}$ para todo $n$, y además es una relación simétrica.
- Juntando todas estas ideas anteriores, tenemos que cualquier sucesión $(s_i)_{i\in\mathbb{N}}$ de ordenes de prioridad son todas soluciones factibles si construimos un camino cerrado en cada $s_i$ que parta desde nuestro vértice inicial y pase por las aristas de la secuencia, intentando pasar por ellas en el orden establecido por su prioridad.

Ahora teniendo en cuenta esto, nuestras configuraciones serán todas las posibles permutaciones de $E$, y la adyacencia está dada por lo antes explicado. Ahora para relacionarlo con el CPP, se tiene que para un orden de prioridad $\sigma$, reconstruiremos un camino cerrado que pase por las aristas con un algoritmo llamado $\textbf{Reconstrucción de Caminos}$, dado que el solo tener aristas en una secuencia puede dar para muchos caminos posibles que cumplan el orden de prioridad, pero queremos que siempre se reconstruya de la misma manera, para así poder hacer comparaciones objetivas, y por lo tanto, buscar soluciones al CPP.

Por simplicidad, fijaremos el primer nodo inicial $v_0 = 0$, de tal manera que siempre la arista inicial deba ser adyacente al cero y solo se permutará por aristas que también sean adyacentes al cero.

### 2. Simulated Annealing



Acá es donde entra el Simulated Annealing, donde tendremos una cadena de Markov $X_n$ que represente el orden de prioridad acutal, para esto, primero nos definimos $X_0 = s_0$ como guess inicial de solución, calculamos su respectivo camino en $G$ con el algoritmo de reconstrucción de caminos, y uniformemente dentro de $s_0$ elegimos dos aristas para permutar, generando su vecino $s_1$, que también le reconstruimos su camino dentro de $G$, con esto, podremos tener el peso $w(s_0)$ y $w(s_1)$, con el que a partir de los parámetros elegidos y la temperatura correspondiente a estas dos configuraciones, actualizamos $X_1$ como $s_0$ o bien $s_1$. (ver https://en.wikipedia.org/wiki/Simulated_annealing)

Por lo visto anteriormente, cualquier sucesión de ordenes de configuraciones es siempre factible para el CPP mientras que el algoritmo de reconstrucción de caminos asegure pasar por todas las aristas y volver al vértice inicial. Por lo que, si definimos iterativamente la cadena de la forma anterior dada una cierta sucesión de ordenes de prioridad elegidas aleatoriamente ${s_0,s_1,\dots}$ , toda la cadena $(X_n)_{n\in\mathbb{N}}$ será siempre de caminos factibles.

### 3. Algoritmo de Reconstrucción de Caminos

Dada un orden de prioridad $s=(e_1,\dots,e_m)$, queremos obtener $P_s$ camino factible para el CPP asociado a $s$, para esto, aplicamos lo siguiente:
1. Definimos $P_s = [0]$ como la lista de los vértices por los que recorre en ese orden, posee al cero porque es la suposición inicial que definimos ($e_1$ incidente en $0$)
2. Como $e_1=0v$, $P_s \leftarrow P_s + v$
3. $i \leftarrow 1$
4. Definimos variable VérticeActual $\leftarrow v$
5. Definimos AristaActual $\leftarrow e_i=a_1b_1$ y AristaSiguiente $\leftarrow e_{i+1}=a_2b_2$ (en el caso $i=1: a_1=0, b_1=v$) 
6. Si VérticeActual es incidente en $e_2$, digamos $e_2 = vb_2$, actualizamos $P_s \leftarrow P_s + b_2$ y VérticeActual $\leftarrow b_2$ (por simplicidad $e_i = e_1, e_{i+1}=e_2$, para fijar ideas)
7. Si VérticeActual == $b_1$ y $e_2$ es incidente en $a_1$, digamos $e_2=a_1b_2$, actualizamos $P_s \leftarrow P_s + a_1 + b_2$ y VérticeActual $\leftarrow b_2$ (representando que nos devolvemos y duplicamos)
8. Si no, calcular camino de peso mínimo desde VérticeActual a $a_2$ y $b_2$ y tomar el de mínimo peso entre ellos, llamémoslo $P$, y digamos que llega a $a_2$, actualizar $P_s \leftarrow P_s + P + b_2$ y VérticeActual $\leftarrow b_2$, además de esto, todas las aristas que se hayan tomado en $P$ que todavía no se hayan recorrido en el orden de prioridad se eliminan del orden de prioridad, ya que al ya haber pasado por ellas no hay necesidad de pasar nuevamente.
9. Si no se ha llegado al final del orden de prioridad, $i\leftarrow i+1$ y volver a 5.
10. Conectar VérticeActual con 0, con las mismas reglas que lo anterior.

El algoritmo es correcto para lo que queremos pues:
- Pasa por todas las aristas, puesto que si esto no pasa, quiere decir que alguna de las aristas del orden de prioridad se ignoró, digamos que esta arista es $e$, hay dos casos, el primero es que en algún momento la variable AristaActual vale $e$, si esto pasa, notamos que necesariamente en el paso anterior en cualquiera de los tres casos, en 6., 7. u 8. se pasa necesariamente por ambos vértices de $e$ por lo que esto no es posible. El segundo caso es que AristaActual nunca haya sido $e$, esto quiere decir que es parte de las aristas eliminadas en 8. pero esto también genera una contradicción pues esta arista se añade a $P_s$ pues está dentro del camino $P$.
- Es efectivamente un camino cerrado, es un camino dado que toda la sucesión de vértices que se añaden a $P_s$ son siempre conectados a partir de aristas, lo que genera un recorrido factible. Por otra parte, considerando 1. y 10., necesariamente el camino empieza y termina en 0.




