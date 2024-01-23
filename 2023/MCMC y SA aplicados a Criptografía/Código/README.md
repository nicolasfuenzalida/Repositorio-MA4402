# Simmulated Anneling Aplicado a Criptografía

Este es el github del proyecto dedicado a "Simmulated Anneling" en el contexto de criptografía. En este github hay 4
grandes grupos bien definidos

1. Cifrado Monoalfabético:
    - Sólo consiste en la función CifradoMonoalfabetico.py, el cual trae todas las funcionalidades necesarias para esta aplicación, estas corresponden a:
        - Una función que crea una clave aleatoria.
        - Una función que crea la estadística de letras desde caracteres de una base de strings.
        - Una función que crea una lista de digramas que se pueden crear de una base de strings.
        - Una función que crea estadísticas de digramas desde caracteres de una base de strings.
        - Una funcion que permite ordenar una lista de digramas que no está por orden lexicográfico (se usa única y exclusivamente para ordenar las estadísticas del idioma inglés).
        - Una función que da la función fitness de Carter-Magoc.
        - Dos funciones que calculan la energía de un texto, que es 1-fitnes del texto, las funciones calculan en base al texto completo en un caso, y en otro conociendo las estadísticas del mismo. Las dos serán útiles para la implementación posterior.
        - Una función que implementa el truco para optimizar el S.A. del caso monoalfabético, que consiste en ver qué digramas cambian con un cambiazo de dos letras en la biyección usada para el alfabeto.
        - Dos funciones de simmulated anneling, una sin el truco para optimizar, y la otra con lo mismo.
        Posteriomente tiene aplicaciones de simmulated anneling.
2. Cifrado de Permutación
    - Sólo consiste en la función Cifrado_permutacion.py, el cual trae todas las funcionalidades necesarias para esta aplicación.
        - Funciones para generar vecinos de nodos en el grafo regular
        - Funcion que permite "alfabetizar" un texto bajo el contexto del problema
        - Estadisticas de digramas en el texto "A Case Of Identity"
        - Funcion que geneara una clave inicial
        - Funcion que implementa S.A
        - Funcion tester0, que permite implementar una cantidad arbitraria de veces S.A
        - 4 Funciones de energia, que son las de la presentacion
        - Parametros de temperatura inicial y factor de reduccion
3. (Intento de) Cifrado Polialfabético:
     Acá hay 2 archivos, que consisten en:
    - cifrado_nletras.py este archivo consiste en un compendio de funciones auxiliares para el salto al cifrado polialfabético.
        - Una función que crea un listado de n-gramas desde una base de caracteres.
        - Una función que crea una biyección en un espacio de n-gramas a partir de una semilla, es decir, es aleatorio.
        - Una función que cifra en base a una biyección en una lista.
        - Dos funciones que calculan las estadísticas de n-gramas separando por palabras, una sin optimizar y una con optimizar
        - Una función que devuelve una biyección vecina a otra, considerando vecindad de biyecciones como que hay una transposición para pasar de una a otra.
        - Una función para las estadísticas optimizadas, que es obtener las palabras de un string largo, considerando palabras como substrings maximales de elementos de la base.
        - Un intento de implementación de estadísticas para n-1 sabiendo n, NO FUNCIONA
    - simmulated_anneling_nletras.py este archivo unicamente implementa, de forma no muy testeada, el decifrado del caso polialfabético. Trae solo la implementación de esto
4. Archivos base de uso común:
    Acá hay más archivos, y consisten en los siguientes:
   - cifrador.py Esta función trae funciones relativas a la codificación de mensajes, funciones que permiten cifrar el texto entero, tanto con el cifrado de permutación como el monoalfabético
   - f1_energía.py Esta función fue una implementación temprana para el contador empírico de digramas, se descartó posteriormente por ser ineficiente.
   - frec.py Trae las frecuencias de digramas y de letras, desde las fuentes respectivas, en formato de lista.
   - FuncionesAuxiliares.py Trae funciones pequeñas útiles para la implementación del cifrado Monoalfabético, en concreto funciones que permite determinar si dos claves son vecinas (es un poco más general que eso, ya que permite ver en cuantos caracteres difieren, potencialmente útil para extensiones de índole que no se exploraron en este proyecto).
   - imagen_de_grafico_diagramas.py Crea el gráfico usado en la presentación para la frecuencia relativa de aparición de distintos digramas en el idioma inglés.
   - Lector.py Archivo con funciones que permiten importar de forma rapida, como strings, los archivos de texto utilizados en el proyecto.

Además de todos estos archivos con funciones, hay dos carpetas con material anexo útil para el proyecto:
1. Textos: Son cuentos de "Las Aventuras de Sherlock Holmes" en inglés, en concreto consiste de los siguientes relatos:
   - "A case of Identiy"
   - "A Scandal in Bohemia"
   - "Speckled Band"
   - "The Boscombe Valley Mystery"
   - "The Five Orange Pips"
   - "The Red Headed League"
  Estos relatos se usaron de modo de prueba para los métodos del proyecto
2. Referencias: Trae los *papers* de referencia para el proyecto, hay uno adicional que sirvió de inspiración inicial, sin embargo no se utilizó con posterioridad
   - Carter-Magoc (2007)
   - Dimoski-Glimorovski (2003)
   - Veroutis-Fajardo (2021) No utilizado más allá de inspiración inicial.
