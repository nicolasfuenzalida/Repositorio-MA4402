# MCMC en alineación de segmentos de ADN

## Integrantes:

Felipe Hernández.

Sebastian Bustos.

## Tema principal:

Metropolis-Hasting.

## Resumen:

El proyecto se vincula al paper “DNA motif alignment by evolving a population of Markov Chains” de Chengpeng Bi ([1]). Se desea estudiar la implementación de algoritmos estocásticos para encontrar secuencias de ADN recurrentes, llamadas motifs (Figura 1), las cuales se asocian a alguna función biológica importante en el proceso de transcripción del ADN.

Haciendo el supuesto que los motifs distribuyen de manera distinta a las zonas que no contienen motifs es posible aplicar Metropolis Hasting Sampler (MHS) para buscar zonas de alta probabilidad que estén asociadas a los motifs.

Se trabajarán dos tipos de algoritmos de MCMC. El primero, llamado IMC, consiste en aplicar Metropolis Hasting repetidas veces de manera secuencial e independiente. El segundo, llamado PMC, consiste en aplicar el algoritmo Metropolis Hasting de manera paralela permitiendo intercambio de información entre las cadenas.

## Referencias:

[1] Bi, C. DNA motif alignment by evolving a population of Markov chains. BMC Bioinformatics 10, S13 (2009). https://doi.org/10.1186/1471-2105-10-S1-S13

[2] Castro-Mondragon JA, Riudavets-Puig R, Rauluseviciute I, Berhanu Lemma R, Turchi L, Blanc-Mathieu R, Lucas J, Boddie P, Khan A, Manosalva P´erez N, Fornes O, Leung TY, Aguirre A, Hammal F, Schmelter D, Baranasic D, Ballester B, Sandelin A, Lenhard B, Vandepoele K, Wasserman WW, Parcy F, and Mathelier A JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles Nucleic Acids Res.
