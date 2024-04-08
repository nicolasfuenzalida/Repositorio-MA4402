# RL Cacho

Implementación de Algoritmos de Reinforcement Learning para aprender a jugar Cachos/Dudo.

Se buscará usar Reinforcement Learning (con métodos de MC/Sarsa/Q-Learning) para implementar una IA que aprenda a jugar Cachos/Dudo (juego chileno de dados). En particular, esto se relaciona con los contenidos del curso en la medida de que es un método estocástico aplicado a IA.

Un posible error que puede ocurrir al ejecutar los notebooks es que el código se esté ejecutando con la CPU y no con la GPU. Esto puede hacer que el kernel se reinicie por el exceso de espacio ocupado.

Damos las siguientes recomendaciones para ejecutar el proyecto:

1. Una vez descargada la carpeta de "Código", crear una carpeta llamada "RL-Cacho-main", y almacenar todo dentro de esta.
2. Ejecutar el proyecto desde Google Colab, moviendo la carpeta "RL-Cacho-main" a "Colab Notebooks" desde Google Drive.
3. Una vez dentro del proyecto en Colab, ir a la barra superior, opción "Runtime", seleccionar "Change runtime type".
4. Dentro de esta pestaña, cambiar "Hardware accelerator" a T4 GPU, luego pulsar "Save".
5. En caso de que se ejecute desde otra ruta a la recomendada, cambiar las celdas que indican cambiar la ruta dentro de los notebooks.
6. Si el algoritmo tarda mucho en ejecutar, modificar n_games.
