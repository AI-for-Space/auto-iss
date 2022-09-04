# Aprendizaje por refuerzo para acoplar de forma autónoma una nave espacial con la Estación Espacial Internacional

Author: Pérez Torregrosa, Daniel ([@DanielTorregrosa](https://github.com/DanielTorregrosa))

Advisor: Guzman Alvarez, Cesar Augusto ([@cguz](https://github.com/cguz)) 

---

## Indice

- [Introducción](#introduccion)
- [Contenido](#contenido)

# Introducción

Este repositorio forma parte del trabajo de fin de master "Aprendizaje por refuerzo para acoplar de forma autónoma una nave espacial con la Estación Espacial Internacional" que tiene como propósito utilizar técnicas de aprendizaje por refuerzo o ‘Reinforcement Learning' para el control autónomo de seis grados de libertad del sistema de acoplamiento de la nave Dragon2 de SpaceX a la Estación Espacial Internacional.  Para el entrenamiento y test del algoritmo se ha utilizado un código Python capaz de conectarse con el simulador oficial de la nave que se encuentra en la siguiente dirección: https://iss-sim.spacex.com

Se ha realizado un estudio de distintos papers dedicados a tareas de acoplamiento de naves y satélites, siendo el algoritmo Proximal Policy Optimization (PPO) el más utilizado. Aun así, se ha decidido llevar a cabo el trabajo con una técnica novedosa que asegura tener mejores resultados que PPO, el algoritmo Phasic Policy Gradient (PPG).

Debido a la limitación de recursos computacionales y los pocos datos que proporciona el simulador se ha simplificado el problema utilizando un entorno matemático con estados y movimientos discretos para entrenar dos agentes, uno encargado de la rotación de la nave y otro de su movimiento, para posteriormente ser probados en el simulador oficial.

Se ha conseguido una solución que posee un éxito del 93.8% en la tarea de reorientación y un 65% en la de movimiento. Aunque el porcentaje de éxito no es elevado, los resultados confirman que el algoritmo seleccionado, PPG, ha sido desarrollado e implementado correctamente y es capaz de aprender.


# Contenido

En el repositorio se pueden encontrar dos carpetas correspondientes a los dos modelos desarrollados

1. auto-iss-complex

  En esta carpeta se encuentra el primer modelo desarrollado en el que el agente utiliza el entorno del simulador oficial para aprender.
  
2. auto-iss-demo

  En esta carpeta se encuentran el segundo modelo, que consta de dos agentes (uno encargado de la orientacion de la nave y el otro de su movimiento) con sus respectivos entornos matematicos para ser entrenados y testeaddos.
  
 Todos los agentes son entrenados con la técnica de aprendizaje por refuerzo Phasic Policy Gradient (PPG)

A continuación se observa los resultados obtenidos en el entorno Demo:


https://user-images.githubusercontent.com/61092361/188248518-1efb2faa-d3b3-451e-afdb-6c628319aaf5.mp4



https://user-images.githubusercontent.com/61092361/188248687-bd31f6b7-b7e3-41c3-b883-61e151fe124f.mp4



Los siguientes videos se pueden ver con mayor calidad en:

- https://youtu.be/r-VbQzVu3VQ
- https://youtu.be/Uro4ii1J7p8
